"""Batch video detection and annotated MP4 export."""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import sys
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v", ".webm"}


@dataclass
class VideoConfig:
    model_path: Path
    videos: list[Path]
    output_root: Path
    confidence: float = 0.25
    image_size: int = 640
    save_dataset: bool = True
    skip_empty: bool = True
    save_interval_seconds: float = 1.0
    overwrite_original: bool = False


@dataclass
class VideoSummary:
    run_dir: Path
    total: int
    succeeded: int
    failed: int
    box_count: int
    cancelled: bool
    kept_frames: int = 0
    saved_images: int = 0


def _replace_mp4(original: Path, result: Path, expected_frames: int) -> Path:
    """Keep a backup and replace an MP4 only after the finished copy is readable."""
    import cv2

    temporary = original.with_name(f".{original.stem}.filtered_{uuid.uuid4().hex}.mp4")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = original.with_name(f"{original.stem}.original_{stamp}.mp4")
    try:
        shutil.copyfile(result, temporary)
        reader = cv2.VideoCapture(str(temporary))
        try:
            readable = reader.isOpened()
            actual_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) if readable else 0
            first_frame, _ = reader.read() if readable else (False, None)
        finally:
            reader.release()
        if not first_frame or actual_frames != expected_frames:
            raise RuntimeError("筛选后的视频验证失败，原视频未替换")
        os.replace(original, backup)
        try:
            os.replace(temporary, original)
        except Exception:
            os.replace(backup, original)
            raise
        return backup
    finally:
        temporary.unlink(missing_ok=True)


def run_videos(
    config: VideoConfig,
    progress: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> VideoSummary:
    import cv2
    from ultralytics import YOLO

    model_path = config.model_path.expanduser().resolve()
    output_root = config.output_root.expanduser().resolve()
    videos = [path.expanduser().resolve() for path in config.videos]
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not videos:
        raise ValueError("请至少选择一个视频")
    if len(set(videos)) != len(videos):
        raise ValueError("视频列表包含重复文件")
    for video in videos:
        if not video.is_file() or video.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"无效的视频文件: {video}")
    if not math.isfinite(config.confidence) or not 0 < config.confidence <= 1:
        raise ValueError("置信度必须在 (0, 1] 内")
    if config.image_size < 32 or config.image_size % 32:
        raise ValueError("推理尺寸必须是 32 的倍数")
    if not math.isfinite(config.save_interval_seconds) or config.save_interval_seconds < 0:
        raise ValueError("数据集保存间隔不能为负数")
    if config.overwrite_original and any(video.suffix.lower() != ".mp4" for video in videos):
        raise ValueError("替换原视频目前仅支持 MP4 输入")
    if config.overwrite_original and not config.skip_empty:
        raise ValueError("替换原视频时必须启用跳过无检测画面")

    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError(f"当前仅支持目标检测模型，实际模型任务为: {model.task}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / datetime.now().strftime("video_run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(exist_ok=False)
    error_log = run_dir / "errors.log"
    error_log.write_text("", encoding="utf-8")
    images_dir = run_dir / "dataset" / "images"
    labels_dir = run_dir / "dataset" / "labels"
    if config.save_dataset:
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        from yolo_auto_annotate import _model_names
        names = _model_names(model.names)
        (run_dir / "dataset" / "classes.txt").write_text(
            "\n".join(names.get(index, "") for index in range(max(names) + 1)) + "\n",
            encoding="utf-8",
        )
    succeeded = failed = box_count = total_kept = total_saved = 0
    cancelled = False

    for index, video in enumerate(videos, 1):
        if should_cancel and should_cancel():
            cancelled = True
            break
        digest = hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:10]
        readable_name = re.sub(r"[^A-Za-z0-9_-]+", "_", video.stem).strip("_")[:48] or "source"
        target = run_dir / f"video_{index:03d}_{readable_name}_{digest}.mp4"
        capture = cv2.VideoCapture(str(video))
        writer = None
        frames = 0
        kept_frames = 0
        saved_images = 0
        last_saved_frame = None
        video_boxes = 0
        remove_target = False
        generated_files: list[Path] = []
        try:
            if not capture.isOpened():
                raise RuntimeError("无法打开视频；请检查格式或编解码器")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
                raise RuntimeError("无法读取视频尺寸或帧率")
            while True:
                if should_cancel and should_cancel():
                    cancelled = True
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                frames += 1
                result = model.predict(
                    frame, conf=config.confidence, imgsz=config.image_size,
                    verbose=False, save=False,
                )[0]
                boxes = result.boxes
                detected = len(boxes) if boxes is not None else 0
                video_boxes += detected
                if detected or not config.skip_empty:
                    if writer is None:
                        writer = cv2.VideoWriter(
                            str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                        )
                        if not writer.isOpened():
                            raise RuntimeError("无法创建结果视频；请检查输出目录和 MP4 编码支持")
                    writer.write(result.plot())
                    kept_frames += 1
                if config.save_dataset and detected and (
                    last_saved_frame is None
                    or (frames - last_saved_frame) / fps >= config.save_interval_seconds
                ):
                    stem = f"video_{index:03d}_{readable_name}_{digest}_frame_{frames:08d}"
                    image_path = images_dir / f"{stem}.jpg"
                    label_path = labels_dir / f"{stem}.txt"
                    generated_files.extend((image_path, label_path))
                    encoded_ok, encoded = cv2.imencode(".jpg", frame)
                    if not encoded_ok:
                        raise RuntimeError(f"无法编码数据集图片: {image_path}")
                    image_path.write_bytes(encoded.tobytes())
                    label_path.write_text(
                        "".join(
                            f"{int(class_id)} " + " ".join(f"{float(value):.6f}" for value in xywh) + "\n"
                            for class_id, xywh in zip(boxes.cls.tolist(), boxes.xywhn.tolist())
                        ),
                        encoding="utf-8",
                    )
                    saved_images += 1
                    last_saved_frame = frames
                if progress and (frames == 1 or frames % 10 == 0):
                    progress(index - 1, len(videos), f"{video.name}：已检测 {frames} 帧，保留 {kept_frames} 帧")
            if cancelled:
                remove_target = True
                break
            if frames == 0:
                raise RuntimeError("视频没有可读取的画面")
            if writer is not None:
                writer.release()
                writer = None
            if config.overwrite_original and kept_frames:
                capture.release()
                backup = _replace_mp4(video, target, kept_frames)
                if progress:
                    progress(index - 1, len(videos), f"{video.name}：原视频已替换，备份：{backup.name}")
            succeeded += 1
            box_count += video_boxes
            total_kept += kept_frames
            total_saved += saved_images
            if progress:
                progress(index, len(videos),
                         f"{video.name}：检测 {frames} 帧，保留 {kept_frames} 帧，"
                         f"保存 {saved_images} 张数据集图片，{video_boxes} 个检测框")
        except Exception:
            failed += 1
            remove_target = True
            with error_log.open("a", encoding="utf-8") as log:
                log.write(f"视频处理失败: {video}\n{traceback.format_exc()}\n")
            if progress:
                progress(index, len(videos), f"{video.name}：失败，详见 errors.log")
            print(f"视频处理失败: {video}", file=sys.stderr)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            if remove_target:
                target.unlink(missing_ok=True)
                for path in generated_files:
                    path.unlink(missing_ok=True)

    return VideoSummary(run_dir, len(videos), succeeded, failed, box_count, cancelled,
                        total_kept, total_saved)
