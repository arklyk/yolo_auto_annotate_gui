"""Batch video detection, matched filtered videos and CVAT-ready frame export."""

from __future__ import annotations

import hashlib
import math
import re
import sys
import traceback
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
    save_interval_seconds: float = 1.0


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
    zip_path: Path | None = None


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

    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError(f"当前仅支持目标检测模型，实际模型任务为: {model.task}")
    from yolo_auto_annotate import (
        _label_text, _make_cvat_zip, _model_names, _write_metadata, extract_annotations,
    )
    names = _model_names(model.names)
    mapping = {class_id: class_id for class_id in names}
    thresholds = {class_id: config.confidence for class_id in names}
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / datetime.now().strftime("video_run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(exist_ok=False)
    error_log = run_dir / "errors.log"
    error_log.write_text("", encoding="utf-8")
    images_dir = run_dir / "images" / "train"
    labels_dir = run_dir / "labels" / "train"
    if config.save_dataset:
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
    succeeded = failed = box_count = total_kept = total_saved = 0
    cancelled = False
    image_lines: list[str] = []
    progress_max = len(videos) * 1000

    for index, video in enumerate(videos, 1):
        if should_cancel and should_cancel():
            cancelled = True
            break
        digest = hashlib.sha256(str(video).encode("utf-8")).hexdigest()[:10]
        readable_name = re.sub(r"[^A-Za-z0-9_-]+", "_", video.stem).strip("_")[:48] or "source"
        stem = f"video_{index:03d}_{readable_name}_{digest}"
        raw_target = run_dir / f"{stem}_filtered.mp4"
        annotated_target = run_dir / f"{stem}_annotated.mp4"
        capture = cv2.VideoCapture(str(video))
        raw_writer = annotated_writer = None
        frames = 0
        kept_frames = 0
        saved_images = 0
        last_saved_frame = None
        video_boxes = 0
        remove_outputs = False
        generated_files: list[Path] = []
        video_image_lines: list[str] = []
        try:
            if not capture.isOpened():
                raise RuntimeError("无法打开视频；请检查格式或编解码器")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            estimated_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
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
                annotations = extract_annotations(result, mapping, thresholds)
                detected = len(annotations)
                video_boxes += detected
                if detected:
                    if raw_writer is None:
                        raw_writer = cv2.VideoWriter(
                            str(raw_target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                        )
                        annotated_writer = cv2.VideoWriter(
                            str(annotated_target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
                        )
                        if not raw_writer.isOpened() or not annotated_writer.isOpened():
                            raise RuntimeError("无法创建结果视频；请检查输出目录和 MP4 编码支持")
                    raw_writer.write(frame)
                    annotated_writer.write(result.plot())
                    kept_frames += 1
                if config.save_dataset and detected and (
                    last_saved_frame is None
                    or (frames - last_saved_frame) / fps >= config.save_interval_seconds
                ):
                    frame_stem = f"{stem}_frame_{frames:08d}"
                    image_path = images_dir / f"{frame_stem}.jpg"
                    label_path = labels_dir / f"{frame_stem}.txt"
                    generated_files.extend((image_path, label_path))
                    encoded_ok, encoded = cv2.imencode(".jpg", frame)
                    if not encoded_ok:
                        raise RuntimeError(f"无法编码数据集图片: {image_path}")
                    image_path.write_bytes(encoded.tobytes())
                    label_path.write_text(_label_text(annotations), encoding="utf-8")
                    video_image_lines.append(f"images/train/{image_path.name}\n")
                    saved_images += 1
                    last_saved_frame = frames
                if progress:
                    message = f"第 {index}/{len(videos)} 个视频 {video.name}：已检测 {frames} 帧，保留 {kept_frames} 帧"
                    if estimated_frames > 0:
                        units = min(999, frames * 1000 // estimated_frames)
                        progress((index - 1) * 1000 + units, progress_max, message)
                    else:
                        progress(0, 0, message)
            if cancelled:
                remove_outputs = True
                break
            if frames == 0:
                raise RuntimeError("视频没有可读取的画面")
            succeeded += 1
            box_count += video_boxes
            total_kept += kept_frames
            total_saved += saved_images
            image_lines.extend(video_image_lines)
            if progress:
                progress(index * 1000, progress_max,
                         f"第 {index}/{len(videos)} 个视频完成：{video.name}；检测 {frames} 帧，"
                         f"保留 {kept_frames} 帧，保存 {saved_images} 张图片")
        except Exception:
            failed += 1
            remove_outputs = True
            with error_log.open("a", encoding="utf-8") as log:
                log.write(f"视频处理失败: {video}\n{traceback.format_exc()}\n")
            if progress:
                progress(index * 1000, progress_max,
                         f"第 {index}/{len(videos)} 个视频失败：{video.name}；详见 errors.log")
            print(f"视频处理失败: {video}", file=sys.stderr)
        finally:
            capture.release()
            for writer in (raw_writer, annotated_writer):
                if writer is not None:
                    writer.release()
            if remove_outputs:
                raw_target.unlink(missing_ok=True)
                annotated_target.unlink(missing_ok=True)
                for path in generated_files:
                    path.unlink(missing_ok=True)

    zip_path = None
    if config.save_dataset:
        _write_metadata(run_dir, image_lines, names)
        if total_saved:
            zip_path = _make_cvat_zip(run_dir)
    return VideoSummary(run_dir, len(videos), succeeded, failed, box_count, cancelled,
                        total_kept, total_saved, zip_path)
