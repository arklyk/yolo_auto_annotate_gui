"""Batch video detection and annotated MP4 export."""

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


@dataclass
class VideoSummary:
    run_dir: Path
    total: int
    succeeded: int
    failed: int
    box_count: int
    cancelled: bool


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

    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError(f"当前仅支持目标检测模型，实际模型任务为: {model.task}")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / datetime.now().strftime("video_run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(exist_ok=False)
    error_log = run_dir / "errors.log"
    error_log.write_text("", encoding="utf-8")
    succeeded = failed = box_count = 0
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
        video_boxes = 0
        remove_target = False
        try:
            if not capture.isOpened():
                raise RuntimeError("无法打开视频；请检查格式或编解码器")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0:
                raise RuntimeError("无法读取视频尺寸或帧率")
            writer = cv2.VideoWriter(
                str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError("无法创建结果视频；请检查输出目录和 MP4 编码支持")
            while True:
                if should_cancel and should_cancel():
                    cancelled = True
                    break
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(
                    frame, conf=config.confidence, imgsz=config.image_size,
                    verbose=False, save=False,
                )[0]
                annotated = result.plot()
                writer.write(annotated)
                frames += 1
                video_boxes += len(result.boxes) if result.boxes is not None else 0
                if progress and (frames == 1 or frames % 10 == 0):
                    progress(index - 1, len(videos), f"{video.name}：已处理 {frames} 帧")
            if cancelled:
                remove_target = True
                break
            if frames == 0:
                raise RuntimeError("视频没有可读取的画面")
            succeeded += 1
            box_count += video_boxes
            if progress:
                progress(index, len(videos), f"{video.name}：完成 {frames} 帧，{video_boxes} 个检测框")
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

    return VideoSummary(run_dir, len(videos), succeeded, failed, box_count, cancelled)
