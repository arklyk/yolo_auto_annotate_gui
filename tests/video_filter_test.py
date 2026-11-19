"""Check filtering, dataset export, cancellation and MP4 replacement without model weights."""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules["ultralytics"] = SimpleNamespace(YOLO=lambda _: FakeModel())

import video_inference  # noqa: E402
from video_inference import VideoConfig, run_videos  # noqa: E402


class FakeBoxes:
    def __init__(self, detected: bool):
        self.cls = np.array([0] if detected else [])
        self.xywhn = np.array([[0.5, 0.5, 0.25, 0.25]] if detected else []).reshape(-1, 4)

    def __len__(self):
        return len(self.cls)


class FakeModel:
    task = "detect"
    names = {0: "object"}

    def predict(self, frame, **_):
        detected = bool(frame.mean() > 100)
        plotted = frame.copy()
        plotted[:10, :10] = (0, 0, 255)
        return [SimpleNamespace(boxes=FakeBoxes(detected), plot=lambda: plotted)]


def make_video(path: Path, bright_frames: set[int], count: int = 10) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 48))
    assert writer.isOpened()
    for index in range(1, count + 1):
        value = 255 if index in bright_frames else 0
        writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    writer.release()


def frame_count(path: Path) -> int:
    reader = cv2.VideoCapture(str(path))
    try:
        assert reader.isOpened(), path
        return int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        reader.release()


def main() -> None:
    with TemporaryDirectory() as folder:
        root = Path(folder)
        model = root / "model.pt"
        model.touch()
        mixed = root / "mixed.mp4"
        empty = root / "empty.mp4"
        make_video(mixed, {2, 3, 7, 8})
        make_video(empty, set())

        summary = run_videos(VideoConfig(model, [mixed, empty], root / "filtered"))
        assert (summary.succeeded, summary.failed, summary.kept_frames, summary.saved_images) == (2, 0, 4, 2)
        assert frame_count(mixed) == 10
        outputs = list(summary.run_dir.glob("*.mp4"))
        assert len(outputs) == 1 and frame_count(outputs[0]) == 4
        images = sorted((summary.run_dir / "dataset" / "images").glob("*.jpg"))
        labels = sorted((summary.run_dir / "dataset" / "labels").glob("*.txt"))
        assert len(images) == len(labels) == 2
        assert [p.stem for p in images] == [p.stem for p in labels]
        assert [int(p.stem.rsplit("_", 1)[1]) for p in images] == [2, 7]
        assert all(np.all(cv2.imread(str(path))[0, 0] > 200) for path in images)
        assert all(label.read_text().strip() == "0 0.500000 0.500000 0.250000 0.250000" for label in labels)
        assert (summary.run_dir / "dataset" / "classes.txt").read_text().strip() == "object"

        overwritten = run_videos(VideoConfig(
            model, [mixed], root / "overwrite", save_dataset=False, overwrite_original=True,
        ))
        assert overwritten.succeeded == 1 and frame_count(mixed) == 4, (
            overwritten, (overwritten.run_dir / "errors.log").read_text(encoding="utf-8")
        )
        backups = list(root.glob("mixed.original_*.mp4"))
        assert len(backups) == 1 and frame_count(backups[0]) == 10
        assert frame_count(next(overwritten.run_dir.glob("*.mp4"))) == 4

        rollback = root / "rollback.mp4"
        make_video(rollback, {2, 3, 7, 8})
        real_replace = video_inference.os.replace

        def fail_second_rename(source: Path, destination: Path) -> None:
            if Path(source).name.startswith(".rollback.filtered_"):
                raise OSError("simulated replacement failure")
            real_replace(source, destination)

        with patch.object(video_inference.os, "replace", side_effect=fail_second_rename):
            failed = run_videos(VideoConfig(
                model, [rollback], root / "failed_replace", save_dataset=False,
                overwrite_original=True,
            ))
        assert failed.failed == 1 and frame_count(rollback) == 10
        assert not list(root.glob("rollback.original_*.mp4"))
        assert not list(root.glob(".rollback.filtered_*.mp4"))

        no_detection = run_videos(VideoConfig(
            model, [empty], root / "empty_replace", save_dataset=False,
            overwrite_original=True,
        ))
        assert no_detection.succeeded == 1 and frame_count(empty) == 10
        assert not list(root.glob("empty.original_*.mp4"))

        cancel = False

        def progress(_done: int, _total: int, message: str) -> None:
            nonlocal cancel
            if "已检测 1 帧" in message:
                cancel = True

        cancelled = run_videos(VideoConfig(
            model, [mixed], root / "cancel", save_interval_seconds=0,
        ), progress, lambda: cancel)
        assert cancelled.cancelled and cancelled.succeeded == 0
        assert not list(cancelled.run_dir.glob("*.mp4"))
        assert not list((cancelled.run_dir / "dataset" / "images").glob("*.jpg"))
        assert not list((cancelled.run_dir / "dataset" / "labels").glob("*.txt"))

    print("PASS: filtered video, dataset interval, empty input, safe replacement, cancellation")


if __name__ == "__main__":
    main()
