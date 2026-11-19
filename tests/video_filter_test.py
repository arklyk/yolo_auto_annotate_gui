"""Check matched filtered videos, CVAT export and frame progress without model weights."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules["ultralytics"] = SimpleNamespace(YOLO=lambda _: FakeModel())

from video_inference import VideoConfig, run_videos  # noqa: E402


class FakeBoxes:
    def __init__(self, detected: bool):
        self.cls = np.array([0] if detected else [])
        self.conf = np.array([0.9] if detected else [])
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


def video_frames(path: Path) -> list[np.ndarray]:
    reader = cv2.VideoCapture(str(path))
    assert reader.isOpened(), path
    try:
        frames = []
        while True:
            ok, frame = reader.read()
            if not ok:
                return frames
            frames.append(frame)
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
        original_bytes = mixed.read_bytes()
        events: list[tuple[int, int, str]] = []

        summary = run_videos(VideoConfig(model, [mixed, empty], root / "filtered"),
                             lambda done, total, message: events.append((done, total, message)))
        assert (summary.succeeded, summary.failed, summary.kept_frames, summary.saved_images) == (2, 0, 4, 2)
        assert mixed.read_bytes() == original_bytes
        assert any(0 < done < 1000 and total == 2000 for done, total, _ in events)
        assert events[-1][0] == events[-1][1] == 2000
        outputs = sorted(summary.run_dir.glob("*.mp4"))
        assert len(outputs) == 2
        raw = next(path for path in outputs if path.stem.endswith("_filtered"))
        annotated = next(path for path in outputs if path.stem.endswith("_annotated"))
        raw_frames, annotated_frames = video_frames(raw), video_frames(annotated)
        assert len(raw_frames) == len(annotated_frames) == 4
        assert np.all(raw_frames[0][0, 0] > 200)
        assert annotated_frames[0][0, 0, 2] > annotated_frames[0][0, 0, 0] + 100

        images = sorted((summary.run_dir / "images" / "train").glob("*.jpg"))
        labels = sorted((summary.run_dir / "labels" / "train").glob("*.txt"))
        assert len(images) == len(labels) == 2
        assert [path.stem for path in images] == [path.stem for path in labels]
        assert [int(path.stem.rsplit("_", 1)[1]) for path in images] == [2, 7]
        assert all(np.all(cv2.imread(str(path))[0, 0] > 200) for path in images)
        assert all(label.read_text().strip() == "0 0.500000 0.500000 0.250000 0.250000" for label in labels)
        assert (summary.run_dir / "train.txt").read_text().splitlines() == [
            f"images/train/{path.name}" for path in images
        ]
        assert '0: "object"' in (summary.run_dir / "data.yaml").read_text()
        assert summary.zip_path and summary.zip_path.is_file()
        with zipfile.ZipFile(summary.zip_path) as archive:
            assert archive.testzip() is None
            assert set(archive.namelist()) == {
                "data.yaml", "train.txt",
                *(f"images/train/{path.name}" for path in images),
                *(f"labels/train/{path.name}" for path in labels),
            }

        all_frames = run_videos(VideoConfig(
            model, [mixed], root / "all_frames", save_interval_seconds=0,
        ))
        assert all_frames.saved_images == 4

        cancel = False

        def progress(_done: int, _total: int, message: str) -> None:
            nonlocal cancel
            if "已检测 2 帧" in message:
                cancel = True

        cancelled = run_videos(VideoConfig(model, [mixed], root / "cancel"),
                               progress, lambda: cancel)
        assert cancelled.cancelled and cancelled.succeeded == 0
        assert not list(cancelled.run_dir.glob("*.mp4"))
        assert not list((cancelled.run_dir / "images" / "train").glob("*.jpg"))
        assert not list((cancelled.run_dir / "labels" / "train").glob("*.txt"))

    print("PASS: paired filtered videos, CVAT ZIP, frame progress, original preserved, cancellation")


if __name__ == "__main__":
    main()
