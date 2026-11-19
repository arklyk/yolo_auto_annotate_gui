"""Exercise paired videos and CVAT export through the GUI-to-worker protocol."""

import argparse
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from app import QApplication, Job  # noqa: E402
from video_inference import VideoConfig  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("image", type=Path)
    args = parser.parse_args()
    frame = cv2.imdecode(np.fromfile(str(args.image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"无法读取测试图片: {args.image}")
    frame = cv2.resize(frame, (320, 240))
    with TemporaryDirectory() as folder:
        root = Path(folder)
        videos = []
        for index, count in ((1, 3), (2, 2)):
            path = root / f"测试视频_{index}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (320, 240))
            assert writer.isOpened()
            for _ in range(count):
                writer.write(frame)
            writer.release()
            videos.append(path)

        app = QApplication([])
        results, errors, progress_events = [], [], []
        job = Job(str(args.model), VideoConfig(args.model, videos, root / "识别结果", 0.1, 320))
        job.progress.connect(lambda done, total, _message: progress_events.append((done, total)))
        job.completed.connect(results.append)
        job.failed.connect(errors.append)
        job.run()
        assert not errors, errors
        assert len(results) == 1
        summary = results[0]
        assert summary.succeeded == 2 and summary.failed == 0 and not summary.cancelled
        assert any(0 < done < total for done, total in progress_events)
        assert progress_events[-1] == (2000, 2000)

        outputs = sorted(summary.run_dir.glob("*.mp4"))
        assert len(outputs) % 2 == 0
        for raw in summary.run_dir.glob("*_filtered.mp4"):
            annotated = raw.with_name(raw.name.replace("_filtered.mp4", "_annotated.mp4"))
            assert annotated.is_file()
            counts = []
            for path in (raw, annotated):
                reader = cv2.VideoCapture(str(path))
                assert reader.isOpened()
                counts.append(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))
                assert int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
                assert int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240
                reader.release()
            assert counts[0] == counts[1] and counts[0] > 0
        images = list((summary.run_dir / "images" / "train").glob("*.jpg"))
        labels = list((summary.run_dir / "labels" / "train").glob("*.txt"))
        assert len(images) == len(labels) == summary.saved_images
        assert (summary.run_dir / "data.yaml").is_file()
        assert (summary.run_dir / "train.txt").is_file()
        if images:
            assert summary.zip_path and summary.zip_path.is_file()
        print("PASS: paired videos, CVAT files and frame progress through GUI worker")

        cancelled_results, cancel_errors = [], []
        cancel_job = Job(str(args.model), VideoConfig(args.model, videos, root / "cancel", 0.1, 320))
        cancel_job.progress.connect(
            lambda _done, _total, message: cancel_job.request_cancel()
            if "已检测 1 帧" in message else None
        )
        cancel_job.completed.connect(cancelled_results.append)
        cancel_job.failed.connect(cancel_errors.append)
        cancel_job.run()
        assert not cancel_errors, cancel_errors
        assert len(cancelled_results) == 1 and cancelled_results[0].cancelled
        assert not list(cancelled_results[0].run_dir.glob("*.mp4"))
        assert not list((cancelled_results[0].run_dir / "images" / "train").glob("*.jpg"))
        app.quit()
        print("PASS: cancellation removes incomplete video and dataset images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
