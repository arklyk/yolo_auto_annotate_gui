"""Exercise two videos through the GUI-to-worker protocol."""

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
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
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
        results, errors = [], []
        job = Job(str(args.model), VideoConfig(args.model, videos, root / "识别结果", 0.1, 320))
        job.completed.connect(results.append)
        job.failed.connect(errors.append)
        job.run()
        app.quit()
        assert not errors, errors
        assert len(results) == 1
        summary = results[0]
        assert summary.succeeded == 2 and summary.failed == 0 and not summary.cancelled
        outputs = sorted(summary.run_dir.glob("*.mp4"))
        assert len(outputs) == 2, (outputs, (summary.run_dir / "errors.log").read_text(encoding="utf-8"))
        for path, expected in zip(outputs, (3, 2)):
            reader = cv2.VideoCapture(str(path))
            assert reader.isOpened()
            assert int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) == expected
            assert int(reader.get(cv2.CAP_PROP_FRAME_WIDTH)) == 320
            assert int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240
            reader.release()
        print("PASS: two annotated videos, expected frame counts and dimensions")

        cancelled_results, cancel_errors = [], []
        cancel_job = Job(str(args.model), VideoConfig(args.model, videos, root / "cancel", 0.1, 320))
        cancel_job.progress.connect(
            lambda done, total, message: cancel_job.request_cancel()
            if done == 0 and "帧" in message else None
        )
        cancel_job.completed.connect(cancelled_results.append)
        cancel_job.failed.connect(cancel_errors.append)
        cancel_job.run()
        assert not cancel_errors, cancel_errors
        assert len(cancelled_results) == 1 and cancelled_results[0].cancelled
        assert not list(cancelled_results[0].run_dir.glob("*.mp4"))
        print("PASS: cancellation removes incomplete video")
    return 0


if __name__ == "__main__":
    sys.exit(main())
