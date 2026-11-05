"""One-image GUI/worker integration check with user-supplied fixtures."""

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from app import self_test  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="YOLO .pt 模型")
    parser.add_argument("sample", type=Path, help="一张测试图片")
    args = parser.parse_args()
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as folder:
        report = Path(folder) / "report.json"
        status = self_test(args.model.resolve(), args.sample.resolve(), report)
        print(json.dumps(json.loads(report.read_text(encoding="utf-8")), ensure_ascii=False))
        return status


if __name__ == "__main__":
    sys.exit(main())
