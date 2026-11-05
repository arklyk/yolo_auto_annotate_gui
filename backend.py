"""YOLO worker process. Kept separate from PyQt to isolate native DLLs."""

from __future__ import annotations

import contextlib
import json
import sys
import traceback
from pathlib import Path

from yolo_auto_annotate import RunConfig, _model_names, run_batch  # noqa: E402

PREFIX = "@@YOLO_GUI_JSON@@"


def emit(event: dict) -> None:
    sys.stdout.write(PREFIX + json.dumps(event, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def config_from_json(path: Path) -> tuple[RunConfig, Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload["config"]
    for key in ("model_path", "input_dir", "output_root"):
        data[key] = Path(data[key])
    for key in ("old_class_names", "new_class_names", "class_map"):
        data[key] = {int(k): v for k, v in data[key].items()}
    data["class_confidence"] = {
        int(k) if k.isdecimal() else k: v
        for k, v in data["class_confidence"].items()
    }
    return RunConfig(**data), Path(payload["cancel_path"])


def main() -> int:
    if len(sys.argv) < 3:
        raise ValueError("用法: annotate_worker classes 模型.pt | run 配置.json")
    action = sys.argv[1]
    if action == "classes":
        from ultralytics import YOLO
        with contextlib.redirect_stdout(sys.stderr):
            names = _model_names(YOLO(sys.argv[2]).names)
        emit({"type": "classes", "names": names})
        return 0
    if action == "run":
        config, cancel_path = config_from_json(Path(sys.argv[2]))
        with contextlib.redirect_stdout(sys.stderr):
            summary = run_batch(
                config,
                lambda done, total, message: emit({
                    "type": "progress", "done": done, "total": total, "message": message,
                }),
                cancel_path.exists,
            )
        emit({
            "type": "complete", "run_dir": str(summary.run_dir),
            "total": summary.total, "succeeded": summary.succeeded,
            "failed": summary.failed, "box_count": summary.box_count,
            "cancelled": summary.cancelled,
            "zip_path": str(summary.zip_path) if summary.zip_path else None,
        })
        return 0
    raise ValueError(f"未知操作: {action}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        emit({"type": "error", "detail": traceback.format_exc()})
        sys.exit(1)
