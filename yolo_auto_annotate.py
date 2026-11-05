#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用训练好的 Ultralytics YOLO 检测模型批量生成预标注。

使用说明
1. 安装 requirements.txt 中的依赖。
2. 图形界面运行：python app.py。
3. 命令行运行：python yolo_auto_annotate.py 模型.pt 图片目录 输出目录。
5. 每次在 OUTPUT_ROOT/run_时间戳/ 新建结果：
   images/train/ 为原图副本；labels/train/ 为标准 YOLO 检测标签；
   data.yaml 和 train.txt 描述数据集；previews/ 为画框预览；
   errors.log 为逐图异常日志；cvat_ultralytics_yolo_detection.zip 可导入 CVAT。
6. 在 CVAT 中选择“Ultralytics YOLO Detection 1.0”导入该 ZIP，逐张复核。

TXT 的每行是“新类别ID x中心 y中心 宽 高”，坐标均归一化到 0~1。
即使没有目标，也会创建空 TXT。预览图不会放进 CVAT ZIP。
模型只能预标注它实际学过的类别；映射只是改类别 ID，不会产生新检测能力。
"""

# 默认路径；图形界面和命令行参数会覆盖这些值。
from pathlib import Path

MODEL_PATH = Path("model.pt")
INPUT_DIR = Path("images")
OUTPUT_ROOT = Path("output")
# 输出示例：output/run_20261008_123456_000000/
#             images\train\  原图副本    labels\train\  YOLO TXT
#             previews\      画框预览    data.yaml      新类别表
#             errors.log      异常日志    cvat_ultralytics_yolo_detection.zip

# 推理和扫描配置
IMAGE_SIZE = 640                                    # 推理输入尺寸
RECURSIVE = True                                   # 是否扫描子目录
SAVE_PREVIEW = True                                # 预览图放在 previews/，不修改原图
CREATE_CVAT_ZIP = True                             # 生成可导入 CVAT 的 ZIP

OLD_CLASS_NAMES = {}
NEW_CLASS_NAMES = {}
SELECT_CLASSES = []
CLASS_MAP = {}
CLASS_CONFIDENCE = {}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
# =====================================================================

import hashlib
import json
import math
import shutil
import sys
import traceback
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable


@dataclass
class RunConfig:
    model_path: Path = MODEL_PATH
    input_dir: Path = INPUT_DIR
    output_root: Path = OUTPUT_ROOT
    image_size: int = IMAGE_SIZE
    recursive: bool = RECURSIVE
    save_preview: bool = SAVE_PREVIEW
    create_cvat_zip: bool = CREATE_CVAT_ZIP
    old_class_names: dict[int, str] = field(default_factory=lambda: OLD_CLASS_NAMES.copy())
    new_class_names: dict[int, str] = field(default_factory=lambda: NEW_CLASS_NAMES.copy())
    select_classes: list[int] = field(default_factory=lambda: SELECT_CLASSES.copy())
    class_map: dict[int, int] = field(default_factory=lambda: CLASS_MAP.copy())
    class_confidence: dict = field(default_factory=lambda: CLASS_CONFIDENCE.copy())


@dataclass
class RunSummary:
    run_dir: Path
    total: int
    succeeded: int
    failed: int
    box_count: int
    cancelled: bool
    zip_path: Path | None


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _model_names(raw) -> dict[int, str]:
    if isinstance(raw, (list, tuple)):
        names = dict(enumerate(raw))
    elif isinstance(raw, dict):
        names = {int(k): v for k, v in raw.items()}
    else:
        raise ValueError("模型没有可读取的类别名称 model.names")
    if not names or any(not isinstance(v, str) or not v for v in names.values()):
        raise ValueError("模型类别名称为空或格式错误")
    if len(set(names.values())) != len(names):
        raise ValueError("模型类别名称有重复，请使用唯一名称")
    return names


def _resolve_class(value, names: dict[int, str], side: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{side} 类别不能为布尔值: {value}")
    if isinstance(value, int):
        if value in names:
            return value
    elif isinstance(value, str):
        matches = [i for i, name in names.items() if name == value]
        if len(matches) == 1:
            return matches[0]
    raise ValueError(f"{side} 类别不存在: {value!r}; 可用类别: {names}")


def build_class_mapping(model_names: dict[int, str], config: RunConfig | None = None) -> dict[int, int]:
    config = config or RunConfig()
    for label, names in (("OLD_CLASS_NAMES", config.old_class_names),
                         ("NEW_CLASS_NAMES", config.new_class_names)):
        if (not isinstance(names, dict) or not names
                or set(names) != set(range(len(names)))
                or any(not isinstance(v, str) or not v for v in names.values())
                or len(set(names.values())) != len(names)):
            raise ValueError(f"{label} 必须是从 0 连续编号、名称不重复的字典")
    if model_names != config.old_class_names:
        raise ValueError(f"OLD_CLASS_NAMES 与 best.pt 不一致。\n"
                         f"配置: {config.old_class_names}\n模型实际: {model_names}")
    if not config.select_classes:
        raise ValueError("SELECT_CLASSES 不能为空")
    selected = {_resolve_class(x, model_names, "旧模型") for x in config.select_classes}
    mapping = {}
    for source, target in config.class_map.items():
        old_id = _resolve_class(source, model_names, "映射源")
        new_id = _resolve_class(target, config.new_class_names, "映射目标")
        if old_id in mapping and mapping[old_id] != new_id:
            raise ValueError(f"旧类别 {old_id} 被映射到多个新类别")
        mapping[old_id] = new_id
    missing = selected - mapping.keys()
    if missing:
        raise ValueError(f"选中的旧类别缺少 CLASS_MAP 映射: {sorted(missing)}")
    return {old_id: mapping[old_id] for old_id in selected}


def build_class_confidence(model_names: dict[int, str], selected_ids,
                           config: RunConfig | None = None) -> dict[int, float]:
    config = config or RunConfig()
    if not isinstance(config.class_confidence, dict):
        raise ValueError("CLASS_CONFIDENCE 必须是字典")
    thresholds = {}
    for source, value in config.class_confidence.items():
        old_id = _resolve_class(source, model_names, "置信度对应的旧模型")
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError(f"旧类别 {source!r} 的置信度必须在 (0, 1] 内: {value!r}")
        if old_id in thresholds:
            raise ValueError(f"旧类别 {model_names[old_id]!r} 的置信度被重复设置")
        thresholds[old_id] = float(value)
    missing = set(selected_ids) - thresholds.keys()
    if missing:
        raise ValueError(f"选中的旧类别缺少 CLASS_CONFIDENCE: "
                         f"{[model_names[i] for i in sorted(missing)]}")
    return {old_id: thresholds[old_id] for old_id in selected_ids}


def discover_images(input_dir: Path, output_root: Path, recursive: bool = RECURSIVE) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        (p for p in iterator if p.is_file()
         and p.suffix.lower() in IMAGE_EXTENSIONS
         and not _under(p, output_root)),
        key=lambda p: str(p).casefold(),
    )


def _output_name(image: Path, input_dir: Path) -> str:
    relative = image.relative_to(input_dir)
    stem = "__".join(relative.with_suffix("").parts)
    digest = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:12]
    return f"{stem[:80]}__{digest}{image.suffix.lower()}"


def _tensor_values(tensor):
    """Accept Ultralytics tensors and simple array-like test doubles."""
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "tolist"):
        return tensor.tolist()
    return list(tensor)


def extract_annotations(result, class_mapping: dict[int, int],
                        class_confidence: dict[int, float]) -> list[tuple[int, float, float, float, float, float]]:
    boxes = result.boxes
    if boxes is None:
        return []
    ids = _tensor_values(boxes.cls)
    confidences = _tensor_values(boxes.conf)
    coords = _tensor_values(boxes.xywhn)
    if not (len(ids) == len(confidences) == len(coords)):
        raise ValueError("模型输出的类别、置信度和坐标数量不一致")
    annotations = []
    for old_id, confidence, xywh in zip(ids, confidences, coords):
        old_id = int(old_id)
        confidence = float(confidence)
        if old_id not in class_mapping or confidence < class_confidence[old_id]:
            continue
        if len(xywh) != 4:
            raise ValueError("模型输出的坐标不是 xywh")
        x, y, w, h = (float(v) for v in xywh)
        # 处理极小的数值越界；无面积的框不写入标签。
        left, top = max(0.0, x - w / 2), max(0.0, y - h / 2)
        right, bottom = min(1.0, x + w / 2), min(1.0, y + h / 2)
        if right <= left or bottom <= top:
            continue
        annotations.append((class_mapping[old_id],
                            (left + right) / 2, (top + bottom) / 2,
                            right - left, bottom - top, confidence))
    return annotations


def _label_text(annotations) -> str:
    return "".join(
        f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n"
        for cls, x, y, w, h, _ in annotations
    )


def _save_preview(source: Path, target: Path, annotations,
                  new_class_names: dict[int, str] | None = None) -> None:
    from PIL import Image, ImageDraw, ImageOps

    with Image.open(source) as opened:
        canvas = ImageOps.exif_transpose(opened).convert("RGB")
    new_class_names = new_class_names or NEW_CLASS_NAMES
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    line_width = max(2, round(min(width, height) / 300))
    for cls, x, y, w, h, confidence in annotations:
        rect = ((x - w / 2) * width, (y - h / 2) * height,
                (x + w / 2) * width, (y + h / 2) * height)
        draw.rectangle(rect, outline="#ff3030", width=line_width)
        caption = f"{new_class_names[cls]} {confidence:.2f}"
        text_y = max(0, int(rect[1]) - 16)
        draw.text((int(rect[0]) + 2, text_y), caption, fill="#ff3030")
    canvas.save(target, format="JPEG", quality=90)


def _write_metadata(run_dir: Path, image_lines: list[str],
                    new_class_names: dict[int, str] | None = None) -> None:
    new_class_names = new_class_names or NEW_CLASS_NAMES
    yaml_lines = ["path: .", "train: train.txt", "names:"]
    yaml_lines.extend(f"  {i}: {json.dumps(name, ensure_ascii=False)}"
                      for i, name in sorted(new_class_names.items()))
    (run_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    (run_dir / "train.txt").write_text("".join(image_lines), encoding="utf-8")


def _make_cvat_zip(run_dir: Path) -> Path:
    zip_path = run_dir / "cvat_ultralytics_yolo_detection.zip"
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED,
                         allowZip64=True) as archive:
        for name in ("data.yaml", "train.txt"):
            archive.write(run_dir / name, name)
        for folder in ("images/train", "labels/train"):
            for path in sorted((run_dir / folder).iterdir()):
                archive.write(path, path.relative_to(run_dir).as_posix())
    return zip_path


def run_batch(config: RunConfig, progress: Callable[[int, int, str], None] | None = None,
              should_cancel: Callable[[], bool] | None = None) -> RunSummary:
    """Run one independent batch; callbacks are called after each attempted image."""
    if config.image_size <= 0:
        raise ValueError("IMAGE_SIZE 必须大于 0")
    model_path, input_dir, output_root = (Path(x).expanduser().resolve()
                                           for x in (config.model_path, config.input_dir, config.output_root))
    print(f"模型文件: {model_path}")
    print(f"读取图片: {input_dir}")
    print(f"保存位置: {output_root} (本次运行会新建 run_时间戳 子目录)")
    if not model_path.is_file():
        raise FileNotFoundError(f"模型不存在: {model_path}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"图片目录不存在: {input_dir}")
    if _under(input_dir, output_root):
        raise ValueError("INPUT_DIR 不能位于 OUTPUT_ROOT 内")
    images = discover_images(input_dir, output_root, config.recursive)
    if not images:
        raise FileNotFoundError(f"未找到图片: {input_dir}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("缺少 ultralytics；请运行 python -m pip install ultralytics pillow") from exc
    if config.save_preview:
        try:
            from PIL import Image  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("预览功能需要 pillow；请运行 python -m pip install pillow") from exc

    model = YOLO(str(model_path))
    model_names = _model_names(model.names)
    class_mapping = build_class_mapping(model_names, config)
    class_confidence = build_class_confidence(model_names, class_mapping, config)
    print(f"旧模型类别: {model_names}")
    print(f"本次映射: {[(model_names[k], config.new_class_names[v]) for k, v in sorted(class_mapping.items())]}")
    print(f"本次各类置信度: {[(model_names[k], v) for k, v in sorted(class_confidence.items())]}")

    # 每次运行独立目录，使用独占创建，防止误覆盖历史结果。
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(exist_ok=False)
    images_dir = run_dir / "images" / "train"
    labels_dir = run_dir / "labels" / "train"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)
    preview_dir = run_dir / "previews"
    if config.save_preview:
        preview_dir.mkdir()
    error_log = run_dir / "errors.log"
    error_log.write_text("", encoding="utf-8")

    succeeded = failed = box_count = 0
    image_lines = []
    cancelled = False
    for index, image in enumerate(images, 1):
        if should_cancel and should_cancel():
            cancelled = True
            break
        output_name = _output_name(image, input_dir)
        image_target = images_dir / output_name
        label_target = labels_dir / (Path(output_name).stem + ".txt")
        preview_target = preview_dir / (Path(output_name).stem + ".jpg")
        try:
            # 先验证图片可读取；损坏文件会写入异常日志，不影响其他图片。
            from PIL import Image
            with Image.open(image) as opened:
                opened.verify()
            result_list = model.predict(source=str(image), conf=min(class_confidence.values()),
                                        imgsz=config.image_size, verbose=False, save=False)
            if len(result_list) != 1:
                raise RuntimeError(f"预期一个推理结果，实际 {len(result_list)} 个")
            annotations = extract_annotations(result_list[0], class_mapping, class_confidence)
            shutil.copy2(image, image_target)  # 保留原始图片字节，不修改源文件
            label_target.write_text(_label_text(annotations), encoding="utf-8")
            if config.save_preview:
                _save_preview(image, preview_target, annotations, config.new_class_names)
            image_lines.append(f"images/train/{output_name}\n")
            succeeded += 1
            box_count += len(annotations)
            print(f"[{index}/{len(images)}] OK {image.relative_to(input_dir)} -> {len(annotations)} 框")
        except Exception:
            failed += 1
            with error_log.open("a", encoding="utf-8") as log_file:
                log_file.write(f"{datetime.now().isoformat(timespec='seconds')} "
                               f"图片处理失败: {image}\n{traceback.format_exc()}\n")
            for partial in (image_target, label_target, preview_target):
                partial.unlink(missing_ok=True)
            print(f"[{index}/{len(images)}] 失败 {image}; 详见 {error_log}", file=sys.stderr)
        if progress:
            progress(index, len(images), f"{image.relative_to(input_dir)}: {'失败' if not label_target.exists() else '完成'}")

    _write_metadata(run_dir, image_lines, config.new_class_names)
    zip_path = _make_cvat_zip(run_dir) if config.create_cvat_zip and succeeded else None
    print(f"完成: 成功 {succeeded} 张，失败 {failed} 张，标签框 {box_count} 个")
    print(f"输出目录: {run_dir}")
    if zip_path:
        print(f"CVAT ZIP: {zip_path}")
    print("预标注必须在 CVAT 中逐张复核；空 TXT 表示该图无所选类别的检出框。")
    return RunSummary(run_dir, len(images), succeeded, failed, box_count, cancelled, zip_path)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="使用 YOLO 模型批量生成预标注")
    parser.add_argument("model", type=Path, help="YOLO .pt 模型")
    parser.add_argument("images", type=Path, help="原图目录")
    parser.add_argument("output", type=Path, help="输出根目录")
    args = parser.parse_args()
    from ultralytics import YOLO
    names = _model_names(YOLO(str(args.model)).names)
    config = RunConfig(
        model_path=args.model, input_dir=args.images, output_root=args.output,
        old_class_names=names, new_class_names=names.copy(),
        select_classes=list(names), class_map={i: i for i in names},
        class_confidence={i: 0.25 for i in names},
    )
    summary = run_batch(config)
    return 0 if not summary.cancelled and summary.failed == 0 and summary.succeeded > 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"运行失败: {exc}", file=sys.stderr)
        sys.exit(1)
