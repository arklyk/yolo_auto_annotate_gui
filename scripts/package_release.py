"""Build and verify the separate PyQt GUI and YOLO worker executables."""

from __future__ import annotations

import json
import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGING = ROOT / "dist_staging"
DIST = ROOT / "dist"
NAME = "YOLO批量预标注"
ARCHIVE = DIST / "yolo-auto-annotate-windows.zip"


def inside_project(path: Path) -> Path:
    resolved = path.resolve()
    if resolved == ROOT or not resolved.is_relative_to(ROOT):
        raise ValueError(f"拒绝操作项目外的目录: {resolved}")
    return resolved


def build(script: str, name: str, *, gui: bool, excludes: list[str]) -> None:
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--onedir",
        "--windowed" if gui else "--console", "--name", name,
        "--paths", str(ROOT), "--distpath", str(STAGING),
        "--workpath", str(ROOT / "build" / name),
    ]
    if not gui:
        command += ["--collect-all", "ultralytics"]
    for module in excludes:
        command += ["--exclude-module", module]
    command.append(str(ROOT / script))
    subprocess.run(command, cwd=ROOT, check=True)


def make_opencv_paths_portable(worker: Path) -> None:
    cv2_dir = worker / "_internal" / "cv2"
    config = cv2_dir / "config.py"
    version_config = cv2_dir / "config-3.py"
    if not config.is_file() or not version_config.is_file():
        raise FileNotFoundError("OpenCV 配置文件未被打包")
    config.write_text(
        "import os\nBINARIES_PATHS = [os.path.dirname(LOADER_DIR)] + BINARIES_PATHS\n",
        encoding="utf-8",
    )
    version_config.write_text(
        "import os\nPYTHON_EXTENSIONS_PATHS = [os.path.join(LOADER_DIR, 'python-3')] + PYTHON_EXTENSIONS_PATHS\n",
        encoding="utf-8",
    )


def test_release(gui_dir: Path, model: Path, sample: Path) -> None:
    if not model.is_file():
        raise FileNotFoundError(f"缺少自测模型: {model}")
    if not sample.is_file():
        raise FileNotFoundError(f"缺少自测图片: {sample}")
    report = STAGING / "self_test.json"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    subprocess.run(
        [str(gui_dir / f"{NAME}.exe"), "--self-test", str(model), str(sample), str(report)],
        cwd=ROOT, env=env, check=True, timeout=180,
    )
    result = json.loads(report.read_text(encoding="utf-8"))
    if not result.get("ok"):
        raise RuntimeError(f"打包版自测失败: {result}")
    print(f"打包版自测通过: {result}")


def create_archive() -> None:
    release_dir = DIST / NAME
    if not (release_dir / f"{NAME}.exe").is_file():
        raise FileNotFoundError(f"缺少发布程序: {release_dir}")
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(release_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(DIST).as_posix())
    print(f"发布压缩包: {ARCHIVE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="打包 PyQt 界面与 YOLO 推理进程")
    parser.add_argument("--model", type=Path, help="用于发布前自测的 YOLO .pt 模型")
    parser.add_argument("--sample", type=Path, help="用于发布前自测的单张图片")
    parser.add_argument("--zip-only", action="store_true", help="仅重新压缩已有发布目录")
    args = parser.parse_args()
    if args.zip_only:
        create_archive()
        return
    if bool(args.model) != bool(args.sample):
        parser.error("--model 和 --sample 必须一起提供")
    if args.model and not args.model.is_file():
        parser.error(f"模型不存在: {args.model}")
    if args.sample and not args.sample.is_file():
        parser.error(f"测试图片不存在: {args.sample}")
    if STAGING.exists():
        shutil.rmtree(inside_project(STAGING))
    STAGING.mkdir()
    build("backend.py", "annotate_worker", gui=False,
          excludes=["PyQt5", "PySide6", "matplotlib", "pandas", "tensorflow"])
    worker = STAGING / "annotate_worker"
    make_opencv_paths_portable(worker)
    build("app.py", NAME, gui=True,
          excludes=["ultralytics", "cv2", "torch", "torchvision", "matplotlib", "numpy"])
    gui_dir = STAGING / NAME
    shutil.move(str(worker), str(gui_dir / "worker"))
    if args.model:
        test_release(gui_dir, args.model.resolve(), args.sample.resolve())
    else:
        print("未提供 --model/--sample，跳过真实模型自测")

    DIST.mkdir(exist_ok=True)
    official = DIST / NAME
    if official.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = DIST / f"{NAME}_旧版_{stamp}"
        shutil.move(str(inside_project(official)), str(inside_project(backup)))
        print(f"旧版已保留: {backup}")
    shutil.move(str(inside_project(gui_dir)), str(inside_project(official)))
    print(f"新版已发布: {official / (NAME + '.exe')}")
    create_archive()


if __name__ == "__main__":
    main()
