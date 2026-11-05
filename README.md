# YOLO 批量预标注

Windows 图形工具：读取 Ultralytics YOLO 检测模型的类别，对图片批量推理，按类别设置阈值与输出名称，生成 YOLO TXT、复核预览图和可选的 CVAT ZIP。原图不会被修改；输出目录中的 `images/train` 是副本。预标注结果仍需人工复核。

## 目录

| 路径 | 用途 |
| --- | --- |
| `app.py` | PyQt5 图形界面 |
| `backend.py` | 独立推理进程，隔离 GUI 与 YOLO 的原生库 |
| `yolo_auto_annotate.py` | 图片扫描、推理和导出逻辑；也可单独从命令行运行 |
| `scripts/package_release.py`、`构建exe.bat` | Windows 可执行程序打包 |
| `requirements.txt`、`requirements-build.txt` | 运行依赖与打包依赖 |
| `tests/smoke_test.py` | 使用自备模型和图片进行端到端检查 |
| `build/`、`dist/` | 本机生成的构建文件与发布文件，已由 `.gitignore` 排除 |

## 从源码运行

建议使用 Python 3.10 或更新版本。在项目目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

也可双击 `启动上位机.bat`；它优先使用本机专用的 `python.local.bat`，其次使用本目录的 `.venv`，最后使用系统 `python`。如要使用已有 Conda 环境，可在项目根目录创建不会被提交的 `python.local.bat`，内容例如 `@"C:\path\to\python.exe" %*`，并按自己的实际路径修改。首次安装 Ultralytics 等依赖可能耗时较长。模型权重和图片由用户自行提供，不随源码分发。

## 使用方法

1. 选择 `.pt` 检测模型、原图目录和输出根目录，点击“读取类别”。
2. 勾选需要的类别，为每类设置输出名称和最低置信度。相同输出名称会合并为同一个新类别。
3. 设置推理尺寸、子目录扫描、预览图和 CVAT ZIP 选项，点击“开始预标注”。
4. 在 `输出根目录/run_时间戳/` 检查 `images/train/`、`labels/train/`、`data.yaml`、`previews/` 和 `errors.log`。空 TXT 表示没有达到阈值的检测框。
5. 可将生成的 ZIP 导入 CVAT 继续人工修正。点击“停止”后，当前图片处理完成再结束。

界面设置保存在当前用户的本机设置中。模型路径、图片和推理结果不会自动上传。

命令行如需直接处理全部模型类别，可运行：

```powershell
python yolo_auto_annotate.py "模型.pt" "图片目录" "输出目录"
```

此命令使用每类 `0.25` 的默认置信度；需要按类别调整时使用图形界面。

## 打包与检查

在 Windows 的同一 Python 环境安装打包依赖后运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\构建exe.bat --model "模型.pt" --sample "测试图片.jpg"
```

脚本使用 PyInstaller 分别打包 GUI 和推理进程，并用提供的模型、图片做真实推理检查。若只想生成包，可直接运行 `构建exe.bat`，此时会跳过真实推理检查。发布时复制整个 `dist/YOLO批量预标注/`，包括其中的 `worker/`；不要只复制 EXE。旧发布包会留在 `dist/` 中。

源码端到端检查：

```powershell
python tests/smoke_test.py "模型.pt" "测试图片.jpg"
```

## 准备公开仓库

本项目源码按 [GNU AGPL-3.0](LICENSE) 发布。Ultralytics 使用 AGPL-3.0，PyQt5 的开源版本使用 GPLv3；如分发打包 EXE，还须遵守其随包依赖的许可证和源码提供义务。模型权重与训练图片不随项目发布，其分发权需另行核对。模型还需与实际安装的 Ultralytics 版本兼容。

仓库仅应包含源码、脚本、许可证和说明。请在首次推送前检查 `git status` 与 `git ls-files`，确认没有模型权重、图片数据、输出结果、构建目录、密钥或个人信息。
