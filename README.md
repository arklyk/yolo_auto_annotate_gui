# YOLO 批量预标注

Windows 图形工具：使用 Ultralytics YOLO 检测模型完成图片批量预标注或视频识别。图片模式生成 YOLO TXT、复核预览图和可选的 CVAT ZIP；视频模式生成带检测框的 MP4，并可导出原始帧与 YOLO 标签作为候选数据集。默认不修改原始图片与视频，预标注结果仍需人工复核。

## 下载后直接使用

请从 GitHub **Releases** 下载 `yolo-auto-annotate-windows.zip`，完整解压后运行 `YOLO批量预标注/YOLO批量预标注.exe`。不要只复制 EXE；同目录的 `_internal/` 和 `worker/` 也必须保留。模型 `.pt` 与待处理图片由使用者自行选择。

GitHub 页面上的 **Code → Download ZIP** 下载的是源码，不包含打包程序。运行源码需要先安装 Python 和下述依赖。

## 目录

| 路径 | 用途 |
| --- | --- |
| `app.py` | PyQt5 图形界面 |
| `backend.py` | 独立推理进程，隔离 GUI 与 YOLO 的原生库 |
| `yolo_auto_annotate.py` | 图片扫描、推理和导出逻辑；也可单独从命令行运行 |
| `video_inference.py` | 多视频逐帧识别、筛选 MP4 和候选数据集导出 |
| `scripts/package_release.py`、`构建exe.bat` | Windows 可执行程序打包 |
| `requirements.txt`、`requirements-build.txt` | 运行依赖与打包依赖 |
| `安装依赖.bat`、`启动上位机.bat` | 从源码安装与启动 |
| `tests/smoke_test.py` | 使用自备模型和图片进行端到端检查 |
| `dist/` | 本机发布目录及 ZIP，已由 `.gitignore` 排除 |

## 从源码运行

建议使用 Python 3.10 或更新版本。Windows 用户可先双击 `安装依赖.bat`，完成后双击 `启动上位机.bat`。也可在项目目录打开 PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

也可双击 `启动上位机.bat`；它优先使用本机专用的 `python.local.bat`，其次使用本目录的 `.venv`，最后使用系统 `python`。如要使用已有 Conda 环境，可在项目根目录创建不会被提交的 `python.local.bat`，内容例如 `@"C:\path\to\python.exe" %*`，并按自己的实际路径修改。首次安装 Ultralytics 等依赖可能耗时较长。模型权重和图片由用户自行提供，不随源码分发。

## 图片预标注

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

## 视频识别

1. 点击顶部“视频识别”按钮，选择检测模型 `.pt`、输出根目录，点击“添加视频…”选择一个或多个视频。支持 MP4、AVI、MOV、MKV、WMV、M4V 和 WebM 输入，能否解码取决于本机 OpenCV 的编解码器支持。
2. 设置统一的最低置信度和推理尺寸。默认勾选“结果视频跳过无检测画面”和“保存数据集原始帧和 YOLO 标签”。程序仍检测每一帧；结果视频只写入有检测框的帧。数据集保存首个有检测框的原始帧，此后每隔至少 1 秒保存一张有检测框的帧。可调整间隔，设为 0 秒则保存所有有检测框的帧。
3. 点击“开始视频识别”。每次运行在 `输出根目录/video_run_时间戳/` 中生成 MP4、`errors.log`，以及 `dataset/images/`、`dataset/labels/` 和 `dataset/classes.txt`。图片未画框，同名 TXT 为归一化的 `类别 中心x 中心y 宽 高`。启用跳过空画面时，完全没有检测结果的视频不生成空 MP4；失败或停止时清理当前视频未完成的结果。已完成的视频保留。
4. 可勾选“替换原 MP4”以直接更新输入 MP4：程序先验证新视频，再把旧视频保存为同目录的 `原文件名.original_时间戳.mp4` 备份。此选项默认关闭，且暂不支持替换 AVI、MOV 等其他格式；没有检测结果时不会替换。结果视频保留原尺寸和帧率，但删除空帧后时长会缩短，当前版本不保留音轨。

候选数据集需人工复核误检和漏检。连续视频帧高度相似；正式训练前应按原视频划分训练集与验证集，避免相邻帧进入两组。这里导出的 `dataset/` 是待复核素材，未自动划分训练集，也未生成 `data.yaml`。

## 打包与检查

在 Windows 的同一 Python 环境安装打包依赖后运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-build.txt
.\构建exe.bat --model "模型.pt" --sample "测试图片.jpg"
```

脚本使用 PyInstaller 分别打包 GUI 和推理进程，并用提供的模型、图片检查图片预标注和视频识别。若只想生成包，可直接运行 `构建exe.bat`，此时会跳过真实推理检查。成功后会生成 `dist/yolo-auto-annotate-windows.zip` 并清理构建中间文件；也可以复制整个 `dist/YOLO批量预标注/`，包括其中的 `worker/`。不要只复制 EXE。旧发布包会留在 `dist/` 中，确认新版可用后可手动删除。

源码端到端检查：

```powershell
python tests/smoke_test.py "模型.pt" "测试图片.jpg"
python tests/video_smoke_test.py "模型.pt" "测试图片.jpg"
```

## 准备公开仓库

本项目源码按 [GNU AGPL-3.0](LICENSE) 发布。Ultralytics 使用 AGPL-3.0，PyQt5 的开源版本使用 GPLv3；如分发打包 EXE，还须遵守其随包依赖的许可证和源码提供义务。模型权重与训练图片不随项目发布，其分发权需另行核对。模型还需与实际安装的 Ultralytics 版本兼容。

仓库仅应包含源码、脚本、许可证和说明。请在首次推送前检查 `git status` 与 `git ls-files`，确认没有模型权重、图片数据、输出结果、构建目录、密钥或个人信息。

## 维护者发布更新

代码更新：在项目目录检查 `git status`，然后提交改动并执行 `git push origin main`。GitHub 的 **Code → Download ZIP** 会随分支更新，但只包含源码。

Windows 免安装版更新：运行上面的打包命令并确认自测通过，再为新版本创建 GitHub Release，将 `dist/yolo-auto-annotate-windows.zip` 作为附件上传。仅推送源码不会更新已有 Release 的附件。
