"""PyQt5 front end for the existing yolo_auto_annotate batch exporter."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

# PyInstaller --windowed may set the standard streams to None.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from PyQt5.QtCore import QObject, QSettings, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from yolo_auto_annotate import OUTPUT_ROOT, RunConfig


class Job(QObject):
    classes_ready = pyqtSignal(object)
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    log_line = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, model_path: str, config: RunConfig | None = None):
        super().__init__()
        self.model_path = model_path
        self.config = config
        self.cancel_flag = threading.Event()
        self.cancel_path: Path | None = None

    @staticmethod
    def backend_command() -> list[str]:
        if getattr(sys, "frozen", False):
            worker = Path(sys.executable).resolve().parent / "worker" / "annotate_worker.exe"
            if not worker.is_file():
                raise FileNotFoundError(f"缺少推理程序: {worker}。请复制完整发布文件夹。")
            return [str(worker)]
        return [sys.executable, "-u", str(Path(__file__).resolve().parent / "backend.py")]

    def request_cancel(self):
        self.cancel_flag.set()
        if self.cancel_path:
            self.cancel_path.touch(exist_ok=True)

    def _execute(self, args: list[str]):
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            self.backend_command() + args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=creationflags,
        )
        result = None
        diagnostics = []
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip("\r\n")
            if not line.startswith("@@YOLO_GUI_JSON@@"):
                if line:
                    diagnostics.append(line)
                    if len(diagnostics) > 80:
                        diagnostics.pop(0)
                    self.log_line.emit(line)
                continue
            event = json.loads(line[len("@@YOLO_GUI_JSON@@"):])
            kind = event.get("type")
            if kind == "progress":
                self.progress.emit(event["done"], event["total"], event["message"])
            elif kind == "error":
                diagnostics.append(event["detail"])
            elif kind in ("classes", "complete"):
                result = event
        exit_code = process.wait()
        if exit_code != 0:
            raise RuntimeError("\n".join(diagnostics[-20:]) or f"推理进程退出码: {exit_code}")
        if result is None:
            raise RuntimeError("推理进程未返回结果。\n" + "\n".join(diagnostics[-20:]))
        return result

    def run(self):
        try:
            if self.config is None:
                result = self._execute(["classes", self.model_path])
                self.classes_ready.emit({int(k): v for k, v in result["names"].items()})
            else:
                output_root = Path(self.config.output_root).expanduser().resolve()
                output_root.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="gui_job_", dir=output_root) as folder:
                    job_dir = Path(folder)
                    self.cancel_path = job_dir / "cancel.flag"
                    if self.cancel_flag.is_set():
                        self.request_cancel()
                    payload = asdict(self.config)
                    for key in ("model_path", "input_dir", "output_root"):
                        payload[key] = str(payload[key])
                    config_path = job_dir / "config.json"
                    config_path.write_text(json.dumps({
                        "config": payload, "cancel_path": str(self.cancel_path),
                    }, ensure_ascii=False), encoding="utf-8")
                    result = self._execute(["run", str(config_path)])
                    result["run_dir"] = Path(result["run_dir"])
                    if result["zip_path"]:
                        result["zip_path"] = Path(result["zip_path"])
                    self.completed.emit(SimpleNamespace(**result))
        except Exception:
            self.failed.emit(traceback.format_exc())
        finally:
            self.cancel_path = None
            self.finished.emit()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 批量预标注")
        self.resize(1000, 790)
        self.settings = QSettings("lyk", "yolo-auto-annotate")
        self.model_names: dict[int, str] = {}
        self.loaded_path = ""
        self.job = None
        self.thread = None
        self.last_output: Path | None = None

        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        layout.setSpacing(12)
        title = QLabel("YOLO 批量预标注")
        title.setFont(QFont("Microsoft YaHei", 19, QFont.Bold))
        layout.addWidget(title)
        subtitle = QLabel("选择模型与原图，设置每类置信度和输出类别，然后批量生成 YOLO 标签及复核预览。")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        paths = QGroupBox("输入与输出")
        form = QFormLayout(paths)
        self.model_edit = QLineEdit(self.settings.value("model", ""))
        self.input_edit = QLineEdit(self.settings.value("input", ""))
        self.output_edit = QLineEdit(self.settings.value("output", str(OUTPUT_ROOT)))
        form.addRow("模型 .pt", self._path_row(self.model_edit, self.choose_model, "读取类别"))
        form.addRow("原图目录", self._path_row(self.input_edit, self.choose_input))
        form.addRow("输出根目录", self._path_row(self.output_edit, self.choose_output))
        layout.addWidget(paths)

        options = QGroupBox("批处理参数")
        option_row = QHBoxLayout(options)
        option_row.addWidget(QLabel("推理尺寸"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(32, 4096)
        self.size_spin.setSingleStep(32)
        self.size_spin.setValue(int(self.settings.value("size", 640)))
        option_row.addWidget(self.size_spin)
        self.recursive = QCheckBox("扫描子目录")
        self.preview = QCheckBox("保存画框预览")
        self.cvat_zip = QCheckBox("生成 CVAT ZIP")
        for key, control in (("recursive", self.recursive), ("preview", self.preview), ("cvat_zip", self.cvat_zip)):
            control.setChecked(self.settings.value(key, "true") == "true")
            option_row.addWidget(control)
        option_row.addStretch()
        layout.addWidget(options)

        class_group = QGroupBox("模型类别与输出设置")
        class_layout = QVBoxLayout(class_group)
        self.class_hint = QLabel("点击“读取类别”，从所选 .pt 获取真实类别。勾选需要预标注的类别。")
        class_layout.addWidget(self.class_hint)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["输出", "模型类别", "输出类别名称", "最低置信度"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 90)
        self.table.setColumnWidth(1, 240)
        self.table.setColumnWidth(2, 260)
        self.table.setMinimumHeight(210)
        class_layout.addWidget(self.table)
        layout.addWidget(class_group, 1)

        action_row = QHBoxLayout()
        self.start_btn = QPushButton("开始预标注")
        self.start_btn.clicked.connect(self.start_batch)
        self.cancel_btn = QPushButton("停止（当前图片完成后）")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel)
        self.open_btn = QPushButton("打开结果文件夹")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self.open_output)
        action_row.addWidget(self.start_btn)
        action_row.addWidget(self.cancel_btn)
        action_row.addWidget(self.open_btn)
        layout.addLayout(action_row)

        self.bar = QProgressBar()
        layout.addWidget(self.bar)
        self.status = QLabel("准备就绪")
        layout.addWidget(self.status)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        layout.addWidget(self.log)
        self.model_edit.textChanged.connect(self.invalidate_model)
        if Path(self.model_edit.text()).is_file():
            self.load_classes()

    def _path_row(self, edit, browse, extra=None):
        row = QWidget()
        box = QHBoxLayout(row)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(edit, 1)
        button = QPushButton("浏览…")
        button.clicked.connect(browse)
        box.addWidget(button)
        if extra:
            read = QPushButton(extra)
            read.clicked.connect(self.load_classes)
            box.addWidget(read)
        return row

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 YOLO 模型", self.model_edit.text(), "PyTorch 权重 (*.pt)")
        if path:
            self.model_edit.setText(path)
            self.load_classes()

    def choose_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择原图目录", self.input_edit.text())
        if path:
            self.input_edit.setText(path)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出根目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def invalidate_model(self):
        if self.model_edit.text().strip() != self.loaded_path:
            self.model_names = {}
            self.table.setRowCount(0)
            self.class_hint.setText("模型路径已改变，请重新读取类别。")

    def _launch(self, config=None):
        self.job = Job(self.model_edit.text().strip(), config)
        self.thread = QThread(self)
        self.job.moveToThread(self.thread)
        self.thread.started.connect(self.job.run)
        self.job.classes_ready.connect(self.show_classes)
        self.job.progress.connect(self.show_progress)
        self.job.completed.connect(self.show_summary)
        self.job.failed.connect(self.show_error)
        self.job.log_line.connect(self.log.append)
        self.job.finished.connect(self.thread.quit)
        self.thread.finished.connect(self.job.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.job_finished)
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(config is not None)
        self.thread.start()

    def load_classes(self):
        path = Path(self.model_edit.text().strip())
        if not path.is_file():
            QMessageBox.warning(self, "模型不存在", f"找不到模型：{path}")
            return
        if self.thread and self.thread.isRunning():
            return
        self.status.setText("正在读取模型类别…")
        self._launch()

    def show_classes(self, names):
        if self.job and self.model_edit.text().strip() != self.job.model_path:
            self.class_hint.setText("模型路径已改变，请重新读取类别。")
            return
        self.model_names = names
        self.loaded_path = self.model_edit.text().strip()
        self.table.setRowCount(len(names))
        preset_key = hashlib.sha256(self.loaded_path.encode("utf-8")).hexdigest()
        try:
            saved = json.loads(self.settings.value(f"classes/{preset_key}", "{}"))
        except (ValueError, TypeError):
            saved = {}
        for row, (class_id, name) in enumerate(sorted(names.items())):
            data = saved.get(str(class_id), {})
            enabled = QCheckBox()
            enabled.setChecked(bool(data.get("enabled", True)))
            self.table.setCellWidget(row, 0, enabled)
            item = QTableWidgetItem(f"{class_id}: {name}")
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            item.setData(Qt.UserRole, class_id)
            self.table.setItem(row, 1, item)
            target = QLineEdit(str(data.get("target", name)))
            self.table.setCellWidget(row, 2, target)
            confidence = QDoubleSpinBox()
            confidence.setRange(0.001, 1.0)
            confidence.setDecimals(3)
            confidence.setSingleStep(0.01)
            confidence.setValue(float(data.get("confidence", 0.25)))
            self.table.setCellWidget(row, 3, confidence)
        self.class_hint.setText(f"已读取 {len(names)} 个模型类别。输出类别按本表第一次出现的名称从 0 连续编号。")
        self.status.setText("模型类别已读取")

    def _collect(self):
        if not self.model_names or self.model_edit.text().strip() != self.loaded_path:
            raise ValueError("请先读取当前模型的类别")
        selected = []
        target_names = []
        mapping = {}
        confidences = {}
        saved = {}
        for row in range(self.table.rowCount()):
            class_id = int(self.table.item(row, 1).data(Qt.UserRole))
            enabled = self.table.cellWidget(row, 0).isChecked()
            target = self.table.cellWidget(row, 2).text().strip()
            conf = self.table.cellWidget(row, 3).value()
            saved[str(class_id)] = {"enabled": enabled, "target": target, "confidence": conf}
            if not enabled:
                continue
            if not target or "\n" in target:
                raise ValueError(f"类别 {class_id} 的输出名称不能为空或包含换行")
            if target not in target_names:
                target_names.append(target)
            selected.append(class_id)
            mapping[class_id] = target_names.index(target)
            confidences[class_id] = conf
        if not selected:
            raise ValueError("至少勾选一个类别")
        model_path = Path(self.model_edit.text().strip())
        input_dir = Path(self.input_edit.text().strip())
        output_text = self.output_edit.text().strip()
        output_root = Path(output_text)
        if not model_path.is_file():
            raise ValueError("模型文件不存在")
        if not input_dir.is_dir():
            raise ValueError("原图目录不存在")
        if not output_text:
            raise ValueError("请选择输出目录")
        key = hashlib.sha256(self.loaded_path.encode("utf-8")).hexdigest()
        self.settings.setValue(f"classes/{key}", json.dumps(saved, ensure_ascii=False))
        for setting, value in (("model", str(model_path)), ("input", str(input_dir)),
                               ("output", str(output_root)), ("size", self.size_spin.value()),
                               ("recursive", str(self.recursive.isChecked()).lower()),
                               ("preview", str(self.preview.isChecked()).lower()),
                               ("cvat_zip", str(self.cvat_zip.isChecked()).lower())):
            self.settings.setValue(setting, value)
        return RunConfig(
            model_path=model_path, input_dir=input_dir, output_root=output_root,
            image_size=self.size_spin.value(), recursive=self.recursive.isChecked(),
            save_preview=self.preview.isChecked(), create_cvat_zip=self.cvat_zip.isChecked(),
            old_class_names=self.model_names.copy(),
            new_class_names=dict(enumerate(target_names)), select_classes=selected,
            class_map=mapping, class_confidence=confidences,
        )

    def start_batch(self):
        try:
            config = self._collect()
        except Exception as exc:
            QMessageBox.warning(self, "配置需要检查", str(exc))
            return
        self.bar.setValue(0)
        self.log.clear()
        self.status.setText("正在处理图片…")
        self._launch(config)

    def show_progress(self, done, total, message):
        self.bar.setMaximum(total)
        self.bar.setValue(done)
        self.status.setText(f"已处理 {done}/{total} 张")
        self.log.append(f"[{done}/{total}] {message}")

    def show_summary(self, summary):
        self.last_output = summary.run_dir
        self.open_btn.setEnabled(True)
        state = "已停止" if summary.cancelled else "处理完成"
        self.status.setText(f"{state}：成功 {summary.succeeded}，失败 {summary.failed}，标注框 {summary.box_count}")
        self.log.append(f"结果：{summary.run_dir}")
        if summary.failed:
            self.log.append(f"失败详情：{summary.run_dir / 'errors.log'}")
        if summary.zip_path:
            self.log.append(f"CVAT ZIP：{summary.zip_path}")

    def show_error(self, detail):
        self.status.setText("运行失败")
        self.log.append(detail)
        QMessageBox.critical(self, "运行失败", detail.splitlines()[-1])

    def job_finished(self):
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.thread = None
        self.job = None

    def cancel(self):
        if self.job:
            self.job.request_cancel()
            self.status.setText("收到停止请求，当前图片处理完后结束…")
            self.cancel_btn.setEnabled(False)

    def open_output(self):
        if self.last_output and self.last_output.is_dir():
            os.startfile(str(self.last_output))

    def closeEvent(self, event):
        if self.thread and self.thread.isRunning():
            QMessageBox.information(self, "正在处理", "请先点击停止，等待当前图片完成后再关闭。")
            event.ignore()
            return
        event.accept()


def main():
    if len(sys.argv) == 5 and sys.argv[1] == "--self-test":
        sys.exit(self_test(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])))
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = Window()
    window.show()
    sys.exit(app.exec_())


def self_test(model: Path, sample: Path, report: Path) -> int:
    """Headless check of the packaged GUI/worker protocol and one real inference."""
    app = QApplication([])
    report.parent.mkdir(parents=True, exist_ok=True)
    try:
        class_results = []
        errors = []
        query = Job(str(model))
        query.classes_ready.connect(class_results.append)
        query.failed.connect(errors.append)
        query.run()
        if errors or not class_results:
            raise RuntimeError("\n".join(errors) or "没有收到模型类别")
        names = class_results[0]
        with tempfile.TemporaryDirectory(prefix="gui_self_test_", dir=report.parent) as folder:
            root = Path(folder)
            input_dir = root / "images"
            input_dir.mkdir()
            shutil.copy2(sample, input_dir / sample.name)
            config = RunConfig(
                model_path=model, input_dir=input_dir, output_root=root / "output",
                old_class_names=names, new_class_names=names.copy(),
                select_classes=list(names), class_map={i: i for i in names},
                class_confidence={i: 0.1 for i in names},
            )
            summaries = []
            batch = Job(str(model), config)
            batch.completed.connect(summaries.append)
            batch.failed.connect(errors.append)
            batch.run()
            if errors or not summaries:
                raise RuntimeError("\n".join(errors) or "没有收到批处理结果")
            summary = summaries[0]
            if summary.succeeded != 1 or summary.failed != 0:
                raise RuntimeError(f"单张推理未成功: {summary.succeeded} 成功 / {summary.failed} 失败")
            result = {"ok": True, "classes": names, "box_count": summary.box_count}
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0
    except Exception:
        report.write_text(json.dumps({"ok": False, "error": traceback.format_exc()},
                                     ensure_ascii=False, indent=2), encoding="utf-8")
        return 1
    finally:
        app.quit()


if __name__ == "__main__":
    main()
