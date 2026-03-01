#!/usr/bin/env python3
# coding: utf-8

"""
GUI: выбор двух импедансных каналов i, j + коэффициент k (ползунок 0..2, шаг 0.01).
Отрисовка: ЭКГ, y_i(t), y_i(t) - k*M*y_j(t), где M = ||y_i||/||y_j|| по выбранной норме.
"""

import os
import sys
import re
import math
import numpy as np
import pandas as pd

from PySide6 import QtCore, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QPushButton, QComboBox, QDoubleSpinBox, QSlider, QMessageBox, QCheckBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator


# ----------------------------
# I/O helpers
# ----------------------------
def find_txt_files(folder: str):
    out = []
    for root, _, files in os.walk(folder):
        if "checkpoint" in root:
            continue
        for fn in files:
            if fn.lower().endswith(".txt") and "checkpoint" not in fn.lower():
                out.append(os.path.join(root, fn))
    return sorted(out)


def read_txt_safely(path, delimiter="\t"):
    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin1", "ascii"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, delimiter=delimiter, encoding=enc).dropna(axis=1, how="all")
        except Exception as e:
            last_error = e
    raise RuntimeError(f"Не удалось прочитать файл {path}: {last_error}")


def is_ecg_column(col_name: str) -> bool:
    # Стабильный критерий как в вашем коде: имя начинается с 'ЭКГ'
    return str(col_name).startswith("ЭКГ")


def compute_norm(y: np.ndarray, norm_kind: str) -> float:
    """
    norm_kind:
      - 'ptp'  : max-min
      - 'l2'   : sqrt(sum(y^2))
    """
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return float("nan")
    if norm_kind == "ptp":
        return float(np.nanmax(y) - np.nanmin(y))
    if norm_kind == "l2":
        return float(np.sqrt(np.nansum(y * y)))
    raise ValueError(f"Unknown norm_kind={norm_kind}")


def extract_channel_number(col_name: str) -> int:
    m = re.search(r"\((\d+)\)", str(col_name))
    return int(m.group(1)) if m else math.inf


def is_prkg_column(col_name: str) -> bool:
    s = str(col_name)
    return ("\u041f\u0420\u041a\u0413" in s) or ("PRKG" in s.upper())


def is_primary_prkg_column(col_name: str) -> bool:
    s = str(col_name).strip()
    return s.startswith("\u041f\u0420\u041a\u0413") or s.upper().startswith("PRKG")


# ----------------------------
# Main GUI
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Impedance correction: yi - k*M*yj")
        self.resize(1200, 800)

        # параметры
        self.fs = 500.0  # Гц (при необходимости вынести в поле GUI)

        # данные
        self.folder_path = None
        self.txt_files = []
        self.df = pd.DataFrame()
        self.time = np.array([])

        # кеш выбранных сигналов на текущем окне
        self._cache = None
        self._ecg_shift_user_defined = False
        self._updating_ecg_shift = False

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        # -------- Left controls
        left = QVBoxLayout()
        main.addLayout(left, 0)

        # Folder
        folder_box = QGroupBox("Каталог с .txt файлами")
        folder_l = QHBoxLayout(folder_box)
        self.folder_label = QLabel("Не выбран")
        btn_folder = QPushButton("Обзор...")
        btn_folder.clicked.connect(self.choose_folder)
        folder_l.addWidget(self.folder_label)
        folder_l.addWidget(btn_folder)
        left.addWidget(folder_box)

        # File
        file_box = QGroupBox("Файл")
        file_l = QHBoxLayout(file_box)
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_selected)
        file_l.addWidget(self.file_combo)
        left.addWidget(file_box)

        # Channels + norm + k
        sel_box = QGroupBox("Выбор каналов и параметров")
        sel_form = QFormLayout(sel_box)

        self.ecg_combo = QComboBox()
        self.i_combo = QComboBox()
        self.j_combo = QComboBox()

        sel_form.addRow("ЭКГ:", self.ecg_combo)
        sel_form.addRow("Канал i (прекардиальный):", self.i_combo)
        sel_form.addRow("Канал j (первый слой):", self.j_combo)

        # Норма
        self.norm_combo = QComboBox()
        self.norm_combo.addItem("Размах (max-min)", userData="ptp")
        self.norm_combo.addItem("Евклидова норма (L2)", userData="l2")
        self.norm_combo.addItem("Отключено (M = 1)", userData="off")
        self.norm_combo.currentIndexChanged.connect(self.recompute_and_redraw)
        sel_form.addRow("Норма ||y||:", self.norm_combo)
        self.norm_details_label = QLabel("Норма считается по текущему отображаемому отрезку времени Start..End.")
        self.norm_details_label.setWordWrap(True)
        sel_form.addRow(self.norm_details_label)

        # Время
        time_row = QWidget()
        time_l = QHBoxLayout(time_row)
        time_l.setContentsMargins(0, 0, 0, 0)
        self.start_time = QDoubleSpinBox()
        self.end_time = QDoubleSpinBox()
        for sp in (self.start_time, self.end_time):
            sp.setDecimals(2)
            sp.setRange(0.0, 1e9)
            sp.setSingleStep(0.1)
            sp.setKeyboardTracking(True)
        self.start_time.valueChanged.connect(self.recompute_and_redraw)
        self.end_time.valueChanged.connect(self.recompute_and_redraw)

        time_l.addWidget(QLabel("Start (s):"))
        time_l.addWidget(self.start_time)
        time_l.addWidget(QLabel("End (s):"))
        time_l.addWidget(self.end_time)
        sel_form.addRow(time_row)

        # Ползунок k: 0..2 шаг 0.01 => int 0..200
        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setRange(0, 200)
        self.k_slider.setValue(0)
        self.k_slider.valueChanged.connect(self.on_k_changed)

        self.k_label = QLabel("k = 0.00")
        k_row = QWidget()
        k_l = QHBoxLayout(k_row)
        k_l.setContentsMargins(0, 0, 0, 0)
        k_l.addWidget(self.k_label)
        k_l.addWidget(self.k_slider)
        sel_form.addRow("Коэффициент k:", k_row)

        # кнопка принудительной перерисовки (не обяз., но полезна)
        btn_plot = QPushButton("Построить/обновить график")
        btn_plot.clicked.connect(self.recompute_and_redraw)
        sel_form.addRow(btn_plot)

        # ECG display settings
        self.ecg_invert_cb = QCheckBox("Invert ECG")
        self.ecg_invert_cb.setChecked(True)
        self.ecg_invert_cb.stateChanged.connect(self.recompute_and_redraw)
        sel_form.addRow(self.ecg_invert_cb)

        self.ecg_scale_spin = QDoubleSpinBox()
        self.ecg_scale_spin.setRange(0.01, 1000.0)
        self.ecg_scale_spin.setDecimals(2)
        self.ecg_scale_spin.setValue(20.0)
        self.ecg_scale_spin.valueChanged.connect(self.recompute_and_redraw)
        sel_form.addRow("ECG amplitude divisor:", self.ecg_scale_spin)

        self.ecg_shift_spin = QDoubleSpinBox()
        self.ecg_shift_spin.setRange(0.0, 1e9)
        self.ecg_shift_spin.setDecimals(2)
        self.ecg_shift_spin.setValue(0.0)
        self.ecg_shift_spin.valueChanged.connect(self.on_ecg_shift_changed)
        sel_form.addRow("ECG Y shift from 0:", self.ecg_shift_spin)
        legend_box = QGroupBox("Легенда")
        legend_layout = QVBoxLayout(legend_box)
        legend_label = QLabel(
            "Обозначения:\n"
            "ECG: ЭКГ (инверсия/масштаб/сдвиг по Y).\n"
            "y_i(t): прекардиальный канал i.\n"
            "y_j(t): канал первого слоя j.\n"
            "Filtered: y_i(t) - k*M*y_j(t).\n\n"
            "Параметры:\n"
            "k: коэффициент из ползунка.\n"
            "M: нормировочный коэффициент.\n"
            "Нормы всегда считаются по текущему окну Start..End.\n"
            "  ptp: M = (max(y_i)-min(y_i)) / (max(y_j)-min(y_j))\n"
            "  L2:  M = ||y_i||_2 / ||y_j||_2\n"
            "  Отключено: M = 1\n"
            "Тогда Filtered = y_i - k*y_j."
        )
        legend_label.setWordWrap(True)
        legend_layout.addWidget(legend_label)
        left.addWidget(legend_box)
        left.addWidget(sel_box)

        # -------- Right plot
        right = QVBoxLayout()
        main.addLayout(right, 1)

        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        right.addWidget(self.canvas, 10)

        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        right.addWidget(self.log, 3)

        self.statusBar().showMessage("Р“РѕС‚РѕРІРѕ")

        # изменение выбора каналов -> пересчёт
        self.ecg_combo.currentIndexChanged.connect(self.recompute_and_redraw)
        self.i_combo.currentIndexChanged.connect(self.recompute_and_redraw)
        self.j_combo.currentIndexChanged.connect(self.recompute_and_redraw)

    def write_log(self, msg: str):
        self.log.append(msg)
        self.statusBar().showMessage(msg, 5000)

    # ---------------- UI actions ----------------
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с .txt файлами", os.path.expanduser("~"))
        if not folder:
            return
        self.folder_path = folder
        self.folder_label.setText(folder)
        self.txt_files = find_txt_files(folder)

        self.file_combo.clear()
        for p in self.txt_files:
            self.file_combo.addItem(os.path.basename(p), p)

        self.write_log(f"Найдено файлов: {len(self.txt_files)}")

    def on_file_selected(self, index: int):
        if index < 0:
            return
        path = self.file_combo.itemData(index)
        if not path:
            return
        try:
            self.load_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def load_file(self, file_path: str):
        self.write_log(f"Загрузка: {file_path}")
        self.df = read_txt_safely(file_path)
        n = len(self.df)
        self.time = np.arange(n, dtype=float) / float(self.fs)

        # диапазон времени
        duration = float(n) / float(self.fs) if n > 0 else 0.0
        self.start_time.setRange(0.0, max(0.0, duration))
        self.end_time.setRange(0.0, max(0.0, duration))
        self.start_time.setValue(0.0)
        self.end_time.setValue(round(min(60.0, duration), 2))

        # заполнение каналов
        cols = list(self.df.columns)

        self.ecg_combo.clear()
        self.i_combo.clear()
        self.j_combo.clear()

        # ЭКГ: по критерию startswith('ЭКГ'); если нет — оставляем пусто
        ecg_cols = [c for c in cols if is_ecg_column(c)]
        for c in ecg_cols:
            self.ecg_combo.addItem(str(c), userData=str(c))

        # импедансные каналы: всё, кроме ЭКГ (и при желании можно исключить 'Б...' — сейчас не исключаем)
        imp_cols = [c for c in cols if not is_ecg_column(c)]
        for c in imp_cols:
            self.i_combo.addItem(str(c), userData=str(c))
            self.j_combo.addItem(str(c), userData=str(c))

        # Default i/j: prefer PRKG pair where j is the nearest higher channel number.
        if imp_cols:
            i_default = None
            j_default = None
            prkg_cols = sorted(
                [c for c in imp_cols if is_primary_prkg_column(c)],
                key=lambda c: (extract_channel_number(c), str(c))
            )
            if len(prkg_cols) >= 2:
                i_default = prkg_cols[0]
                i_num = extract_channel_number(i_default)
                higher = [c for c in prkg_cols if extract_channel_number(c) > i_num]
                j_default = higher[0] if higher else prkg_cols[1]
            elif len(imp_cols) >= 2:
                i_default = imp_cols[0]
                j_default = imp_cols[1]

            if i_default is not None:
                self.i_combo.setCurrentText(str(i_default))
            if j_default is not None:
                self.j_combo.setCurrentText(str(j_default))

        self.write_log(f"Загружено: {len(cols)} каналов, длительность {duration:.2f} с")
        self._ecg_shift_user_defined = False
        self._cache = None
        self.recompute_and_redraw()

    def get_k(self) -> float:
        return float(self.k_slider.value()) / 100.0

    def on_k_changed(self, _value: int):
        # динамическое обновление: пересчитываем только скорректированный сигнал и перерисовываем
        k = self.get_k()
        self.k_label.setText(f"k = {k:.2f}")
        self.redraw_only_with_new_k()

    def on_ecg_shift_changed(self, _value: float):
        if not self._updating_ecg_shift:
            self._ecg_shift_user_defined = True
            self.recompute_and_redraw()

    # ---------------- core computation ----------------
    def _get_window_indices(self):
        if self.df.empty:
            return None
        start = float(self.start_time.value())
        end = float(self.end_time.value())
        if end <= start:
            return None
        sidx = max(0, int(math.floor(start * self.fs)))
        eidx = min(len(self.df), int(math.ceil(end * self.fs)))
        if eidx <= sidx:
            return None
        return sidx, eidx

    def recompute_and_redraw(self):
        """
        Полный пересчёт: извлечение сигналов на окне, вычисление M, построение графика.
        """
        if self.df.empty:
            return

        idx = self._get_window_indices()
        if idx is None:
            return
        sidx, eidx = idx

        i_col = self.i_combo.currentData()
        j_col = self.j_combo.currentData()
        ecg_col = self.ecg_combo.currentData()  # может быть None

        if not i_col or not j_col:
            return

        yi = self.df[i_col].iloc[sidx:eidx].astype(float).to_numpy()
        yj = self.df[j_col].iloc[sidx:eidx].astype(float).to_numpy()
        win_start = sidx / float(self.fs)
        win_end = eidx / float(self.fs)

        # норма
        norm_kind = self.norm_combo.currentData()
        if norm_kind == "off":
            M = 1.0
            self.norm_details_label.setText(
                f"Норма: отключена на [{win_start:.2f}, {win_end:.2f}] c. "
                f"M = 1.000000, поэтому Filtered = y_i - k*y_j."
            )
        else:
            ni = compute_norm(yi, norm_kind)
            nj = compute_norm(yj, norm_kind)
            if not np.isfinite(ni) or not np.isfinite(nj) or nj == 0.0:
                self.norm_details_label.setText(
                    f"Норма на [{win_start:.2f}, {win_end:.2f}] c не вычислена: "
                    "некорректные значения или ||y_j|| = 0."
                )
                self.write_log("Невозможно вычислить M: некорректная норма или ||y_j|| = 0.")
                return
            M = ni / nj
            if norm_kind == "ptp":
                formula = "M = (max(y_i)-min(y_i)) / (max(y_j)-min(y_j))"
            else:
                formula = "M = ||y_i||_2 / ||y_j||_2"
            self.norm_details_label.setText(
                f"Норма на [{win_start:.2f}, {win_end:.2f}] c: {formula}; "
                f"ni={ni:.6g}, nj={nj:.6g}, M={M:.6g}."
            )
        k = self.get_k()
        ycorr = yi - k * M * yj

        t = self.time[sidx:eidx]

        # ЭКГ (если есть)
        ecg = None
        if ecg_col:
            ecg = self.df[ecg_col].iloc[sidx:eidx].astype(float).to_numpy()

        self._cache = {
            "t": t,
            "yi": yi,
            "yj": yj,
            "ycorr": ycorr,
            "ecg": ecg,
            "M": M,
            "i_col": i_col,
            "j_col": j_col,
            "ecg_col": ecg_col,
            "norm_kind": norm_kind,
        }

        self.draw_plot()

    def redraw_only_with_new_k(self):
        """
        Быстрое обновление при изменении k: не пересчитывает M, только ycorr и график.
        """
        if not self._cache:
            self.recompute_and_redraw()
            return

        k = self.get_k()
        yi = self._cache["yi"]
        t = self._cache["t"]
        i_col = self._cache["i_col"]
        j_col = self._cache["j_col"]
        M = self._cache["M"]

        # Пересчитываем yj только если окно/канал могли измениться без recompute (обычно не нужно),
        # но безопаснее взять текущее состояние.
        idx = self._get_window_indices()
        if idx is None:
            return
        sidx, eidx = idx
        yj = self.df[j_col].iloc[sidx:eidx].astype(float).to_numpy()

        ycorr = yi - k * M * yj
        self._cache["yj"] = yj
        self._cache["ycorr"] = ycorr

        self.draw_plot()

    def draw_plot(self):
        if not self._cache:
            return

        t = self._cache["t"]
        yi = self._cache["yi"]
        yj = self._cache["yj"]
        ycorr = self._cache["ycorr"]
        ecg_raw = self._cache["ecg"]

        i_col = self._cache["i_col"]
        j_col = self._cache["j_col"]
        M = self._cache["M"]
        k = self.get_k()

        self.fig.clear()
        ax = self.fig.add_subplot(111)

        # ECG: invert + amplitude scaling + vertical shift.
        # Default rule: ECG max is 50 units below min(yi, yj, ycorr).
        if ecg_raw is not None:
            ecg = np.array(ecg_raw, dtype=float, copy=True)
            if self.ecg_invert_cb.isChecked():
                ecg = -ecg
            ecg = ecg / float(self.ecg_scale_spin.value())
            signal_min = float(np.nanmin(np.concatenate([yi, yj, ycorr])))
            ecg_max = float(np.nanmax(ecg))
            if not self._ecg_shift_user_defined:
                default_gap = 50.0
                default_down_shift = max(0.0, ecg_max - (signal_min - default_gap))
                self._updating_ecg_shift = True
                self.ecg_shift_spin.setValue(default_down_shift)
                self._updating_ecg_shift = False
            ecg = ecg - float(self.ecg_shift_spin.value())
            ax.plot(t, ecg, label=f"ECG: {self._cache['ecg_col']}",
                    color="#ff7f0e", linewidth=1.8)

        ax.plot(t, yj, label=f"First layer y_j(t): {j_col}",
                color="#1f77b4", linewidth=1.8)
        ax.plot(t, yi, label=f"Precordial y_i(t): {i_col}",
                color="#d62728", linewidth=1.8)
        ax.plot(t, ycorr, label=f"Filtered y_i - k*M*y_j (k={k:.2f}, M={M:.4g})",
                color="#8c564b", linewidth=2.1)

        # Adaptive major grid with limited tick count to avoid overly dense mesh.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=4))
        ax.grid(True, which="major", color="#9aa0a6", alpha=0.35, linewidth=0.8)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize="small")
        ax.set_title(os.path.basename(self.file_combo.currentData()) if self.file_combo.currentData() else "-")

        self.canvas.draw_idle()

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()






