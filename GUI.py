#!/usr/bin/env python3
# coding: utf-8

"""
Impedance analysis GUI (PySide6)
- Выбираем папку с .txt файлами (рекурсивно)
- Выбираем файл
- Список каналов (чекбоксы)
- Инверсия ЭКГ / масштабирование
- Диапазон времени (start/end)
- Построение графика (matplotlib embedded)
- Интерфейс для выбора ПРКГ-канала и расчёта половины амплитуды
- Построение тепловой карты медиан скрытых каналов 'Б...'
- Расчёт rho1 по выбранному медианному z и параметрам a,b
"""

from PySide6.QtWidgets import QSlider
from PySide6.QtCore import Qt, QRect, Signal
import os
import re
import sys
import math

from functools import partial
from scipy.signal import butter, filtfilt

import numpy as np
import pandas as pd
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QListWidget, QListWidgetItem, QCheckBox, QComboBox,
    QSpinBox, QDoubleSpinBox, QGroupBox, QFormLayout, QMessageBox
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


# ----------------------------
# Helper functions (logic)
# ----------------------------
comments = {
    1: "Горизонтальный спина",
    5: "Вертикальный спина",
    9: "Мягкие ткани",
    12: "Горизонтальный грудь",
    15: "Вертикальный грудь",
    16: "ЭКГ",
    32: "Трансторакальный",
    'default': "Нет описания"
}


def apply_highpass(signal, fs, cutoff_hz):
    """
    Применяет высокочастотный фильтр (убирает низкие частоты).
    cutoff_hz — частота среза HPF.
    """
    b, a = butter(3, cutoff_hz / (fs / 2), btype="highpass")
    return filtfilt(b, a, signal)


def extract_number(signal_name: str):
    """
    Извлекает число в скобках, например "Канал (13)". Если не найдено — возвращает math.inf
    """
    match = re.search(r'\((\d+)\)', signal_name)
    return int(match.group(1)) if match else math.inf


def get_signal_with_comment(signal_name: str) -> str:
    """
    Возвращает метку сигнала с комментарием из словаря comments.
    Например: "ЭКГ (16) — ЭКГ"
    """
    number = extract_number(signal_name)
    comment = comments.get(number, comments['default'])
    return f"{signal_name} — {comment}"


def get_comment_only(col: str) -> str:
    """
    Возвращает только комментарий сигнала (в нижнем регистре),
    например 'горизонтальный грудь', 'мягкие ткани', 'экг'
    """
    full = get_signal_with_comment(col)
    if "—" in full:
        return full.split("—", 1)[1].strip().lower()
    return full.lower()


def is_vertical(col: str) -> bool:
    return "вертикальный" in get_comment_only(col)


def is_horizontal(col: str) -> bool:
    return "горизонтальный" in get_comment_only(col)


def is_soft_tissue(col: str) -> bool:
    return "мягкие ткани" in get_comment_only(col)


def is_ecg(col: str) -> bool:
    return "экг" in get_comment_only(col)


def find_txt_files(folder: str):
    txt_files = []
    for root, _, files in os.walk(folder):
        if "checkpoint" in root:
            continue
        for file in files:
            if file.lower().endswith('.txt') and "checkpoint" not in file.lower():
                txt_files.append(os.path.join(root, file))
    return sorted(txt_files)


def read_txt_safely(path, delimiter="\t"):
    """
    Пытается открыть файл в нескольких кодировках, чтобы избежать ошибок UTF-8.
    Возвращает DataFrame или выбрасывает исключение.
    """
    encodings_to_try = ["utf-8", "utf-8-sig", "cp1251", "latin1", "ascii"]

    last_error = None
    for enc in encodings_to_try:
        try:
            return pd.read_csv(path, delimiter=delimiter, encoding=enc)
        except Exception as e:
            last_error = e
            continue

    raise UnicodeDecodeError(
        "Не удалось прочитать файл никакой из кодировок",
        last_error.args[0] if last_error else "",
        0, 0, str(last_error)
    )


class RangeSlider(QtWidgets.QWidget):
    """
    Простой двухползунковый горизонтальный слайдер:
    - min_value, max_value — текущие значения
    """
    valueChanged = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self._min = 0
        self._max = 1000

        self._left = 200
        self._right = 800

        self.left_slider = QSlider(Qt.Horizontal)
        self.right_slider = QSlider(Qt.Horizontal)

        self.left_slider.setMinimum(self._min)
        self.left_slider.setMaximum(self._max)
        self.right_slider.setMinimum(self._min)
        self.right_slider.setMaximum(self._max)

        self.left_slider.setValue(self._left)
        self.right_slider.setValue(self._right)

        layout = QVBoxLayout(self)
        layout.addWidget(self.left_slider)
        layout.addWidget(self.right_slider)
        layout.setContentsMargins(0, 0, 0, 0)

        self.left_slider.valueChanged.connect(self._sync)
        self.right_slider.valueChanged.connect(self._sync)

    def setRange(self, a, b):
        self._min = a
        self._max = b
        self.left_slider.setRange(a, b)
        self.right_slider.setRange(a, b)

    def setValues(self, left, right):
        self.left_slider.setValue(left)
        self.right_slider.setValue(right)

    def _sync(self):
        L = self.left_slider.value()
        R = self.right_slider.value()
        if L > R:
            if self.sender() is self.left_slider:
                self.right_slider.setValue(L)
            else:
                self.left_slider.setValue(R)
        self.valueChanged.emit(self.left_slider.value(),
                               self.right_slider.value())


# ----------------------------
# Main Window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Impedance / Signals Analyzer")
        self.resize(1200, 800)

        # default sampling rate
        self.fs = 500.0

        # default geometric parameters (meters)
        self.fat_a = 70/1000
        self.fat_b = 35/1000
        self.heart_a = 90/1000
        self.heart_b = 45/1000

        # data
        self.folder_path = None
        self.txt_files = []
        self.df = pd.DataFrame()
        self.time = np.array([])

        # GUI
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left panel: controls
        left = QVBoxLayout()
        main_layout.addLayout(left, 0)

        # Folder select
        folder_box = QGroupBox("Каталог с .txt файлами")
        folder_layout = QHBoxLayout(folder_box)
        self.folder_label = QLabel("Не выбран")
        btn_browse = QPushButton("Обзор...")
        btn_browse.clicked.connect(self.choose_folder)
        folder_layout.addWidget(self.folder_label)
        folder_layout.addWidget(btn_browse)
        left.addWidget(folder_box)

        # File selection
        file_box = QGroupBox("Выбор файла")
        file_layout = QHBoxLayout(file_box)
        self.file_combo = QComboBox()
        self.file_combo.currentIndexChanged.connect(self.on_file_selected)
        file_layout.addWidget(self.file_combo)
        left.addWidget(file_box)

        # Channels list (checkable)
        channels_box = QGroupBox("Каналы (отметьте для отображения)")
        channels_layout = QVBoxLayout(channels_box)
        self.channels_list = QListWidget()
        self.channels_list.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection)
        channels_layout.addWidget(self.channels_list)
        left.addWidget(channels_box, 1)

        # ECG options & time range
        opts_box = QGroupBox("Параметры отображения")
        opts_layout = QFormLayout(opts_box)

        # soft_tissue_filter
        self.use_filter_coeff_cb = QCheckBox(
            "Использовать коэффициент фильтрации")
        opts_layout.addRow(self.use_filter_coeff_cb)

        self.invert_ecg_cb = QCheckBox("Инвертировать ЭКГ")
        self.invert_ecg_cb.setChecked(True)
        opts_layout.addRow(self.invert_ecg_cb)

        self.hp_filter_cb = QCheckBox("Убрать низкие частоты (<0,5 Hz)")
        self.hp_filter_cb.setChecked(False)

        self.hp_cutoff_spin = QDoubleSpinBox()
        self.hp_cutoff_spin.setRange(0.01, 50.0)
        self.hp_cutoff_spin.setValue(0.5)  # 0.5 Hz default HPF cutoff
        self.hp_cutoff_spin.setDecimals(2)

        opts_layout.addRow(self.hp_filter_cb)
        opts_layout.addRow("Срез HPF (Гц):", self.hp_cutoff_spin)

        self.ecg_scale_spin = QDoubleSpinBox()
        self.ecg_scale_spin.setRange(0.01, 1000.0)
        self.ecg_scale_spin.setValue(50.0)
        opts_layout.addRow(QLabel("Масштаб ЭКГ (делитель):"),
                           self.ecg_scale_spin)

        # time controls
        time_hbox = QWidget()
        th = QHBoxLayout(time_hbox)
        self.start_time_spin = QDoubleSpinBox()
        self.start_time_spin.setRange(0.0, 1e6)
        self.start_time_spin.setDecimals(2)
        self.start_time_spin.setSingleStep(0.1)
        self.start_time_spin.setKeyboardTracking(True)
        self.end_time_spin = QDoubleSpinBox()
        self.end_time_spin.setRange(0.0, 1e6)
        self.end_time_spin.setDecimals(2)
        self.end_time_spin.setSingleStep(0.1)
        self.end_time_spin.setKeyboardTracking(True)
        th.addWidget(QLabel("Start (s):"))
        th.addWidget(self.start_time_spin)
        th.addWidget(QLabel("End (s):"))
        th.addWidget(self.end_time_spin)
        opts_layout.addRow(time_hbox)

        left.addWidget(opts_box)

        # Buttons
        btns_box = QWidget()
        bb = QHBoxLayout(btns_box)
        self.show_plot_btn = QPushButton("Показать график")
        self.show_plot_btn.setStyleSheet(
            "background-color: #4CAF50; color: white;")
        self.show_plot_btn.clicked.connect(self.on_show_plot)
        self.heatmap_btn = QPushButton("Показать медианы (heatmap)")
        self.heatmap_btn.clicked.connect(self.on_heatmap)
        bb.addWidget(self.show_plot_btn)
        bb.addWidget(self.heatmap_btn)
        left.addWidget(btns_box)

        # PRKG amplitude group
        prkg_box = QGroupBox(
            "PRKG — расчёт половины амплитуды (half_amplitude_Z)")
        prkg_layout = QVBoxLayout(prkg_box)
        prkg_top = QHBoxLayout()
        self.prkg_combo = QComboBox()
        self.prkg_range_start = QDoubleSpinBox()
        self.prkg_range_start.setRange(0, 1e6)
        self.prkg_range_start.setDecimals(2)
        self.prkg_range_end = QDoubleSpinBox()
        self.prkg_range_end.setRange(0, 1e6)
        self.prkg_range_end.setDecimals(2)
        prkg_top.addWidget(QLabel("Канал:"))
        prkg_top.addWidget(self.prkg_combo)
        prkg_layout.addLayout(prkg_top)
        prkg_range_h = QHBoxLayout()
        prkg_range_h.addWidget(QLabel("Start (s):"))
        prkg_range_h.addWidget(self.prkg_range_start)
        prkg_range_h.addWidget(QLabel("End (s):"))
        prkg_range_h.addWidget(self.prkg_range_end)
        prkg_layout.addLayout(prkg_range_h)
        self.prkg_save_btn = QPushButton("Сохранить амплитуду")
        self.prkg_save_btn.clicked.connect(self.on_save_prkg)
        prkg_layout.addWidget(self.prkg_save_btn)
        self.prkg_result_label = QLabel("half_amplitude_Z: —")
        prkg_layout.addWidget(self.prkg_result_label)
        left.addWidget(prkg_box)

        # Physics params and rho1
        phys_box = QGroupBox("Физические параметры и rho1")
        phys_layout = QFormLayout(phys_box)
        self.fat_a_spin = QDoubleSpinBox()
        self.fat_a_spin.setRange(0.0, 1.0)
        self.fat_a_spin.setDecimals(6)
        self.fat_b_spin = QDoubleSpinBox()
        self.fat_b_spin.setRange(0.0, 1.0)
        self.fat_b_spin.setDecimals(6)
        self.heart_a_spin = QDoubleSpinBox()
        self.heart_a_spin.setRange(0.0, 1.0)
        self.heart_a_spin.setDecimals(6)
        self.heart_b_spin = QDoubleSpinBox()
        self.heart_b_spin.setRange(0.0, 1.0)
        self.heart_b_spin.setDecimals(6)
        # set defaults from original script
        self.fat_a_spin.setValue(70/1000)
        self.fat_b_spin.setValue(35/1000)
        self.heart_a_spin.setValue(90/1000)
        self.heart_b_spin.setValue(45/1000)
        phys_layout.addRow("fat_a (m):", self.fat_a_spin)
        phys_layout.addRow("fat_b (m):", self.fat_b_spin)
        phys_layout.addRow("heart_a (m):", self.heart_a_spin)
        phys_layout.addRow("heart_b (m):", self.heart_b_spin)
        # z selection from computed median table
        z_h = QHBoxLayout()
        self.z_file_combo = QComboBox()
        self.z_channel_combo = QComboBox()
        z_h.addWidget(QLabel("Файл:"))
        z_h.addWidget(self.z_file_combo)
        z_h.addWidget(QLabel("Канал:"))
        z_h.addWidget(self.z_channel_combo)
        phys_layout.addRow(z_h)
        self.rho1_btn = QPushButton("Вычислить rho1 из выбранного z")
        self.rho1_btn.clicked.connect(self.on_compute_rho1)
        phys_layout.addRow(self.rho1_btn)
        self.rho1_label = QLabel("rho1: —")
        phys_layout.addRow(self.rho1_label)
        left.addWidget(phys_box)

        # Right panel: plots
        right = QVBoxLayout()
        main_layout.addLayout(right, 1)

        # Matplotlib canvas
        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.fig)
        right.addWidget(self.canvas, 8)

        # Log / messages area
        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        right.addWidget(self.log_text, 2)

        # Status bar like
        self.statusBar().showMessage("Готово")

    # -------------------------
    # UI actions
    # -------------------------
    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Выберите папку с .txt файлами", os.path.expanduser("~"))
        if not folder:
            return
        self.folder_path = folder
        self.folder_label.setText(folder)
        self.txt_files = find_txt_files(folder)
        self.file_combo.clear()
        for p in self.txt_files:
            self.file_combo.addItem(os.path.basename(p), p)
        self.log(f"Найдено файлов: {len(self.txt_files)}")
        # update median-related combos
        self.update_median_combos()

    def log(self, msg: str):
        self.log_text.append(msg)
        self.statusBar().showMessage(msg, 5000)

    def on_file_selected(self, index):
        if index < 0:
            return
        path = self.file_combo.itemData(index)
        if not path:
            return
        try:
            self.load_file(path)
        except Exception as e:
            QMessageBox.critical(
                self, "Ошибка", f"Не удалось загрузить файл:\n{e}")

    def load_file(self, file_path):
        # read
        self.log(f"Загрузка {file_path}...")
        df_local = read_txt_safely(file_path).dropna(axis=1, how='all')
        self.df = df_local
        n = len(df_local)
        self.time = np.arange(n) / self.fs
        # populate channels list
        self.channels_list.clear()
        self.prkg_combo.clear()
        for col in df_local.columns:
            it = QListWidgetItem(col)
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            # default: channels starting with 'Б' скрыты (unchecked), others checked
            if str(col).startswith('Б'):
                it.setCheckState(QtCore.Qt.Unchecked)
            else:
                it.setCheckState(QtCore.Qt.Checked)
            self.channels_list.addItem(it)
            # if PRKG channel -> add to prkg selector
            if str(col).startswith('ПРКГ'):
                self.prkg_combo.addItem(col)
        # set time spin ranges
        duration = n / self.fs if n > 0 else 0
        self.start_time_spin.setRange(0.0, max(0.0, duration))
        self.end_time_spin.setRange(0.0, max(0.0, duration))
        self.start_time_spin.setValue(0.0)
        self.end_time_spin.setValue(round(min(60.0, duration), 2))
        # prkg default range
        self.prkg_range_start.setRange(0.0, duration)
        self.prkg_range_end.setRange(0.0, duration)
        self.prkg_range_start.setValue(0.0)
        self.prkg_range_end.setValue(round(min(60.0, duration), 2))
        self.log(
            f"Загружено: {len(df_local.columns)} каналов, длительность {duration:.2f} с")

    def _get_selected_channels(self):
        cols = []
        for i in range(self.channels_list.count()):
            it = self.channels_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                cols.append(it.text())
        return cols

    def on_show_plot(self):
        if self.df.empty:
            QMessageBox.warning(self, "Нет данных", "Сначала выберите файл.")
            return
        selected = self._get_selected_channels()
        if not selected:
            QMessageBox.warning(self, "Нет каналов",
                                "Выберите хотя бы один канал для отображения.")
            return
        start = float(self.start_time_spin.value())
        end = float(self.end_time_spin.value())
        if end <= start:
            QMessageBox.warning(
                self, "Время", "Конечное время должно быть больше начального.")
            return

        # Найдём сигнал "Мягкие ткани", если требуется
        use_filter = self.use_filter_coeff_cb.isChecked()

        if use_filter:
            # ищем колонку с номером (9), которая в comments = "Мягкие ткани"
            soft_col = None
            for col in self.df.columns:
                if is_soft_tissue(col):
                    soft_col = col
                    break

            if soft_col is None:
                QMessageBox.warning(
                    self, "Ошибка",
                    "Вы включили коэффициент фильтрации, но канал с комментарием 'Мягкие ткани' не найден."
                )
                return

            soft_tissue_raw = self.df[soft_col].astype(float).to_numpy()

            # коэффициент фильтрации
            try:
                heart_a = float(self.heart_a_spin.value())
                fat_a = float(self.fat_a_spin.value())
                if fat_a == 0:
                    raise ZeroDivisionError
                filter_coeff = heart_a / fat_a
            except:
                QMessageBox.warning(
                    self, "Ошибка", "Неверные значения heart_a или fat_a.")
                return

            # коэффициент фильтрации: heart_a / fat_a
            try:
                heart_a = float(self.heart_a_spin.value())
                fat_a = float(self.fat_a_spin.value())
                if fat_a == 0:
                    raise ZeroDivisionError
                filter_coeff = heart_a / fat_a
            except:
                QMessageBox.warning(
                    self, "Ошибка", "Неверные значения heart_a или fat_a.")
                return

        # prepare data
        long_data = []
        for col in selected:
            signal = self.df[col].copy().astype(float)
            # Коррекция по коэффициенту фильтрации
            filtered_flag = False
            if use_filter and (is_vertical(col) or is_horizontal(col)):
                signal = signal - soft_tissue_raw * filter_coeff
                filtered_flag = True

            # High-pass filter
            if self.hp_filter_cb.isChecked():
                cutoff = float(self.hp_cutoff_spin.value())
                signal = apply_highpass(signal, self.fs, cutoff)

            if self.invert_ecg_cb.isChecked() and str(col).startswith('ЭКГ'):
                signal = -signal
            if str(col).startswith('ЭКГ'):
                signal = signal / float(self.ecg_scale_spin.value())

            # --- Правильная метка ---
            full_label = get_signal_with_comment(col)
            if filtered_flag:
                full_label += "_Filtered"

            # --- записываем в long-data ---
            df_temp = pd.DataFrame({
                'Time': self.time,
                'Value': signal,
                'Signal': col,
                'SignalFull': full_label
            })
            long_data.append(df_temp)

        if not long_data:
            QMessageBox.warning(self, "Нет данных",
                                "Данные отсутствуют после фильтрации.")
            return
        combined = pd.concat(long_data, ignore_index=True)
        mask = (combined['Time'] >= start) & (combined['Time'] <= end)
        combined = combined.loc[mask]

        # Получаем минимальное значение по НЕ ЭКГ каналам
        other_signals = combined[~combined["Signal"].apply(
            lambda x: is_ecg(x))]
        if not other_signals.empty:
            other_min = other_signals["Value"].min()
        else:
            other_min = 0
        # Находим максимум ЭКГ
        ecg_signals = combined[combined["Signal"].apply(lambda x: is_ecg(x))]
        if not ecg_signals.empty:
            ecg_max = ecg_signals["Value"].max()
            margin = 20  # пикселей запаса, можно изменить
            shift = (other_min - margin) - ecg_max

            # применяем сдвиг
            combined.loc[combined["Signal"].str.startswith(
                "ЭКГ"), "Value"] += shift

        # plot with matplotlib
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        for name, grp in combined.groupby('SignalFull'):
            ax.plot(grp['Time'], grp['Value'], label=name)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Value")
        ax.legend(loc='upper right', fontsize='small')
        ax.set_title(os.path.basename(self.file_combo.currentData()))
        self.canvas.draw()
        self.log("График построен")

    def on_heatmap(self):
        # compute medians of 'Б' channels across all files in folder
        if not self.txt_files:
            QMessageBox.warning(self, "Нет файлов",
                                "Сначала выберите папку с .txt файлами.")
            return
        median_values = {}
        for file_path in self.txt_files:
            try:
                df_local = read_txt_safely(file_path).dropna(axis=1, how='all')
                hidden_B_columns = [
                    col for col in df_local.columns if str(col).startswith('Б')]
                if not hidden_B_columns:
                    continue
                medians = {col: float(df_local[col].median(
                )) for col in hidden_B_columns if col in df_local.columns}
                if medians:
                    median_values[generate_title_text(file_path)] = medians
            except Exception as e:
                self.log(f"Ошибка при обработке {file_path}: {e}")

        if not median_values:
            QMessageBox.information(
                self, "Медианы", "Не найдено скрытых каналов 'Б' в доступных файлах.")
            return

        median_df = pd.DataFrame(median_values).T.fillna(np.nan)
        # draw heatmap
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        sns.heatmap(median_df, annot=True, fmt=".2f", cmap="YlGnBu",
                    ax=ax, cbar_kws={'label': 'Медианное значение'})
        ax.set_title("Медианные значения базового импеданса")
        ax.set_xlabel("Каналы")
        ax.set_ylabel("Файлы")
        self.canvas.draw()
        self.log("Тепловая карта медиан построена")

        # update z selection combos
        self.update_median_combos(median_df)

    def update_median_combos(self, median_df: pd.DataFrame = None):
        """
        Обновляет combobox'ы для выбора файла/канала z. Если median_df не передан,
        пытается пересчитать по self.txt_files (быстро: только имена)
        """
        self.z_file_combo.clear()
        self.z_channel_combo.clear()
        # If median_df provided, fill combos with its index/columns
        if isinstance(median_df, pd.DataFrame):
            for idx in median_df.index:
                self.z_file_combo.addItem(idx)
            for col in median_df.columns:
                self.z_channel_combo.addItem(col)
            self._last_median_df = median_df
        else:
            # attempt to prefill with filenames and channels found in current file
            if self.txt_files:
                # use file basenames
                for p in self.txt_files:
                    self.z_file_combo.addItem(generate_title_text(p))
            if not self.df.empty:
                for col in self.df.columns:
                    if str(col).startswith('Б'):
                        self.z_channel_combo.addItem(col)

    def on_save_prkg(self):
        if self.df.empty:
            QMessageBox.warning(self, "Нет данных", "Сначала загрузите файл.")
            return
        col = self.prkg_combo.currentText()
        if not col:
            QMessageBox.warning(self, "Нет PRKG", "В файле нет каналов ПРКГ.")
            return
        start = float(self.prkg_range_start.value())
        end = float(self.prkg_range_end.value())
        if end <= start:
            QMessageBox.warning(
                self, "Время", "Конечное время должно быть больше начального.")
            return
        sidx = int(start * self.fs)
        eidx = int(end * self.fs)
        sig = self.df[col].iloc[sidx:eidx].astype(float)
        amplitude = float(sig.max() - sig.min())
        half_amp = amplitude / 2.0 / 1000.0  # как в оригинале: делим на 1000
        self.prkg_result_label.setText(f"half_amplitude_Z = {half_amp:.6e} Ом")
        self.half_amplitude_Z = half_amp
        self.log(
            f"PRKG {col}: amplitude={amplitude:.3f}, half_amplitude_Z={half_amp:.6e}")

    def on_compute_rho1(self):
        # need a z from the median table
        if not hasattr(self, "_last_median_df") or self._last_median_df is None:
            QMessageBox.warning(
                self, "Нет медиан", "Сначала постройте/выберите медиану (heatmap).")
            return
        file_key = self.z_file_combo.currentText()
        channel = self.z_channel_combo.currentText()
        if not file_key or not channel:
            QMessageBox.warning(
                self, "Выбор z", "Выберите файл и канал для z.")
            return
        try:
            z = float(self._last_median_df.loc[file_key, channel])
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось получить z: {e}")
            return

        # read geometry from spins
        fat_a = float(self.fat_a_spin.value())
        fat_b = float(self.fat_b_spin.value())
        # Use formula: rho1 = (z * pi * (a^2 - b^2)) / (2 * b)
        rho1 = (z * math.pi * (fat_a ** 2 - fat_b ** 2)) / (2.0 * fat_b)
        self.rho1_label.setText(f"rho1: {rho1:.6e} Ом·м")
        self.log(f"Computed rho1 from z={z}: rho1={rho1:.6e}")

# ----------------------------
# Utilities
# ----------------------------


def generate_title_text(path):
    base = os.path.basename(os.path.splitext(path)[0])
    parent = os.path.basename(os.path.dirname(path))
    grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
    return f"({grandparent}_{parent}_{base})"


# ----------------------------
# Main
# ----------------------------
def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
