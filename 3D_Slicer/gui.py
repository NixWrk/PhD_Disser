"""
gui.py — Простой GUI для пайплайна сегментации КТ → COMSOL.

Запуск:
    python gui.py
"""

import json
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parent


# ─── Цвета и стиль ──────────────────────────────────────────────────────────

BG        = "#1e1e2e"
BG2       = "#2a2a3e"
BG3       = "#313145"
ACCENT    = "#7c6af7"
ACCENT2   = "#5a4fd4"
GREEN     = "#50fa7b"
YELLOW    = "#f1fa8c"
RED       = "#ff5555"
ORANGE    = "#ffb86c"
CYAN      = "#8be9fd"
FG        = "#cdd6f4"
FG2       = "#a6adc8"
BORDER    = "#45475a"

STATUS_COLOR = {
    "PASS":    GREEN,
    "WARN":    YELLOW,
    "FAIL":    RED,
    "MISSING": FG2,
    "—":       FG2,
}


# ─── Загрузка данных ────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def get_series_list() -> list[dict]:
    """Получает список DICOM-серий через 01_preprocess.py --list."""
    try:
        result = subprocess.run(
            [sys.executable, str(ROOT / "01_preprocess.py"), "--list", "--json"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return json.loads(result.stdout)
    except Exception:
        pass
    return []


def get_qc_status(patient_id: str) -> str:
    """Читает статус QC из JSON-отчёта."""
    cfg = load_config()
    log_dir = ROOT / cfg["paths"]["logs"]
    qc_file = log_dir / f"qc_{patient_id}.json"
    if not qc_file.exists():
        return "—"
    try:
        with open(qc_file, encoding="utf-8") as f:
            return json.load(f).get("status", "—")
    except Exception:
        return "—"


def get_processed_patients() -> list[str]:
    """Возвращает список пациентов у которых есть labelmap."""
    cfg = load_config()
    final_dir = ROOT / cfg["paths"]["seg_final"]
    if not final_dir.exists():
        return []
    return sorted(
        f.name.replace("_labelmap.nii.gz", "")
        for f in final_dir.glob("*_labelmap.nii.gz")
    )


def get_preprocessed_patients() -> list[str]:
    """Возвращает список пациентов у которых есть .nii.gz после препроцессинга."""
    cfg = load_config()
    prep_dir = ROOT / cfg["paths"]["preprocessed"]
    if not prep_dir.exists():
        return []
    return sorted(f.name.replace(".nii.gz", "") for f in prep_dir.glob("*.nii.gz"))


# ─── Главное окно ───────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CT Segmentation Pipeline")
        self.geometry("1100x720")
        self.minsize(900, 600)
        self.configure(bg=BG)

        self._job_thread: threading.Thread | None = None
        self._log_queue: queue.Queue = queue.Queue()
        self._running = False

        self._build_ui()
        self._refresh_patients()
        self._poll_log_queue()

    # ── Построение UI ───────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()

    def _build_left_panel(self):
        left = tk.Frame(self, bg=BG2, width=320)
        left.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        left.grid_propagate(False)
        left.grid_rowconfigure(2, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # Заголовок
        tk.Label(
            left, text="CT Segmentation → COMSOL",
            bg=BG2, fg=ACCENT, font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))

        tk.Label(
            left, text="Пациенты / серии",
            bg=BG2, fg=FG2, font=("Segoe UI", 9),
        ).grid(row=1, column=0, sticky="w", padx=12, pady=(0, 4))

        # Таблица пациентов
        table_frame = tk.Frame(left, bg=BG2)
        table_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 4))
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Treeview",
            background=BG3, foreground=FG, fieldbackground=BG3,
            rowheight=26, borderwidth=0, font=("Consolas", 9),
        )
        style.configure("Treeview.Heading",
            background=BG2, foreground=FG2, borderwidth=0,
            font=("Segoe UI", 9, "bold"),
        )
        style.map("Treeview",
            background=[("selected", ACCENT2)],
            foreground=[("selected", "#ffffff")],
        )

        self.tree = ttk.Treeview(
            table_frame,
            columns=("patient", "status"),
            show="headings",
            selectmode="extended",
        )
        self.tree.heading("patient", text="Пациент")
        self.tree.heading("status", text="QC")
        self.tree.column("patient", width=200, stretch=True)
        self.tree.column("status", width=60, anchor="center", stretch=False)

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        # Кнопки управления пациентами
        btn_row = tk.Frame(left, bg=BG2)
        btn_row.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 4))
        self._btn(btn_row, "Обновить", self._refresh_patients, ACCENT).pack(side="left")
        self._btn(btn_row, "Выбрать все", self._select_all, BG3).pack(side="left", padx=4)

        # Разделитель
        tk.Frame(left, bg=BORDER, height=1).grid(row=4, column=0, sticky="ew", padx=8, pady=4)

        # Шаги пайплайна
        tk.Label(
            left, text="Шаги пайплайна",
            bg=BG2, fg=FG2, font=("Segoe UI", 9),
        ).grid(row=5, column=0, sticky="w", padx=12, pady=(2, 4))

        steps_frame = tk.Frame(left, bg=BG2)
        steps_frame.grid(row=6, column=0, sticky="ew", padx=8)
        steps_frame.grid_columnconfigure(0, weight=1)
        steps_frame.grid_columnconfigure(1, weight=1)

        step_buttons = [
            ("1. Предобработка",  "preprocess",  CYAN),
            ("2. Сегментация",    "segment",     ORANGE),
            ("3. Постобработка",  "postprocess", ACCENT),
            ("4. QC",             "qc",          YELLOW),
            ("5. Экспорт COMSOL", "export",      GREEN),
            ("6. Визуализация",   "visualize",   FG2),
        ]

        for i, (label, step, color) in enumerate(step_buttons):
            row, col = divmod(i, 2)
            btn = self._btn(
                steps_frame, label,
                lambda s=step: self._run_step(s),
                BG3, fg=color,
            )
            btn.grid(row=row, column=col, sticky="ew", padx=3, pady=3)

        # Разделитель
        tk.Frame(left, bg=BORDER, height=1).grid(row=7, column=0, sticky="ew", padx=8, pady=6)

        # Главные кнопки
        big_frame = tk.Frame(left, bg=BG2)
        big_frame.grid(row=8, column=0, sticky="ew", padx=8, pady=(0, 4))
        big_frame.grid_columnconfigure(0, weight=1)

        self._btn(
            big_frame, "▶  Запустить всё",
            self._run_all, ACCENT,
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="ew", pady=3)

        self._btn(
            big_frame, "Сводный отчёт когорты",
            self._run_summary, BG3, fg=CYAN,
        ).grid(row=1, column=0, sticky="ew", pady=3)

        # Опции
        opt_frame = tk.Frame(left, bg=BG2)
        opt_frame.grid(row=9, column=0, sticky="ew", padx=12, pady=(4, 8))

        self._var_fast = tk.BooleanVar(value=False)
        self._var_cpu  = tk.BooleanVar(value=False)
        self._var_mhd  = tk.BooleanVar(value=False)

        for var, label in [
            (self._var_fast, "--fast (быстрее, ↓ качество)"),
            (self._var_cpu,  "--cpu (без GPU)"),
            (self._var_mhd,  "--mhd (экспорт MHD вместо NRRD)"),
        ]:
            tk.Checkbutton(
                opt_frame, text=label, variable=var,
                bg=BG2, fg=FG2, selectcolor=BG3,
                activebackground=BG2, activeforeground=FG,
                font=("Segoe UI", 9),
            ).pack(anchor="w")

    def _build_right_panel(self):
        right = tk.Frame(self, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        # Строка статуса
        status_bar = tk.Frame(right, bg=BG2, height=36)
        status_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        status_bar.grid_propagate(False)
        status_bar.grid_columnconfigure(1, weight=1)

        self._status_dot = tk.Label(status_bar, text="●", bg=BG2, fg=FG2, font=("Segoe UI", 14))
        self._status_dot.grid(row=0, column=0, padx=(10, 4), pady=6)

        self._status_label = tk.Label(
            status_bar, text="Готов",
            bg=BG2, fg=FG2, font=("Segoe UI", 10),
        )
        self._status_label.grid(row=0, column=1, sticky="w")

        self._stop_btn = self._btn(status_bar, "Стоп", self._stop_job, RED)
        self._stop_btn.grid(row=0, column=2, padx=8, pady=4)
        self._stop_btn.config(state="disabled")

        self._btn(status_bar, "Очистить лог", self._clear_log, BG3).grid(
            row=0, column=3, padx=(0, 8), pady=4
        )

        # Лог-панель
        log_frame = tk.Frame(right, bg=BG2, bd=0)
        log_frame.grid(row=1, column=0, sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)

        self._log = tk.Text(
            log_frame,
            bg=BG, fg=FG, insertbackground=FG,
            font=("Consolas", 9),
            wrap="none", bd=0, relief="flat",
            state="disabled",
        )
        self._log.grid(row=0, column=0, sticky="nsew")

        vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self._log.yview)
        hsb = ttk.Scrollbar(log_frame, orient="horizontal", command=self._log.xview)
        self._log.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Цветовые теги для лога
        self._log.tag_config("INFO",    foreground=FG)
        self._log.tag_config("ERROR",   foreground=RED)
        self._log.tag_config("WARNING", foreground=YELLOW)
        self._log.tag_config("OK",      foreground=GREEN)
        self._log.tag_config("HEADER",  foreground=ACCENT, font=("Consolas", 9, "bold"))
        self._log.tag_config("DIM",     foreground=FG2)

    # ── Вспомогательные виджеты ─────────────────────────────────────────────

    def _btn(self, parent, text, command, bg, fg=FG, font=None):
        f = font or ("Segoe UI", 9)
        return tk.Button(
            parent, text=text, command=command,
            bg=bg, fg=fg, activebackground=ACCENT2, activeforeground="#fff",
            relief="flat", padx=8, pady=4,
            font=f, cursor="hand2", bd=0,
        )

    # ── Данные ──────────────────────────────────────────────────────────────

    def _refresh_patients(self):
        self.tree.delete(*self.tree.get_children())

        # Сначала пациенты с labelmap (полностью обработанные)
        processed = set(get_processed_patients())
        preprocessed = set(get_preprocessed_patients())
        all_patients = sorted(processed | preprocessed)

        for pid in all_patients:
            status = get_qc_status(pid) if pid in processed else "—"
            color = STATUS_COLOR.get(status, FG2)
            iid = self.tree.insert("", "end", values=(pid, status), tags=(status,))
            self.tree.tag_configure(status, foreground=color)

        if not all_patients:
            self.tree.insert("", "end", values=("(нет данных)", ""), tags=("DIM",))
            self.tree.tag_configure("DIM", foreground=FG2)

        self._log_line(
            f"Обновлено: {len(all_patients)} пациентов "
            f"({len(processed)} с labelmap, {len(preprocessed - processed)} только nii)\n",
            "DIM",
        )

    def _select_all(self):
        self.tree.selection_set(self.tree.get_children())

    def _get_selected_patients(self) -> list[str]:
        selected = self.tree.selection()
        if not selected:
            return []
        return [self.tree.item(iid)["values"][0] for iid in selected]

    # ── Логирование ─────────────────────────────────────────────────────────

    def _log_line(self, text: str, tag: str = "INFO"):
        self._log.configure(state="normal")
        self._log.insert("end", text, tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _poll_log_queue(self):
        """Переносит строки из очереди в виджет лога (из фонового потока)."""
        try:
            while True:
                line, tag = self._log_queue.get_nowait()
                self._log_line(line, tag)
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    def _classify_line(self, line: str) -> str:
        lo = line.lower()
        if any(w in lo for w in ["error", "ошибка", "fail", "failed", "traceback"]):
            return "ERROR"
        if any(w in lo for w in ["warning", "warn", "внимание"]):
            return "WARNING"
        if any(w in lo for w in ["успешно", "готово", "pass", "ok ", "saved", "сохранено"]):
            return "OK"
        if line.startswith("===") or line.startswith("ШАГ"):
            return "HEADER"
        return "INFO"

    # ── Запуск процессов ────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = FG2):
        self._status_label.config(text=text)
        self._status_dot.config(fg=color)

    def _build_args(self, step: str | None, patient: str | None) -> list[str]:
        args = [sys.executable, str(ROOT / "main.py")]
        if step:
            args += ["--step", step]
        else:
            args += ["--all"]
        if patient:
            args += ["--patient", patient]
        if self._var_fast.get() and step in (None, "segment"):
            args.append("--fast")
        if self._var_cpu.get() and step in (None, "segment"):
            args.append("--cpu")
        if self._var_mhd.get() and step in (None, "export"):
            args.append("--mhd")
        return args

    def _run_step(self, step: str):
        patients = self._get_selected_patients()
        if not patients:
            self._launch_job(step=step, patient=None)
        else:
            for pid in patients:
                self._launch_job(step=step, patient=pid, chain=True)

    def _run_all(self):
        patients = self._get_selected_patients()
        if not patients:
            self._launch_job(step=None, patient=None)
        else:
            for pid in patients:
                self._launch_job(step=None, patient=pid, chain=True)

    def _run_summary(self):
        args = [sys.executable, str(ROOT / "09_cohort_summary.py")]
        self._launch_raw(args, label="Сводный отчёт")

    def _launch_job(self, step: str | None, patient: str | None, chain: bool = False):
        if self._running:
            messagebox.showwarning("Занято", "Дождитесь завершения текущего процесса.")
            return
        args = self._build_args(step, patient)
        label = step or "всё"
        if patient:
            label += f" [{patient}]"
        self._launch_raw(args, label)

    def _launch_raw(self, args: list[str], label: str):
        if self._running:
            return
        self._running = True
        self._stop_btn.config(state="normal")
        self._set_status(f"Выполняется: {label}…", ACCENT)

        self._log_queue.put((f"\n{'='*60}\n", "HEADER"))
        self._log_queue.put((f"  {label}\n", "HEADER"))
        self._log_queue.put((f"{'='*60}\n", "HEADER"))
        self._log_queue.put((f"CMD: {' '.join(args)}\n\n", "DIM"))

        self._proc: subprocess.Popen | None = None

        def worker():
            try:
                self._proc = subprocess.Popen(
                    args,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                )
                for line in self._proc.stdout:
                    tag = self._classify_line(line)
                    self._log_queue.put((line, tag))
                self._proc.wait()
                rc = self._proc.returncode
                if rc == 0:
                    self._log_queue.put((f"\n✓ Завершено успешно\n", "OK"))
                else:
                    self._log_queue.put((f"\n✗ Завершено с кодом {rc}\n", "ERROR"))
            except Exception as e:
                self._log_queue.put((f"\nОШИБКА ЗАПУСКА: {e}\n", "ERROR"))
            finally:
                self._running = False
                self.after(0, self._on_job_done)

        self._job_thread = threading.Thread(target=worker, daemon=True)
        self._job_thread.start()

    def _on_job_done(self):
        self._stop_btn.config(state="disabled")
        self._set_status("Готов", FG2)
        self._refresh_patients()

    def _stop_job(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._log_queue.put(("\n⚠ Процесс остановлен пользователем\n", "WARNING"))


# ─── Точка входа ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
