# NIX_HnL_V.1.0

GUI-приложение на `PySide6` для анализа кардио-респираторных сигналов из `.txt`:

- выбор ЭКГ, канала `i` (прекардиальный) и `j` (первый слой);
- расчёт и отображение:
  - `ECG`,
  - `y_j(t)`,
  - `y_i(t)`,
  - `Filtered = y_i(t) - k*M*y_j(t)`;
- выбор нормы для `M`:
  - `Размах (max-min)`,
  - `L2`,
  - `Отключено (M = 1)`;
- работа по выбранному временному окну `Start..End`;
- настройки ЭКГ: инверсия, масштаб, сдвиг по оси `Y`.

## Запуск из Python

```powershell
python NIX_HnL_V.1.0.py
```

## Зависимости

```powershell
python -m pip install PySide6 matplotlib numpy pandas scipy
```

## Сборка EXE

Сборка делается через `auto-py-to-exe`/`pyinstaller`.

Минимально:

```powershell
auto-py-to-exe
```

или CLI-эквивалент через `pyinstaller`:

```powershell
pyinstaller --noconfirm --windowed --onefile --name NIX_HnL_V1_0 NIX_HnL_V.1.0.py
```

## Примечание по GitHub

`exe`-файлы получаются большими и обычно не хранятся в репозитории.  
Рекомендуется выкладывать их как релизы (`GitHub Releases`) или через внешнее хранилище.
