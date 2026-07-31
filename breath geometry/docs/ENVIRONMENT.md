# Окружение и внешние инструменты

## Основная стратегия

Локальная разработка выполняется в .venv на Python 3.11. Точные версии сторонних
Python-пакетов после установки фиксируются в `requirements.lock.txt`. Editable-запись
самого проекта исключена из lock, чтобы не коммитить абсолютный локальный путь.
Медицинские данные не входят в environment и не копируются внутрь репозитория.

## Уровни инструментов

### Установить сейчас

- Python 3.11;
- pydicom;
- nibabel;
- SimpleITK;
- NumPy/SciPy/pandas;
- Pydantic/Typer;
- pytest/ruff/mypy;
- ipykernel/matplotlib/nbconvert;
- dcm2niix;
- Gmsh.

### Подключить после Evaluation MVP

- ANTs или elastix;
- VTK/PyVista;
- FEBio;
- 3D Slicer;
- Snakemake.

### Подключить только при внешних данных

- PyTorch/MONAI;
- nnU-Net;
- registration networks;
- learned inspiration-to-expiration models.

## Команды

~~~powershell
powershell -ExecutionPolicy Bypass -File tools/bootstrap.ps1 -WithGeometry
powershell -ExecutionPolicy Bypass -File tools/install_dcm2niix.ps1
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\mypy.exe src
powershell -ExecutionPolicy Bypass -File tools/execute_notebook.ps1 `
  -Notebook notebooks/05_sliding_phantom.ipynb
~~~

Если повторный bootstrap выполняется в уже собранной `.venv`, а обновление
`pip/setuptools/wheel` недоступно из-за сети, разрешён
`-SkipPackagingUpgrade`. В этом режиме project install использует `--no-index` и
`--no-build-isolation`: недостающая зависимость приводит к явной ошибке. На чистой машине
этот флаг не следует использовать без предварительно установленного совместимого
toolchain и всех зависимостей.

На Windows Jupyter может вывести предупреждение о дополнительном selector thread для ZMQ
и о локальном TCP kernel. Они не меняют verdict, если процесс завершился с кодом 0 и в
notebook нет output типа `error`. Ошибка записи history не допускается: скрипт направляет
IPython/runtime-файлы в локальную `.venv`.

## Внешние GUI/solver

3D Slicer и FEBio не устанавливаются молча из исследовательского кода. Их установочный источник, версия, лицензия и путь фиксируются отдельно. ANTs на Windows желательно изолировать через WSL/container или проверенный binary distribution, а Python-код должен обращаться к нему через adapter.

## Toolchain lock

Файл tools/toolchain.lock.yaml хранит ожидаемые версии и SHA-256 загружаемых архивов. Сами бинарные файлы находятся в tools/bin и исключены из Git.
