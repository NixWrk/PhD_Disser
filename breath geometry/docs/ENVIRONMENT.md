# Окружение и внешние инструменты

## Основная стратегия

Локальная разработка выполняется в .venv на Python 3.11. Точные версии Python-пакетов после установки фиксируются в requirements.lock.txt. Медицинские данные не входят в environment и не копируются внутрь репозитория.

## Уровни инструментов

### Установить сейчас

- Python 3.11;
- pydicom;
- nibabel;
- SimpleITK;
- NumPy/SciPy/pandas;
- Pydantic/Typer;
- pytest/ruff/mypy;
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
~~~

## Внешние GUI/solver

3D Slicer и FEBio не устанавливаются молча из исследовательского кода. Их установочный источник, версия, лицензия и путь фиксируются отдельно. ANTs на Windows желательно изолировать через WSL/container или проверенный binary distribution, а Python-код должен обращаться к нему через adapter.

## Toolchain lock

Файл tools/toolchain.lock.yaml хранит ожидаемые версии и SHA-256 загружаемых архивов. Сами бинарные файлы находятся в tools/bin и исключены из Git.
