# Breath Geometry — завершённая историческая гипотеза

> **Статус на 23.08.2026:** направление завершено и хранится как история.
> Расчёты проверяли гипотезу об изменении формы и толщины мягких тканей между
> вдохом и выдохом. После литературного обзора по УЗИ, подтвердившего минимальные
> изменения толщины, дальнейшая математическая сложность признана
> неоправданной для основной задачи.

Код, документы и результаты этой папки фиксируют ход гипотезы и отрицательный
исследовательский результат. Они не являются активным ядром диссертации и не
должны автоматически переноситься в `Colab Notebooks` или возобновляться без
новой научной мотивации.

Старые roadmap, project status, notebook outputs и численные оценки следует
читать как исторические записи своего этапа, а не как текущий план и не как
окончательно валидированный собственный результат.

Полный отчёт: [AUDIT_2026-08-23.md](AUDIT_2026-08-23.md).

## Ранее зафиксированное техническое описание

Ниже сохранена документация исследовательской попытки для воспроизводимости и
понимания принятых решений. Её активные формулировки подчиняются текущему
историческому статусу выше.

---

# Breath Geometry

Исследовательский конвейер для оценки изменения грудной клетки и лёгких между вдохом и
выдохом и для построения ансамбля возможных геометрий выдоха по одной КТ вдоха.

Проект связан с четырёхэлектродными TRKG-моделями из соседнего
`MATLAB_TRKG4_real_subjects`. Итоговая цель — оценивать не только геометрическую ошибку в
миллиметрах, но и её влияние на вычисленный импеданс.

---

## Текущий статус коротко

| Трек | Состояние | Ключевой итог |
|---|---|---|
| Реестр данных и пар | **работает** | 40/40 полных пар, protocol/FOV/phase labels, checksum |
| Профили кожа→лёгкое | **работает** | шум `0.19 мм` на субъект, `0.41 мм` на бин |
| Карта `h` внутри фазы | **работает** | устойчива на `93–94%` площади, зоны различимы |
| Регистрация лёгких | **не проходит gate** | elastix и ConvexAdam `0/13`, S1.0–S1.2 и J1.2 отвергнуты |
| Лагранжева карта `Δh` | **закрыта отрицательно** | рёберная система координат недоступна, `R1/R2/R3 = 1/0/0 из 10` |
| Эйлерова карта `Δh` | построена, **не интерпретируема** | содержит артефакт `1.4–12 мм` при эффекте того же порядка |
| Ансамбль выдоха для добровольцев | не начат | заблокирован отсутствием подтверждённого соответствия |

**Главный вывод на сегодня.** Метод состоятелен как измеритель распределения толщины
мягких тканей по всей поверхности тела **внутри одной фазы** и несостоятелен как
измеритель её изменения между фазами на парах задержек дыхания с шагом среза `2.5 мм`.
Развёрнуто — [docs/WALL_TRACK_SUMMARY.md](docs/WALL_TRACK_SUMMARY.md).

Причина не в реализации: сдвиг ткани между фазами `12–42 мм` по осевой компоненте при
межрёберном шаге `20–25 мм` делает рёберную индексацию неоднозначной, и это подтверждается
независимо опубликованными УЗ-данными о значимом изменении ширины межрёберного промежутка
при дыхании.

---

## С чего начать чтение

1. [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md) — что подтверждено и что результатом
   **не** является;
2. [docs/MEASUREMENT_MODEL.md](docs/MEASUREMENT_MODEL.md) — что именно измеряется;
3. [docs/WALL_TRACK_SUMMARY.md](docs/WALL_TRACK_SUMMARY.md) — итог основного трека;
4. [docs/README.md](docs/README.md) — полный указатель документации по группам;
5. [AGENTS.md](AGENTS.md) — правила работы с репозиторием.

---

## Структура репозитория

~~~text
.
├── src/breathgeom/        переиспользуемая логика; всё, что можно протестировать
│   ├── io/                чтение DICOM, DIR-Lab, реестры открытых данных, pair-manifest
│   ├── domain/            физические пространства, LPS/RAS
│   ├── measure/           маски, профили, регистрация, phantom-генераторы
│   ├── uncertainty_w1.py  бюджет неопределённости профиля
│   ├── spatial_map_w2.py  цилиндрическая система координат и разрешение карты
│   ├── rib_frame_r1..r3.py рёберная индексация и её preflight
│   ├── eulerian_delta_e1.py позвоночная карта Δh
│   └── cli.py             единая точка входа `breathgeom`
├── tests/                 инварианты и gate-логика, 218 тестов
├── configs/               замороженные протоколы, по одному JSON на этап
├── schemas/               схемы артефактов
├── notebooks/             отчёты над batch-артефактами, не место реализации
├── docs/                  канонический статус, протоколы, журнал
│   └── archive/           ранние мастер-план, литобзор и протокол сравнения
├── tools/                 bootstrap, установка внешних инструментов, запуск notebook
├── data/                  локальные манифесты, **не коммитится**
└── results/               batch-артефакты, **не коммитится**
~~~

Правило разделения: логика живёт в `src/breathgeom` и покрыта тестами, параметры этапа —
в `configs/`, числа — в `results/`, а notebook только читает артефакт и показывает его.

---

## Установка на Windows

~~~powershell
powershell -ExecutionPolicy Bypass -File tools/bootstrap.ps1 -WithGeometry
powershell -ExecutionPolicy Bypass -File tools/install_dcm2niix.ps1
.\.venv\Scripts\python.exe -m pip install -e ".[registration]"
powershell -ExecutionPolicy Bypass -File tools/setup_registration_env.ps1
Copy-Item configs/paths.local.example.yaml configs/paths.local.yaml
.\.venv\Scripts\breathgeom.exe tools status
.\.venv\Scripts\breathgeom.exe project validate --config configs/paths.local.yaml
~~~

`configs/paths.local.yaml` содержит реальные пути, исключён из Git и не должен попадать в
публикуемые outputs. PyTorch и ConvexAdam ставятся отдельно в `.venv-registration`, чтобы
основная среда тестов не зависела от CUDA.

---

## Команды

### Данные и реестр

~~~powershell
.\.venv\Scripts\breathgeom.exe data list
.\.venv\Scripts\breathgeom.exe data dirlab-inventory "E:\КТ папка\dirlab_copdgene"
.\.venv\Scripts\breathgeom.exe data pairs-manifest `
  --dirlab-root "E:\КТ папка\dirlab_copdgene" `
  --lungct-root "E:\КТ папка\learn2reg_lungct" --checksums
.\.venv\Scripts\breathgeom.exe manifest scan --config configs/paths.local.yaml `
  --output data/interim/manifest.local.csv
~~~

### Регистрация и её QC

~~~powershell
.\.venv\Scripts\breathgeom.exe registration benchmark `
  --manifest data/interim/respiratory_pairs.local.csv `
  --dataset dirlab_copdgene --subject copd1
.\.venv\Scripts\breathgeom.exe registration benchmark `
  --manifest data/interim/respiratory_pairs.local.csv `
  --dataset learn2reg_lungct --subject LungCT_0001 `
  --method convexadam --params configs/convexadam_locked.json `
  --registration-python .venv-registration/Scripts/python.exe
~~~

### Sliding и piecewise-SVF

~~~powershell
.\.venv\Scripts\breathgeom.exe registration sliding-real-development
.\.venv\Scripts\breathgeom.exe registration sliding-s12-heuristic-screen
.\.venv\Scripts\breathgeom.exe registration piecewise-svf-j10-numeric
.\.venv\Scripts\breathgeom.exe registration piecewise-svf-j11-representation
.\.venv\Scripts\breathgeom.exe registration piecewise-svf-j12-preflight
~~~

### Трек грудной стенки

~~~powershell
.\.venv\Scripts\breathgeom.exe measure uncertainty-budget-w1
.\.venv\Scripts\breathgeom.exe measure spatial-map-w2
.\.venv\Scripts\breathgeom.exe measure rib-preflight-r1
.\.venv\Scripts\breathgeom.exe measure rib-preflight-r3
.\.venv\Scripts\breathgeom.exe measure eulerian-delta-e1
.\.venv\Scripts\breathgeom.exe profiles extract-pair `
  --manifest data/interim/respiratory_pairs.local.csv `
  --dataset dirlab_copdgene --subject copd1
.\.venv\Scripts\breathgeom.exe measure wall data/interim/ct.nii.gz --side right
~~~

`registration benchmark` сохраняет поле только для случая, прошедшего все QC-gates. Флаг
`--save-failed-fields` кладёт проваленное поле в локальный
`quarantine_NOT_FOR_MEASUREMENT`, и profile-конвейер такие поля не принимает.
`profiles extract-pair` всегда может выдать раздельные однофазные профили, но создаёт
парные дельты только при наличии прошедшего gate поля.

---

## Проверки перед завершением изменения

~~~powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m mypy src
~~~

Зелёные тесты подтверждают программные инварианты, но не заменяют landmark TRE, проверку
регистрации, сегментации и внешнюю физиологическую валидацию.

---

## Notebooks

Восемнадцать отчётов над batch-артефактами. Числа всегда берутся из `results/`, а не из
notebook. Подробности и порядок запуска — [notebooks/README.md](notebooks/README.md).

`01_inhale_exhale_inspection.ipynb` — исторический прототип для `copd1`, не
перезапускается; его разбор в [docs/NOTEBOOK_AUDIT.md](docs/NOTEBOOK_AUDIT.md).

---

## Ключевые правила

- исходные DICOM неизменяемы и не попадают в Git;
- основное анатомическое представление — маски и поля в физическом пространстве КТ, STL
  является производным;
- source/target space, LPS/RAS и единицы всегда записываются явно;
- respiratory, 4DCT и longitudinal данные не смешиваются;
- импедансная кривая не считается кривой объёма без калибровки;
- одна КТ вдоха даёт ансамбль возможных выдохов, а не единственное доказанное состояние;
- параметры этапа замораживаются **до** вычисления; смягчение порога после просмотра
  чисел запрещено;
- научный результат должен воспроизводиться batch-командой и иметь provenance и QC.

Полные инструкции для разработчиков и автоматизированных агентов — [AGENTS.md](AGENTS.md).
