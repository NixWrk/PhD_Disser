# Однородная серия с электродами

Сгенерировано файлом `java/comsol_homogeneous_electrode_study_6_0.java`.

Во всех расчетах тело однородное: каждому импортированному STEP-домену задано
`rho_soft = 4.728[ohm*m]`, кроме явных металлических электродов/пэдов там, где
они есть.

## Результаты

| Сценарий | ZTT, Ohm | Что менялось |
| --- | ---: | --- |
| `old_domain_electrodes` | 104.2584 | Старые STEP-домены электродов оставлены как металл |
| `window_body_top_faces` | 108.8205 | Boundary Terminal/Ground и voltage windows на body-top faces |
| `point_arm_top` | 111.0224 | Point current source/sink и point voltage probes сверху на цилиндрах рук |
| `smallpads_body_top` | 109.0986 | Добавлены видимые маленькие металлические пэды около body-top координат |
| `smallpads_arm_top` | 111.0231 | Добавлены видимые маленькие металлические пэды сверху на цилиндрах рук |

Полная численная таблица: `homogeneous_electrode_summary.csv`.

Быстрый визуальный обзор: `electrode_layout_contact_sheet.png`.

## Экспорт картинок

Каждый расчет экспортирует `.mph` модель и PNG-скриншоты:

| Суффикс | Смысл |
| --- | --- |
| `_geometry.png` | Вид импортированной геометрии |
| `_matSoft.png` | Выборка материала мягких тканей |
| `_matElectrode.png` | Выборка металлических электродов, если есть явные электродные домены |
| `_phys_term1.png`, `_phys_gnd1.png` | Выборки Terminal и Ground |
| `_phys_pcsSource.png`, `_phys_pcsSink.png` | Выборки point current source/sink для точечной постановки |
| `_cpl_aveVR.png`, `_cpl_aveVL.png` | Выборки усреднения напряжения |

Для быстрой проверки, где именно находятся электроды, начинать удобно с:

- `old_domain_electrodes_matElectrode.png`
- `smallpads_body_top_matElectrode.png`
- `smallpads_arm_top_matElectrode.png`
- `window_body_top_faces_phys_term1.png`
- `point_arm_top_phys_pcsSource.png`

## Интерпретация

Смена вида и положения электродов в однородной модели двигает рассчитанный
трансторакальный импеданс только в диапазоне примерно `104-111 Ohm`. Поэтому
большой разлет с экспериментальным масштабом, скорее всего, не объясняется
одним только расположением электродов.
