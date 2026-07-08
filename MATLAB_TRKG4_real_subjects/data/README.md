# data

Эта папка предназначена для локальных данных испытуемых. STL и CSV сигналов игнорируются git.

Структура:

```text
data/<subject>/
  stl/
    body.stl
    lungs.stl
    heart.stl
    bones.stl
    blood.stl          optional
  electrodes/
    electrodes_4_xyz_mm.csv
  signal/
    optional validation signals
```

Координаты STL и электродов должны быть в одной системе координат.
