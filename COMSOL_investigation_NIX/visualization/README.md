# Экспорт визуализаций COMSOL

Сгенерировано из текущих диагностических COMSOL 6.0 MPH-файлов в `C:\tmp`:

- `TTrans_NIX_point_radius_6_0.mph`
- `TTrans_NIX_window_radius_6_0.mph`

Это диагностические картинки для проверки постановки, не иллюстрации для
публикации.

## Point-модель

- [point_geometry.png](point_geometry.png) - общий вид геометрии.
- [point_mat_soft.png](point_mat_soft.png) - выборка материала soft/body.
- [point_mat_lung.png](point_mat_lung.png) - домены легких внутри тела.
- [point_mat_myocardium.png](point_mat_myocardium.png) - домен myocardium/sphere shell.
- [point_mat_blood.png](point_mat_blood.png) - домен blood/inner sphere.
- [point_mat_electrode.png](point_mat_electrode.png) - выборка материала электродов.
- [point_ec_physics.png](point_ec_physics.png) - выборки физики Electric Currents.

## Window-модель

- [window_geometry.png](window_geometry.png) - общий вид геометрии.
- [window_mat_soft.png](window_mat_soft.png) - выборка материала soft/body.
- [window_mat_lung.png](window_mat_lung.png) - домены легких внутри тела.
- [window_mat_myocardium.png](window_mat_myocardium.png) - домен myocardium/sphere shell.
- [window_mat_blood.png](window_mat_blood.png) - домен blood/inner sphere.
- [window_mat_electrode.png](window_mat_electrode.png) - выборка материала электродов.
- [window_ec_physics.png](window_ec_physics.png) - выборки физики Electric Currents.

## Быстрое чтение

Внешняя геометрия явно все еще содержит старые электродные цилиндры/отверстия.
В однородных проверках этим частям можно задать ту же проводимость, что и мягким
тканям, но сама геометрия все равно не становится чистым однородным фантомом.
