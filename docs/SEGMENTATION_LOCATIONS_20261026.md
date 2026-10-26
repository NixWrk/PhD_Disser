# Где находятся сегментации Георгия, Ника и Ярослава

Все приведённые пути отсчитываются от локального корня репозитория
`D:\Аспа\Kalmykov_PhD`. Медицинские массивы и STL исключены из Git.

Общий каталог готовых анатомических STL:

```text
Experiments/scorpion_20261022/derived/exploratory/universal_meshing_20261026/
```

- Георгий: `georg_segmented_STL_v1/`.
- Ник: `nik_segmented_STL_v1/`.
- Ярослав: `yaroslav_segmented_STL_v1/`.

В каждой папке находятся `body.stl`, `lung.stl`, `bone.stl`, `heart.stl`,
`fat.stl` и паспорт `segmented_STL.local.json`. STL сохраняют все компоненты
и внутренние интерфейсы исходной автоматической сегментации. Это результат
TotalSegmentator 2.18.0, ожидающий анатомического просмотра.

## Исходные маски NIfTI

Свежая маска тела и её два исходных класса находятся в `segmentation/`
соответствующего каталога STL: `body.nii.gz` и `raw_body/`.
Четыре собственные тканевые маски повторно использованы из проверенных
результатов TotalSegmentator:

- Георгий: `MATLAB_TRKG4_real_subjects/output/exploratory/scorpion_georg_fem_20261008/segmentation/`.
- Ник: `MATLAB_TRKG4_real_subjects/output/exploratory/lateral_fat_20260916/segmentation/`.
- Ярослав: `Experiments/scorpion_20261022/derived/exploratory/universal_meshing_20261026/yaroslav_ct_pipeline_v1/segmentation/`.

Там находятся `lung.nii.gz`, `bone.nii.gz`, `heart.nii.gz`, `fat.nii.gz`,
а в `raw_total/` и `raw_tissue_types/` — отдельные классы сегментатора.
Паспорта STL связывают каждый вход и выход контрольными суммами.

## Отдельные маски вычислительной области

Подготовка объёмной сетки использует самостоятельные производные маски
`*_within_domain.nii.gz`. Они находятся в том же общем каталоге,
в `georg_ct_pipeline_v2/`, `nik_ct_pipeline_v5/`, `yaroslav_ct_pipeline_v4/`.
Эти маски ограничены явно описанной вычислительной областью и не подменяют
исходные маски или перечисленные выше анатомические STL.
