# Парный registration benchmark

Дата запуска: 2026-07-30. Код: `238e9ac4d44fae5ff25b16e564f19d789955b1cf9`.

## Постановка

- fixed: выдох; moving: вдох;
- поле: `fixed/exhale → moving/inhale` в локальных RAS+ миллиметрах;
- метод: ITK-elastix `rigid → affine → B-spline`, без registration-маски;
- параметры: 4 resolution, 4096 samples, 128/128/256 iterations, финальная сетка
  B-spline 24 мм, bending-energy weight 0.1, seed 20260730;
- маски применялись только для независимого Dice/surface QC;
- expert landmarks и image-derived keypoints считались раздельно;
- единица итоговой таблицы — субъект.

Gate: mean expert TRE ≤ 2 мм, expert TRE p95 ≤ 5 мм, lung Dice ≥ 0.90,
lung surface p95 ≤ 5 мм, Jacobian p01 ≥ 0.10, доля `Jacobian ≤ 0` равна нулю.

## DIR-Lab COPDgene: 10/10

| Случай | TRE до, мм | TRE после, мм | Dice лёгких | surface p95, мм | Jac p01 | Gate |
|---|---:|---:|---:|---:|---:|---|
| copd1 | 26.33 | 15.30 | 0.938 | 9.04 | 0.840 | FAIL |
| copd2 | 21.79 | 14.83 | 0.896 | 7.93 | 0.679 | FAIL |
| copd3 | 12.64 | 9.27 | 0.903 | 12.50 | 0.861 | FAIL |
| copd4 | 29.58 | 29.34 | 0.739 | 33.52 | 0.782 | FAIL |
| copd5 | 30.08 | 29.92 | 0.709 | 39.12 | 0.841 | FAIL |
| copd6 | 28.46 | 18.61 | 0.844 | 16.14 | 0.821 | FAIL |
| copd7 | 21.60 | 14.73 | 0.856 | 15.93 | 0.896 | FAIL |
| copd8 | 26.46 | 26.91 | 0.769 | 31.70 | 0.849 | FAIL |
| copd9 | 14.86 | 11.90 | 0.892 | 18.77 | 0.837 | FAIL |
| copd10 | 21.81 | 31.73 | 0.771 | 36.76 | 0.821 | FAIL |

Медиана mean TRE после регистрации 16.96 мм, IQR 14.76–28.73 мм, диапазон
9.27–31.73 мм. TRE улучшился у 8/10, но Gate 1 прошли 0/10. Folding отсутствует у
10/10. Следовательно, положительный якобиан подтверждает гладкость, но не точность
анатомического соответствия.

Локальный артефакт: `results/registration/dirlab/benchmark.csv`, SHA-256
`02582FC3470E4A57A3B9CE69F5BC0AB5D6CA018CF04B0D6B21726EC262657796`.

## Learn2Reg LungCT: только 3 случая с expert landmarks

| Случай | TRE до, мм | TRE после, мм | keypoint TRE после, мм | Dice лёгких | surface p95, мм | Gate |
|---|---:|---:|---:|---:|---:|---|
| LungCT_0001 | 20.18 | 4.52 | 7.89 | 0.898 | 24.50 | FAIL |
| LungCT_0002 | 15.60 | 13.16 | 15.22 | 0.931 | 10.21 | FAIL |
| LungCT_0003 | 8.14 | 3.65 | 5.19 | 0.917 | 20.54 | FAIL |

Медиана expert TRE 4.52 мм; Gate 1 прошли 0/3; folding отсутствует у 3/3.
Остальные LungCT не добавлялись в независимую landmark-оценку: их автоматически
полученные keypoints не являются заменой expert ground truth.

Локальный артефакт: `results/registration/lungct_expert/benchmark.csv`, SHA-256
`66FF91DB8F508AC04088876A2CF15AF6B0103F795C062185C7DB761759E17819`.

## Отвергнутые варианты на copd1

- Масочная инициализация по геометрическим центрам дала неверный глобальный сдвиг:
  TRE 68.99 мм, Dice 0.351. В batch не использовалась.
- Diffeomorphic Demons улучшил Dice, но оставил TRE около 18 мм.
- Остаточная lung-masked B-spline после глобального поля снизила TRE до 9.94 мм, но
  создала 2.36% неположительных якобианов. Композиция отвергнута.

Порог после просмотра этих вариантов не менялся.

## Следствие

Этот elastix-вариант сохраняется как воспроизводимый отрицательный baseline. Его нельзя
использовать для парного изменения формы лёгких, толщины тканей или генерации выдоха по
одной КТ. Следующий кандидат должен явно учитывать плевральное sliding/раздельные лёгкие
либо быть обучаемой respiratory registration-моделью, но оцениваться на тех же скрытых
expert landmarks без подгонки по ним.

## Команды

~~~powershell
.\.venv\Scripts\breathgeom.exe registration benchmark `
  --manifest data/interim/respiratory_pairs.local.csv `
  --dataset dirlab_copdgene --output results/registration/dirlab --force

.\.venv\Scripts\breathgeom.exe registration benchmark `
  --manifest data/interim/respiratory_pairs.local.csv `
  --dataset learn2reg_lungct --max-cases 3 `
  --output results/registration/lungct_expert --force
~~~

`results/` исключён из Git. JSON каждого субъекта содержит параметры, SHA-256 pair
manifest и версию кода. Поля провалившихся случаев не сохраняются.
