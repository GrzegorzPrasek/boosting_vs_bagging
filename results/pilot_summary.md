# Pilotaż – wyniki

Wygenerowano: 2026-11-08 22:33. 2×5-fold CV (ziarna (42, 2026)), stałe hiperparametry: `{'B': 10, 'M': 200, 'max_depth': 3, 'learning_rate': 0.1, 'alpha': 1.0, 'beta': 1.0, 'k': 20, 'p': 2}`. Referencje Bagging/RF: `{'n_estimators': 100}`.

## Klasyfikacja – Macro F1 (↑), średnia ± std z 10 przebiegów

| dataset       | GBDT            | Bagging         | RF              | Baseline        | Baseline+1      | Baseline+2      | Baseline+1+2    |
|:--------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| banknote      | 0.9948 ± 0.0070 | 0.9897 ± 0.0042 | 0.9919 ± 0.0065 | 0.9930 ± 0.0066 | 0.9959 ± 0.0053 | 0.9926 ± 0.0072 | 0.9956 ± 0.0060 |
| breast_cancer | 0.9554 ± 0.0235 | 0.9508 ± 0.0168 | 0.9510 ± 0.0145 | 0.9564 ± 0.0207 | 0.9602 ± 0.0142 | 0.9527 ± 0.0196 | 0.9621 ± 0.0173 |
| wine          | 0.9360 ± 0.0483 | 0.9613 ± 0.0376 | 0.9804 ± 0.0190 | 0.9586 ± 0.0270 | 0.9700 ± 0.0268 | 0.9586 ± 0.0270 | 0.9730 ± 0.0283 |

Średnia pozycja w rankingu (1 = najlepsza):

|       |   GBDT |   Bagging |   RF |   Baseline |   Baseline+1 |   Baseline+2 |   Baseline+1+2 |
|:------|-------:|----------:|-----:|-----------:|-------------:|-------------:|---------------:|
| ranga |   4.67 |         6 | 4.33 |       4.17 |            2 |         5.17 |           1.67 |

## Regresja – MAE (↓), średnia ± std z 10 przebiegów

| dataset   | GBDT            | Bagging         | RF              | Baseline        | Baseline+1      | Baseline+2      | Baseline+1+2    |
|:----------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|
| auto_mpg  | 2.0700 ± 0.2198 | 1.9879 ± 0.2179 | 1.9917 ± 0.2119 | 2.0236 ± 0.2020 | 2.0293 ± 0.2162 | 2.0474 ± 0.1944 | 2.0538 ± 0.1989 |
| concrete  | 3.4028 ± 0.2777 | 3.7364 ± 0.2961 | 3.7308 ± 0.2951 | 3.4669 ± 0.2934 | 3.4320 ± 0.2743 | 3.4805 ± 0.2974 | 3.4538 ± 0.2931 |
| yacht     | 0.3513 ± 0.0717 | 0.5432 ± 0.1241 | 0.5402 ± 0.1276 | 0.3727 ± 0.0764 | 0.3644 ± 0.0679 | 0.3784 ± 0.0752 | 0.3708 ± 0.0695 |

Średnia pozycja w rankingu (1 = najlepsza):

|       |   GBDT |   Bagging |   RF |   Baseline |   Baseline+1 |   Baseline+2 |   Baseline+1+2 |
|:------|-------:|----------:|-----:|-----------:|-------------:|-------------:|---------------:|
| ranga |      3 |         5 | 4.67 |       3.67 |         2.67 |            5 |              4 |

## Koszt: zbudowane drzewa i czas treningu (średnia na fold)

| dataset       |   ('GBDT', 'drzewa (zbud.)') |   ('GBDT', 'fit [s]') |   ('Bagging', 'drzewa (zbud.)') |   ('Bagging', 'fit [s]') |   ('RF', 'drzewa (zbud.)') |   ('RF', 'fit [s]') |   ('Baseline', 'drzewa (zbud.)') |   ('Baseline', 'fit [s]') |   ('Baseline+1', 'drzewa (zbud.)') |   ('Baseline+1', 'fit [s]') |   ('Baseline+2', 'drzewa (zbud.)') |   ('Baseline+2', 'fit [s]') |   ('Baseline+1+2', 'drzewa (zbud.)') |   ('Baseline+1+2', 'fit [s]') |
|:--------------|-----------------------------:|----------------------:|--------------------------------:|-------------------------:|---------------------------:|--------------------:|---------------------------------:|--------------------------:|-----------------------------------:|----------------------------:|-----------------------------------:|----------------------------:|-------------------------------------:|------------------------------:|
| auto_mpg      |                          200 |                  0.06 |                             100 |                     0.08 |                        100 |                0.06 |                             2000 |                      0.57 |                               2000 |                        0.56 |                               1112 |                        0.4  |                                 1070 |                          0.38 |
| banknote      |                          200 |                  0.16 |                             100 |                     0.11 |                        100 |                0.07 |                             2000 |                      1.59 |                               2000 |                        1.61 |                               1888 |                        1.89 |                                 1894 |                          1.91 |
| breast_cancer |                          200 |                  0.29 |                             100 |                     0.17 |                        100 |                0.06 |                             2000 |                      2.73 |                               2000 |                        2.75 |                               1456 |                        2.19 |                                 1566 |                          2.4  |
| concrete      |                          200 |                  0.11 |                             100 |                     0.14 |                        100 |                0.12 |                             2000 |                      1.04 |                               2000 |                        1.04 |                               1990 |                        1.37 |                                 1992 |                          1.37 |
| wine          |                          600 |                  0.17 |                             100 |                     0.07 |                        100 |                0.04 |                             6000 |                      1.57 |                               6000 |                        1.59 |                               3852 |                        1.46 |                                 3714 |                          1.41 |
| yacht         |                          200 |                  0.06 |                             100 |                     0.09 |                        100 |                0.07 |                             2000 |                      0.56 |                               2000 |                        0.57 |                               1886 |                        0.96 |                                 1906 |                          0.97 |

## Oszczędność Ensemble-Aware ES

| dataset       | porównanie                 | drzewa   | czas   |
|:--------------|:---------------------------|:---------|:-------|
| wine          | Baseline+2 vs Baseline     | -36%     | -7%    |
| wine          | Baseline+1+2 vs Baseline+1 | -38%     | -11%   |
| breast_cancer | Baseline+2 vs Baseline     | -27%     | -20%   |
| breast_cancer | Baseline+1+2 vs Baseline+1 | -22%     | -13%   |
| banknote      | Baseline+2 vs Baseline     | -6%      | +19%   |
| banknote      | Baseline+1+2 vs Baseline+1 | -5%      | +19%   |
| yacht         | Baseline+2 vs Baseline     | -6%      | +71%   |
| yacht         | Baseline+1+2 vs Baseline+1 | -5%      | +71%   |
| auto_mpg      | Baseline+2 vs Baseline     | -44%     | -29%   |
| auto_mpg      | Baseline+1+2 vs Baseline+1 | -46%     | -33%   |
| concrete      | Baseline+2 vs Baseline     | -1%      | +32%   |
| concrete      | Baseline+1+2 vs Baseline+1 | -0%      | +31%   |

## Szacunek czasu pełnych eksperymentów

Model: czas ~ c_metoda · n·log n · d · K, c z pilotażu; d dla nowych zbiorów przybliżone. Wynik orientacyjny.

| dataset            |     n |   d≈ |   K |   1 przebieg CV, wszystkie metody [min] |
|:-------------------|------:|-----:|----:|----------------------------------------:|
| dry_bean           | 13611 |   16 |   7 |                                   343.7 |
| mushroom           |  8124 |  117 |   2 |                                   202.1 |
| spambase           |  4601 |   57 |   2 |                                    52.1 |
| car                |  1728 |   21 |   4 |                                    25.3 |
| wine_quality_white |  4898 |   11 |   1 |                                    10.8 |
| german_credit      |  1000 |   61 |   2 |                                     9.8 |
| ccpp               |  9568 |    4 |   1 |                                     8.3 |
| abalone            |  4177 |   10 |   1 |                                     8.2 |
| student_por        |   649 |   56 |   1 |                                     5.4 |
| rice               |  3810 |    7 |   2 |                                     5.2 |
| breast_cancer      |   569 |   30 |   2 |                                     2.5 |
| forest_fires       |   517 |   29 |   1 |                                     2.2 |
| ionosphere         |   351 |   34 |   2 |                                     1.6 |
| sonar              |   208 |   60 |   2 |                                     1.5 |
| glass              |   214 |    9 |   6 |                                     1.4 |
| concrete           |  1030 |    8 |   1 |                                     1.3 |
| airfoil            |  1503 |    5 |   1 |                                     1.3 |
| energy             |   768 |    8 |   1 |                                     0.9 |
| banknote           |  1372 |    4 |   2 |                                     0.9 |
| wine               |   178 |   13 |   3 |                                     0.8 |
| auto_mpg           |   398 |    9 |   1 |                                     0.5 |
| real_estate        |   414 |    6 |   1 |                                     0.3 |
| yacht              |   308 |    6 |   1 |                                     0.2 |
| computer_hardware  |   209 |    6 |   1 |                                     0.2 |

- **bez strojenia [h, 1 rdzeń]**: 11.4
- **ze strojeniem [h, 1 rdzeń]**: 62.7
- **ze strojeniem [h, 8 rdzeni]**: 7.8
