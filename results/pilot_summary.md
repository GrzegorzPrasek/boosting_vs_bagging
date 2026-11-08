# Pilotaż – wyniki

Wygenerowano: 2026-11-08 22:44. 2×5-fold CV (ziarna (42, 2026)), stałe hiperparametry: `{'B': 10, 'M': 200, 'max_depth': 3, 'learning_rate': 0.1, 'alpha': 1.0, 'beta': 1.0, 'k': 20, 'p': 2}`. Referencje Bagging/RF: `{'n_estimators': 100}`.

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

## Koszt: zbudowane drzewa / czas treningu [s] (średnia na fold)

| dataset       | GBDT       | Bagging    | RF         | Baseline    | Baseline+1   | Baseline+2   | Baseline+1+2   |
|:--------------|:-----------|:-----------|:-----------|:------------|:-------------|:-------------|:---------------|
| auto_mpg      | 200 / 0.05 | 100 / 0.07 | 100 / 0.05 | 2000 / 0.50 | 2000 / 0.51  | 1112 / 0.31  | 1070 / 0.30    |
| banknote      | 200 / 0.15 | 100 / 0.10 | 100 / 0.07 | 2000 / 1.42 | 2000 / 1.43  | 1888 / 1.46  | 1894 / 1.47    |
| breast_cancer | 200 / 0.29 | 100 / 0.17 | 100 / 0.06 | 2000 / 2.75 | 2000 / 2.77  | 1456 / 2.10  | 1566 / 2.26    |
| concrete      | 200 / 0.11 | 100 / 0.14 | 100 / 0.12 | 2000 / 1.03 | 2000 / 1.04  | 1990 / 1.11  | 1992 / 1.12    |
| wine          | 600 / 0.24 | 100 / 0.09 | 100 / 0.06 | 6000 / 2.13 | 6000 / 2.19  | 3852 / 1.69  | 3714 / 1.64    |
| yacht         | 200 / 0.04 | 100 / 0.06 | 100 / 0.04 | 2000 / 0.37 | 2000 / 0.38  | 1886 / 0.40  | 1906 / 0.41    |

## Oszczędność Ensemble-Aware ES

| dataset       | porównanie                 | drzewa   | czas   |
|:--------------|:---------------------------|:---------|:-------|
| wine          | Baseline+2 vs Baseline     | -36%     | -21%   |
| wine          | Baseline+1+2 vs Baseline+1 | -38%     | -25%   |
| breast_cancer | Baseline+2 vs Baseline     | -27%     | -24%   |
| breast_cancer | Baseline+1+2 vs Baseline+1 | -22%     | -19%   |
| banknote      | Baseline+2 vs Baseline     | -6%      | +2%    |
| banknote      | Baseline+1+2 vs Baseline+1 | -5%      | +3%    |
| yacht         | Baseline+2 vs Baseline     | -6%      | +8%    |
| yacht         | Baseline+1+2 vs Baseline+1 | -5%      | +8%    |
| auto_mpg      | Baseline+2 vs Baseline     | -44%     | -37%   |
| auto_mpg      | Baseline+1+2 vs Baseline+1 | -46%     | -41%   |
| concrete      | Baseline+2 vs Baseline     | -1%      | +8%    |
| concrete      | Baseline+1+2 vs Baseline+1 | -0%      | +7%    |

## Szacunek czasu pełnych eksperymentów

Model: czas ~ c_metoda · n·log n · d · K, c z pilotażu; d dla nowych zbiorów przybliżone. Wynik orientacyjny.

| dataset            |     n |   d≈ |   K |   1 przebieg CV, wszystkie metody [min] |
|:-------------------|------:|-----:|----:|----------------------------------------:|
| dry_bean           | 13611 |   16 |   7 |                                   310.4 |
| mushroom           |  8124 |  117 |   2 |                                   182.6 |
| spambase           |  4601 |   57 |   2 |                                    47   |
| car                |  1728 |   21 |   4 |                                    22.8 |
| wine_quality_white |  4898 |   11 |   1 |                                     9.7 |
| german_credit      |  1000 |   61 |   2 |                                     8.8 |
| ccpp               |  9568 |    4 |   1 |                                     7.5 |
| abalone            |  4177 |   10 |   1 |                                     7.4 |
| student_por        |   649 |   56 |   1 |                                     4.9 |
| rice               |  3810 |    7 |   2 |                                     4.7 |
| breast_cancer      |   569 |   30 |   2 |                                     2.3 |
| forest_fires       |   517 |   29 |   1 |                                     2   |
| ionosphere         |   351 |   34 |   2 |                                     1.5 |
| sonar              |   208 |   60 |   2 |                                     1.4 |
| glass              |   214 |    9 |   6 |                                     1.3 |
| concrete           |  1030 |    8 |   1 |                                     1.2 |
| airfoil            |  1503 |    5 |   1 |                                     1.2 |
| energy             |   768 |    8 |   1 |                                     0.9 |
| banknote           |  1372 |    4 |   2 |                                     0.8 |
| wine               |   178 |   13 |   3 |                                     0.7 |
| auto_mpg           |   398 |    9 |   1 |                                     0.4 |
| real_estate        |   414 |    6 |   1 |                                     0.3 |
| yacht              |   308 |    6 |   1 |                                     0.2 |
| computer_hardware  |   209 |    6 |   1 |                                     0.1 |

- **bez strojenia [h, 1 rdzeń]**: 10.3
- **ze strojeniem [h, 1 rdzeń]**: 56.6
- **ze strojeniem [h, 8 rdzeni]**: 7.1
