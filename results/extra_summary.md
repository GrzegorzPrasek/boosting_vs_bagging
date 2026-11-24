# Eksperymenty dodatkowe: szum w etykietach i ablacja Adaptive Sampling

Zbiory: 23 (bez Dry Bean). Szum 0% z `full.csv`, 10% i 20% z `extra.csv`. Przebiegów: 6210.

## Klasyfikacja (Macro F1)

### Średnie rangi 7 metod

|                             |   GBDT |   Bagging |   RF |   Baseline |   Baseline+1 |   Baseline+2 |   Baseline+1+2 |
|:----------------------------|-------:|----------:|-----:|-----------:|-------------:|-------------:|---------------:|
| szum 0% (Friedman p=0.042)  |   3.45 |      4.82 | 3.55 |       5    |         2.64 |         5.09 |           3.45 |
| szum 10% (Friedman p=0.039) |   5.18 |      4    | 3.09 |       3.27 |         3    |         5.36 |           4.09 |
| szum 20% (Friedman p=0.033) |   5.73 |      3.64 | 2.91 |       3.45 |         4    |         4.91 |           3.36 |

### Względna różnica vs Baseline [%] (+ = lepiej), średnia po zbiorach

|               |    0% |   10% |   20% |
|:--------------|------:|------:|------:|
| GBDT          |  0.44 | -0.85 | -2.08 |
| Bagging       |  0.91 |  0.53 |  3.06 |
| RF            |  1.69 |  1.29 |  3.69 |
| Baseline+1    |  0.94 | -0.07 | -0.75 |
| Baseline+2    | -0.27 | -1.93 | -2.1  |
| Baseline+1+2  |  0.5  | -1.07 | -0.57 |
| Baseline+1(e) |  0.92 | -0.07 | -0.23 |
| Baseline+1(d) |  0.18 |  0.2  |  0.75 |

### Testy Wilcoxona – szum 0%

| porównanie                | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:--------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1     | 10 / 1            |             0.94 | 0.002  |     0.0156 |
| Baseline → Baseline+2     | 4 / 5             |            -0.27 | 0.7344 |     1      |
| Baseline → Baseline+1+2   | 7 / 4             |             0.5  | 0.2061 |     1      |
| Baseline+1 → Baseline+1+2 | 3 / 6             |            -0.42 | 0.25   |     1      |
| Baseline+2 → Baseline+1+2 | 9 / 2             |             0.78 | 0.042  |     0.2939 |
| GBDT → Baseline+1         | 7 / 4             |             0.5  | 0.3203 |     1      |
| GBDT → Baseline+1+2       | 6 / 5             |             0.07 | 0.9658 |     1      |
| RF → Baseline+1+2         | 6 / 5             |            -0.94 | 0.7002 |     1      |

Ablacja (osobna rodzina testów, Holm):

| porównanie                    | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:------------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1(e)      | 9 / 2             |             0.92 | 0.0322 |     0.1611 |
| Baseline → Baseline+1(d)      | 7 / 4             |             0.18 | 0.5195 |     1      |
| Baseline+1(e) → Baseline+1    | 4 / 7             |             0.02 | 0.6377 |     1      |
| Baseline+1(d) → Baseline+1    | 9 / 2             |             0.77 | 0.1475 |     0.4424 |
| Baseline+1(e) → Baseline+1(d) | 2 / 9             |            -0.72 | 0.042  |     0.168  |

### Testy Wilcoxona – szum 10%

| porównanie                | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:--------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1     | 6 / 5             |            -0.07 | 0.8311 |     1      |
| Baseline → Baseline+2     | 2 / 9             |            -1.93 | 0.0244 |     0.1953 |
| Baseline → Baseline+1+2   | 5 / 6             |            -1.07 | 0.3203 |     1      |
| Baseline+1 → Baseline+1+2 | 3 / 8             |            -0.99 | 0.2783 |     1      |
| Baseline+2 → Baseline+1+2 | 8 / 3             |             0.9  | 0.0674 |     0.4717 |
| GBDT → Baseline+1         | 9 / 2             |             0.82 | 0.123  |     0.7383 |
| GBDT → Baseline+1+2       | 6 / 5             |            -0.21 | 0.9658 |     1      |
| RF → Baseline+1+2         | 5 / 6             |            -2.21 | 0.3203 |     1      |

Ablacja (osobna rodzina testów, Holm):

| porównanie                    | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:------------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1(e)      | 4 / 7             |            -0.07 | 0.4131 |          1 |
| Baseline → Baseline+1(d)      | 6 / 5             |             0.2  | 0.8311 |          1 |
| Baseline+1(e) → Baseline+1    | 8 / 3             |             0.01 | 0.4131 |          1 |
| Baseline+1(d) → Baseline+1    | 3 / 8             |            -0.26 | 0.2402 |          1 |
| Baseline+1(e) → Baseline+1(d) | 8 / 3             |             0.29 | 0.2402 |          1 |

### Testy Wilcoxona – szum 20%

| porównanie                | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:--------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1     | 4 / 7             |            -0.75 | 0.2783 |     1      |
| Baseline → Baseline+2     | 4 / 7             |            -2.1  | 0.1748 |     0.874  |
| Baseline → Baseline+1+2   | 6 / 5             |            -0.57 | 1      |     1      |
| Baseline+1 → Baseline+1+2 | 6 / 5             |             0.2  | 0.4648 |     1      |
| Baseline+2 → Baseline+1+2 | 9 / 2             |             1.62 | 0.0244 |     0.1953 |
| GBDT → Baseline+1         | 9 / 2             |             1.43 | 0.0322 |     0.2256 |
| GBDT → Baseline+1+2       | 9 / 2             |             1.65 | 0.0537 |     0.3223 |
| RF → Baseline+1+2         | 5 / 6             |            -3.59 | 0.4131 |     1      |

Ablacja (osobna rodzina testów, Holm):

| porównanie                    | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:------------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1(e)      | 5 / 6             |            -0.23 | 0.7646 |     1      |
| Baseline → Baseline+1(d)      | 7 / 4             |             0.75 | 0.1475 |     0.5898 |
| Baseline+1(e) → Baseline+1    | 4 / 7             |            -0.51 | 0.6377 |     1      |
| Baseline+1(d) → Baseline+1    | 2 / 9             |            -1.48 | 0.0137 |     0.0684 |
| Baseline+1(e) → Baseline+1(d) | 7 / 4             |             0.99 | 0.1475 |     0.5898 |

### Odporność: zmiana metryki przy 20% szumu względem 0% [%] (− = pogorszenie)

| method        |   zmiana [%] |
|:--------------|-------------:|
| GBDT          |        -9.11 |
| Bagging       |        -5.21 |
| RF            |        -5.4  |
| Baseline      |        -6.91 |
| Baseline+1    |        -8.36 |
| Baseline+2    |        -8.43 |
| Baseline+1+2  |        -7.75 |
| Baseline+1(e) |        -7.89 |
| Baseline+1(d) |        -6.43 |

## Regresja (MAE / std(y))

### Średnie rangi 7 metod

|                             |   GBDT |   Bagging |   RF |   Baseline |   Baseline+1 |   Baseline+2 |   Baseline+1+2 |
|:----------------------------|-------:|----------:|-----:|-----------:|-------------:|-------------:|---------------:|
| szum 0% (Friedman p=0.384)  |   4    |      3.83 | 2.75 |       4.08 |         4.08 |         4.83 |           4.42 |
| szum 10% (Friedman p=0.001) |   4.42 |      3.08 | 2.92 |       2.83 |         5.33 |         3.58 |           5.83 |
| szum 20% (Friedman p=0.000) |   4    |      3.33 | 2.33 |       2.92 |         5.17 |         4.33 |           5.92 |

### Względna różnica vs Baseline [%] (+ = lepiej), średnia po zbiorach

|               |    0% |   10% |   20% |
|:--------------|------:|------:|------:|
| GBDT          |  1.72 | -1.52 | -0.42 |
| Bagging       |  0.18 |  1.54 |  0.83 |
| RF            |  2.16 |  0.86 |  1.79 |
| Baseline+1    |  0.07 | -3.84 | -3.59 |
| Baseline+2    | -0.07 | -0.11 | -0.85 |
| Baseline+1+2  | -0.05 | -3.45 | -3.8  |
| Baseline+1(e) |  0.28 | -5.24 | -3.62 |
| Baseline+1(d) |  0.09 |  0.33 |  1.15 |

### Testy Wilcoxona – szum 0%

| porównanie                | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:--------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1     | 7 / 5             |             0.07 | 0.7334 |     1      |
| Baseline → Baseline+2     | 2 / 10            |            -0.07 | 0.1514 |     1      |
| Baseline → Baseline+1+2   | 8 / 4             |            -0.05 | 0.9697 |     1      |
| Baseline+1 → Baseline+1+2 | 4 / 8             |            -0.14 | 0.6772 |     1      |
| Baseline+2 → Baseline+1+2 | 7 / 5             |            -0.03 | 0.6221 |     1      |
| GBDT → Baseline+1         | 4 / 8             |            -1.8  | 0.2661 |     1      |
| GBDT → Baseline+1+2       | 4 / 8             |            -1.95 | 0.5693 |     1      |
| RF → Baseline+1+2         | 3 / 9             |            -4.06 | 0.0425 |     0.3398 |

Ablacja (osobna rodzina testów, Holm):

| porównanie                    | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:------------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1(e)      | 5 / 7             |             0.28 | 0.0923 |     0.4614 |
| Baseline → Baseline+1(d)      | 5 / 7             |             0.09 | 0.4238 |     1      |
| Baseline+1(e) → Baseline+1    | 8 / 4             |            -0.23 | 0.2334 |     0.9336 |
| Baseline+1(d) → Baseline+1    | 6 / 6             |            -0.02 | 0.5693 |     1      |
| Baseline+1(e) → Baseline+1(d) | 7 / 5             |            -0.24 | 0.4697 |     1      |

### Testy Wilcoxona – szum 10%

| porównanie                | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:--------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1     | 1 / 11            |            -3.84 | 0.0015 |     0.0088 |
| Baseline → Baseline+2     | 3 / 9             |            -0.11 | 0.0923 |     0.3691 |
| Baseline → Baseline+1+2   | 0 / 12            |            -3.45 | 0.0005 |     0.0039 |
| Baseline+1 → Baseline+1+2 | 5 / 7             |             0.27 | 0.9697 |     0.9697 |
| Baseline+2 → Baseline+1+2 | 0 / 12            |            -3.35 | 0.0005 |     0.0039 |
| GBDT → Baseline+1         | 4 / 8             |            -2.31 | 0.1514 |     0.4541 |
| GBDT → Baseline+1+2       | 4 / 8             |            -1.97 | 0.3394 |     0.6787 |
| RF → Baseline+1+2         | 2 / 10            |            -4.72 | 0.021  |     0.105  |

Ablacja (osobna rodzina testów, Holm):

| porównanie                    | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:------------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1(e)      | 1 / 11            |            -5.24 | 0.001  |     0.0039 |
| Baseline → Baseline+1(d)      | 6 / 6             |             0.33 | 0.791  |     0.791  |
| Baseline+1(e) → Baseline+1    | 9 / 3             |             1.26 | 0.0161 |     0.0322 |
| Baseline+1(d) → Baseline+1    | 1 / 11            |            -4.22 | 0.001  |     0.0039 |
| Baseline+1(e) → Baseline+1(d) | 12 / 0            |             5.03 | 0.0005 |     0.0024 |

### Testy Wilcoxona – szum 20%

| porównanie                | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:--------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1     | 1 / 11            |            -3.59 | 0.0015 |     0.0103 |
| Baseline → Baseline+2     | 2 / 10            |            -0.85 | 0.0522 |     0.209  |
| Baseline → Baseline+1+2   | 1 / 11            |            -3.8  | 0.0015 |     0.0103 |
| Baseline+1 → Baseline+1+2 | 4 / 8             |            -0.27 | 0.3804 |     0.3804 |
| Baseline+2 → Baseline+1+2 | 2 / 10            |            -2.94 | 0.0024 |     0.0122 |
| GBDT → Baseline+1         | 5 / 7             |            -3.25 | 0.1294 |     0.2588 |
| GBDT → Baseline+1+2       | 3 / 9             |            -3.45 | 0.064  |     0.209  |
| RF → Baseline+1+2         | 0 / 12            |            -5.82 | 0.0005 |     0.0039 |

Ablacja (osobna rodzina testów, Holm):

| porównanie                    | lepsze / gorsze   |   śr. zmiana [%] |      p |   p (Holm) |
|:------------------------------|:------------------|-----------------:|-------:|-----------:|
| Baseline → Baseline+1(e)      | 1 / 11            |            -3.62 | 0.001  |     0.0029 |
| Baseline → Baseline+1(d)      | 7 / 5             |             1.15 | 0.3804 |     0.7607 |
| Baseline+1(e) → Baseline+1    | 6 / 6             |             0.04 | 0.6221 |     0.7607 |
| Baseline+1(d) → Baseline+1    | 0 / 12            |            -4.89 | 0.0005 |     0.0024 |
| Baseline+1(e) → Baseline+1(d) | 12 / 0            |             4.53 | 0.0005 |     0.0024 |

### Odporność: zmiana metryki przy 20% szumu względem 0% [%] (− = pogorszenie)

| method        |   zmiana [%] |
|:--------------|-------------:|
| GBDT          |      -131.68 |
| Bagging       |      -105.53 |
| RF            |      -108.07 |
| Baseline      |      -121.81 |
| Baseline+1    |      -140.61 |
| Baseline+2    |      -122.17 |
| Baseline+1+2  |      -134.19 |
| Baseline+1(e) |      -140.52 |
| Baseline+1(d) |      -117.7  |

## Early Stopping przy szumie: zbudowane drzewa / Baseline (średnia geom.)

|                |   Baseline+2 |   Baseline+1+2 |
|:---------------|-------------:|---------------:|
| ('clf', '0%')  |        0.814 |          0.877 |
| ('clf', '10%') |        0.611 |          0.638 |
| ('clf', '20%') |        0.589 |          0.56  |
| ('reg', '0%')  |        0.813 |          0.784 |
| ('reg', '10%') |        0.813 |          0.738 |
| ('reg', '20%') |        0.773 |          0.698 |
