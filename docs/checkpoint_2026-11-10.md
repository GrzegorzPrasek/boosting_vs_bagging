# Adaptive Bagged Boosting – Progress Checkpoint

**Autorzy:** Mykhailo Shamrai, Grzegorz Prasek · **Prowadzący:** Stanisław Kaźmierczak · **Data:** 10.11.2026

## 1. Stan realizacji

| Etap z harmonogramu | Stan |
|---|---|
| Implementacja algorytmu bazowego, modyfikacji 1 i 2 | zrobione, z testami poprawności (pytest) |
| Przygotowanie danych | zrobione dla 6 zbiorów pilotażowych; pozostałe 18 do 14.11 |
| Wstępne eksperymenty (pilotaż) | zrobione: 6 zbiorów × 7 metod × 10 przebiegów CV |
| Strojenie hiperparametrów | do zrobienia (pilotaż na stałych parametrach) |

Zaimplementowane metody: pojedynczy GBDT, Bagging drzew, Random Forest oraz cztery warianty baggingu modeli GBDT (Baseline, +1 Adaptive Sampling, +2 Ensemble-Aware Early Stopping, +1+2). Procedura walidacji zgodna z planem: 2×5-fold CV (ziarna 42 i 2026), w każdym foldzie 20% części treningowej jako walidacja wewnętrzna, fold testowy wyłącznie do oceny końcowej.

## 2. Wyniki pilotażu (wstępne)

Parametry stałe: B=10, M=200, max_depth=3, learning_rate=0.1, α=β=1, k=20, p=2. Bagging i RF: 100 drzew.

**Metryka główna, średnia z 10 przebiegów** (pełne tabele z odchyleniami: `results/pilot_summary.md`)

| Zbiór | GBDT | Bagging | RF | Baseline | +1 | +2 | +1+2 |
|---|---|---|---|---|---|---|---|
| banknote (F1 ↑) | 0.995 | 0.990 | 0.992 | 0.993 | **0.996** | 0.993 | **0.996** |
| breast_cancer (F1 ↑) | 0.955 | 0.951 | 0.951 | 0.956 | 0.960 | 0.953 | **0.962** |
| wine (F1 ↑) | 0.936 | 0.961 | **0.980** | 0.959 | 0.970 | 0.959 | 0.973 |
| auto_mpg (MAE ↓) | 2.070 | **1.988** | 1.992 | 2.024 | 2.029 | 2.047 | 2.054 |
| concrete (MAE ↓) | **3.403** | 3.736 | 3.731 | 3.467 | 3.432 | 3.481 | 3.454 |
| yacht (MAE ↓) | **0.351** | 0.543 | 0.540 | 0.373 | 0.364 | 0.378 | 0.371 |
| **Śr. ranga – klasyfikacja** | 4.67 | 6.00 | 4.33 | 4.17 | 2.00 | 5.17 | **1.67** |
| **Śr. ranga – regresja** | 3.00 | 5.00 | 4.67 | 3.67 | **2.67** | 5.00 | 4.00 |

**Ensemble-Aware Early Stopping – zbudowane drzewa i czas treningu względem wariantu bez ES**

| Zbiór | wine | breast_cancer | auto_mpg | banknote | yacht | concrete |
|---|---|---|---|---|---|---|
| Drzewa (+2 vs Baseline) | −36% | −27% | −44% | −6% | −6% | −1% |
| Czas (+2 vs Baseline) | −21% | −24% | −37% | +2% | +8% | +8% |

**Obserwacje:**

- Adaptive Sampling (+1) poprawia Baseline na 5 z 6 zbiorów; w klasyfikacji wariant +1+2 ma najlepszą średnią rangę. Różnice mieszczą się jednak w odchyleniu standardowym – przy 6 zbiorach to obserwacja, nie wniosek.
- Na regresji pojedynczy GBDT jest konkurencyjny (najlepszy na concrete i yacht), a na auto_mpg wygrywają Bagging i RF.
- Early Stopping zmniejsza liczbę drzew (22–46%) i czas treningu (19–41%) bez utraty jakości tam, gdzie GBDT szybko się nasyca. Gdzie strata walidacyjna spada aż do M=200 (banknote, yacht, concrete), ES prawie nie przycina modeli, a koszt checkpointów to +2–8% czasu.

## 3. Doprecyzowania i odstępstwa od planu

1. **Kryterium stopu ES:** log-loss (klasyfikacja) i MAE (regresja) zespołu na walidacji wewnętrznej, zamiast Macro F1. F1 na małej walidacji (np. 29 obserwacji w Wine) zmienia się skokowo i powodował zbyt wczesne zatrzymanie. Poprawa = spadek straty o więcej niż 0,1%.
2. **Normalizacja e_i, d_i w Adaptive Sampling:** podział przez kwantyl 99% i obcięcie do 1, zamiast podziału przez maksimum – pojedyncza obserwacja odstająca nie spłaszcza wag pozostałych. Przy α=β=1 wagi mieszczą się w [1, 3]; efektywna liczność próbki to 92–96% n.
3. **Wspólne dane treningowe:** wszystkie metody trenują na tych samych 80% części treningowej foldu; walidację wewnętrzną wykorzystuje tylko ES (a w pełnych eksperymentach – strojenie).
4. **Przywracanie najlepszego checkpointu ES:** zamiast kopiować model przy każdej poprawie, zapamiętujemy liczbę iteracji i na końcu przycinamy model (test potwierdza identyczne predykcje). Pierwsza wersja z kopiowaniem wydłużała trening nawet o 71%.
5. **Koszt:** raportujemy zarówno drzewa w końcowym zespole, jak i drzewa faktycznie zbudowane (z ES także te za najlepszym checkpointem). W klasyfikacji wieloklasowej sklearn buduje K drzew na iterację.
6. **Dane:** Yacht Hydrodynamics nie jest dostępny przez `ucimlrepo` – pobieramy go bezpośrednio z UCI. Braki liczbowe uzupełniamy medianą z części treningowej foldu (bez wycieku z foldu testowego).

## 4. Problemy i ryzyka

| Ryzyko | Plan działania |
|---|---|
| ES nie przycina modeli przy M=200 (regresja) | Rozważyć wyższy limit M dla wariantów z ES (np. 500) – do decyzji |
| Czas pełnych eksperymentów: ok. 57 h na 1 rdzeniu ze strojeniem, ok. 7 h na 8 rdzeniach; Dry Bean i Mushroom to ok. 80% kosztu | Zrównoleglenie po foldach; w razie potrzeby zawężenie siatki dla dużych zbiorów lub `HistGradientBoosting` |
| Mała liczba zbiorów w pilotażu | Wnioski dopiero po 24 zbiorach i testach Friedmana/Wilcoxona |

## 5. Plan do 24.11

- **11–14.11** – loadery pozostałych 18 zbiorów, strojenie (random search, do 8 konfiguracji), uruchomienie pełnych eksperymentów
- **15–17.11** – tabele wyników, rankingi, test Friedmana i testy Wilcoxona, analiza kosztu
- **18–24.11** – raport końcowy, prezentacja, uporządkowanie kodu
