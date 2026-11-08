# Checkpoint 10.11 – etapy (minimum)

Termin: wtorek 10.11. Do dyspozycji: reszta niedzieli 8.11 + poniedziałek 9.11.

---

## Etap 1 – Setup (niedziela)
- [x] Struktura repo:
  ```
  src/
    data.py          # loadery zbiorów + preprocessing
    models.py        # BaggedGBDT (4 warianty)
    evaluation.py    # CV, metryki, zapis wyników
  scripts/
    run_pilot.py     # uruchomienie pilotażu
  results/           # CSV z wynikami
  requirements.txt
  ```
- [x] `requirements.txt`: numpy, pandas, scikit-learn, ucimlrepo, scipy, matplotlib
- [x] venv + instalacja zależności
- [x] Stałe hiperparametry pilotażu: B=10, M=200, max_depth=3, lr=0.1, subsample=1, α=β=1, k=20, p=2

**Gotowe gdy:** `import src` działa.

## Etap 2 – Dane (niedziela)
- [x] `load_dataset(name) -> (X, y, task)` dla 6 zbiorów pilotażowych przez `ucimlrepo`:
  - clf: Wine (109), Breast Cancer (17), Banknote (267)
  - reg: Yacht (243), Auto MPG (9), Concrete (165)
- [x] Preprocessing: mediana dla liczbowych, najczęstsza wartość dla kategorycznych, one-hot, label encoding y dla clf
- [x] Cache na dysku (`data/*.csv`)

**Gotowe gdy:** wszystkie 6 zbiorów się ładuje. NaN w cechach liczbowych (Auto MPG: 6) uzupełnia `impute_median` w obrębie foldu (bez wycieku). Yacht pobierany bezpośrednio z UCI (brak w `ucimlrepo`).

## Etap 3 – Baseline: Bagging + GBDT (niedziela)
- [x] Klasa `BaggedGBDT(task, B, M, max_depth, lr, adaptive_sampling=False, ensemble_es=False, alpha, beta, k, p, random_state)`
- [x] `fit(X_train, y_train, X_val, y_val)`, `predict`, `predict_proba`; atrybuty `n_trees_total_`, `fit_time_`
- [x] Bootstrap 100% ze zwracaniem, zapamiętanie indeksów (do OOB)
- [x] Agregacja: średnia predykcji (reg) / średnia `predict_proba` (clf)
- [x] `predict_proba` z wszystkimi klasami, nawet gdy jakiejś brak w próbce bootstrapowej

**Gotowe gdy:** baseline daje sensowny wynik na Wine i Yacht.

## Etap 4 – Pipeline CV i metryki (poniedziałek rano)
- [ ] 2×5-fold, ziarna 42 i 2026; `StratifiedKFold` dla clf, `KFold` dla reg
- [ ] W każdym foldzie: 20% train → walidacja wewnętrzna (stratyfikowana dla clf), ten sam podział dla wszystkich metod
- [ ] Metryki: Macro F1 + Accuracy / MAE + RMSE; czas treningu, czas predykcji, liczba drzew
- [ ] Referencje: `GradientBoosting*`, `Bagging*` (drzewa), `RandomForest*`
- [ ] Zapis wiersza (zbiór, metoda, seed, fold, metryki) do `results/pilot.csv`

**Gotowe gdy:** pipeline działa end-to-end dla baseline + 3 referencji na 1 zbiorze.

## Etap 5 – Modyfikacja 1: Adaptive Sampling (poniedziałek)
- [ ] Modele trenowane sekwencyjnie; model 1 zwykły bootstrap
- [ ] Przed modelem b: predykcje OOB modeli 1..b-1 (dla obs. i tylko modele, które jej nie widziały)
- [ ] `e_i`: błąd OOB zespołu (reg: |y − ŷ|; clf: 1 − p(prawdziwej klasy)), normalizacja do [0,1]
- [ ] `d_i`: rozbieżność modeli (reg: std predykcji; clf: średnia wariancja proba), normalizacja do [0,1]
- [ ] Obserwacje bez predykcji OOB: `e_i = d_i = 0`
- [ ] `w_i = 1 + α·e_i + β·d_i`, `P(i) = w_i / Σw`, losowanie N obserwacji ze zwracaniem

**Gotowe gdy:** wariant +1 działa, rozkład P(i) nie jest zdegenerowany (sprawdzić min/max).

## Etap 6 – Modyfikacja 2: Ensemble-Aware Early Stopping (poniedziałek)
- [ ] GBDT z `warm_start=True`, dokładanie po k drzew aż do M
- [ ] Na każdym checkpoincie ocena zespołu (gotowe modele + aktualny) na walidacji wewnętrznej (F1 / MAE)
- [ ] Przy poprawie `deepcopy` modelu jako najlepszy; stop po p checkpointach bez poprawy
- [ ] Do zespołu trafia najlepsza kopia; zapis liczby drzew każdego modelu
- [ ] Wariant +1+2: oba mechanizmy naraz

**Gotowe gdy:** wariant +2 kończy z mniej niż B·M drzewami na części zbiorów.

## Etap 7 – Pilotaż (poniedziałek wieczór)
- [ ] `scripts/run_pilot.py`: 6 zbiorów × 7 metod × 10 foldów
- [ ] Najpierw test na 1 zbiorze i 1 foldzie, potem pełne uruchomienie
- [ ] Tabela: średnia ± std metryki głównej (metoda × zbiór)
- [ ] Tabela: średnia liczba drzew i czas treningu (metoda × zbiór)
- [ ] Oszacowanie czasu pełnych eksperymentów (24 zbiory + strojenie 8 konfiguracji), zwłaszcza Dry Bean i CCPP

**Gotowe gdy:** obie tabele są w `results/` i jest szacunek czasu.

## Etap 8 – Materiał na checkpoint (wtorek rano)
- [ ] Co jest zrobione (4 warianty + referencje, pipeline, pilotaż)
- [ ] Pierwsze obserwacje (ile drzew oszczędza ES, czy AS zmienia wynik)
- [ ] Odstępstwa od planu (stałe hiperparametry w pilotażu, brak strojenia)
- [ ] Ryzyka: czas obliczeń na dużych zbiorach (plan B: `HistGradientBoosting*` / zawężenie siatki)
- [ ] Plan do 24.11:
  - 11–14.11: pozostałe 18 zbiorów, strojenie, pełne eksperymenty
  - 15–17.11: analiza, rankingi, Friedman + Wilcoxon
  - 18–24.11: raport, prezentacja, porządki w kodzie

---

### Czego NIE robimy przed wtorkiem
- strojenia hiperparametrów
- pozostałych 18 zbiorów
- testów statystycznych
- Learning Progress (poza zakresem)
