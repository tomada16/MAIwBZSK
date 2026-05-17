# Klasyfikacja ryzyka zaburzeń snu

## Opis projektu

Celem projektu jest przewidywanie ryzyka zaburzeń snu na podstawie danych o stylu życia, zdrowiu i nawykach związanych ze snem.

## Zbiór danych

**Sleep Health and Daily Performance Dataset**  
Źródło: https://www.kaggle.com/datasets/mohankrishnathalla/sleep-health-and-daily-performance-dataset

## Problem rozpoznawania

Klasyfikacja wieloklasowa — na podstawie cech wejściowych przewidujemy jedną z 4 klas ryzyka zaburzeń snu (`sleep_disorder_risk`):

| Klasa | Opis |
|-------|------|
| Healthy | Brak zaburzeń snu |
| Mild | Łagodne ryzyko zaburzeń snu |
| Moderate | Umiarkowane ryzyko zaburzeń snu |
| Severe | Wysokie ryzyko zaburzeń snu |

## Cechy wejściowe (X)

- **Demograficzne:** wiek, płeć, zawód, BMI
- **Parametry snu:** czas snu, jakość snu, faza REM, głęboki sen, latencja snu, epizody wybudzenia
- **Styl życia:** liczba kroków, ćwiczenia, kofeina, alkohol, czas przed ekranem, drzemki
- **Zdrowie:** tętno spoczynkowe, poziom stresu, chronotyp, stan zdrowia psychicznego
- **Inne:** temperatura pokoju, praca zmianowa, typ dnia, sezon

## Eksperymenty

### Eksperyment 1 — Porównanie klasyfikatorów
Porównanie trzech klasyfikatorów: Gaussian Naive Bayes, K-Nearest Neighbors i Decision Tree przy użyciu `RepeatedStratifiedKFold`. Wyniki oceniane metryką `balanced_accuracy_score`. Analiza statystyczna testem Shapiro-Wilka i parowym testem t-Studenta.

### Eksperyment 2 — Wpływ SMOTE na klasyfikację
Zbadanie wpływu oversamplingu metodą SMOTE na jakość klasyfikacji w przypadku niezbalansowanych klas. Porównanie wyników klasyfikatorów bez i z zastosowaniem SMOTE.
