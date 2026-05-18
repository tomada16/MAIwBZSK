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

### Eksperyment 1: Wpływ stopniowego undersamplingu na jakość klasyfikacji

Eksperyment bada, jak sztuczne wyrównywanie dysproporcji klas za pomocą podpróbkowania losowego (*Random Undersampling*) wpływa na zdolności predykcyjne modeli. Modele oceniane są za pomocą metryki **Balanced Accuracy Score** w procedurze 2-krotnej powtórzonej walidacji krzyżowej (5 powtórzeń, łącznie 10 foldów).

Badanie podzielono na 4 etapy (kroki) stopniowego redukowania klas większościowych:
*   **Brak:** Oryginalny, niezbalansowany zbiór danych.
*   **Under k1:** Redukcja wyłącznie najliczniejszej klasy do poziomu liczności drugiej najliczniejszej klasy.
*   **Under k2:** Redukcja dwóch najliczniejszych klas do poziomu trzeciej klasy.
*   **Under k3 (Pełny balans):** Redukcja wszystkich klas do poziomu klasy najmniej licznej (`Severe`).

**Testowane klasyfikatory:**
*   Gaussian Naive Bayes (`GNB`)
*   K-Nearest Neighbors (`KNN`)
*   Decision Tree (`DT`)