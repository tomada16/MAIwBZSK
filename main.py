import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('TkAgg')

# ============================================================
# WCZYTANIE I PRZYGOTOWANIE DANYCH
# ============================================================

# Wczytanie danych
data = pd.read_csv('sleep_health_dataset.csv')

print("=== PODSTAWOWE INFORMACJE ===")
print(f"Rozmiar danych: {data.shape}")
print(f"\nPierwsze 3 wiersze:\n{data.head(3)}")
print(f"\nTypy danych:\n{data.dtypes}")
print(f"\nBrakujące wartości:\n{data.isnull().sum()}")

# Usunięcie kolumny ID
data = data.drop(columns=['person_id'])

# Kodowanie zmiennych kategorycznych
categorical_cols = data.select_dtypes(include='str').columns.tolist()
print(f"\n=== KOLUMNY KATEGORYCZNE ===")
print(categorical_cols)

le = LabelEncoder()
for col in categorical_cols:
    if col != 'sleep_disorder_risk':  # etykieta zostanie zakodowana osobno
        print(f"{col}: {data[col].unique()}")
        data[col] = le.fit_transform(data[col])

# Zakodowanie etykiety
print(f"\nKlasy sleep_disorder_risk: {data['sleep_disorder_risk'].unique()}")
data['sleep_disorder_risk'] = le.fit_transform(data['sleep_disorder_risk'])
class_names = le.classes_

# Podział na X i y
X = data.drop(columns=['sleep_disorder_risk']).values
y = data['sleep_disorder_risk'].values

print(f"\n=== WYNIK ===")
print(f"Rozmiar X: {X.shape}")
print(f"Rozmiar y: {y.shape}")
print(f"\nKlasy i liczności:")
unique, counts = np.unique(y, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Klasa {u}: {c} próbek")

# ============================================================
# WIZUALIZACJA ZBIORU DANYCH
# ============================================================

colors = ['steelblue', 'tomato', 'mediumseagreen', 'mediumpurple']

# --- Wykres 1: Rozkład klas ---
unique_names, counts = np.unique(data['sleep_disorder_risk'], return_counts=True)
x_pos = np.arange(len(unique_names))

plt.figure(figsize=(7, 4))
bars = plt.bar(x_pos, counts, color=colors, edgecolor='black', width=0.5)
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
             str(count), ha='center', va='bottom', fontsize=11)
plt.xticks(x_pos, class_names)
plt.title('Rozkład klas: sleep_disorder_risk')
plt.xlabel('Klasa')
plt.ylabel('Liczność')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('wykres1_rozkład_klas.png', dpi=120)

# --- Wykres 2: Histogramy wybranych cech numerycznych ---
num_cols = [
    'age', 'sleep_duration_hrs', 'sleep_quality_score',
    'stress_score', 'steps_that_day', 'heart_rate_resting_bpm'
]

fig, axes = plt.subplots(2, 3, figsize=(14, 7))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    axes[i].hist(data[col], bins=20, color='steelblue', edgecolor='black', alpha=0.8)
    axes[i].set_title(col)
    axes[i].set_ylabel('Liczność')
    axes[i].grid(axis='y', alpha=0.3)

plt.suptitle('Rozkłady wybranych cech numerycznych', fontsize=13)
plt.tight_layout()
plt.savefig('wykres2_histogramy.png', dpi=120)

# --- Wykres 3: Scatter — czas snu vs jakość snu ---
plt.figure(figsize=(7, 5))
for cls_idx, cls_name in enumerate(class_names):
    mask = y == cls_idx
    plt.scatter(
        data['sleep_duration_hrs'][mask],
        data['sleep_quality_score'][mask],
        s=15, alpha=0.6, label=cls_name, color=colors[cls_idx]
    )
plt.xlabel('sleep_duration_hrs')
plt.ylabel('sleep_quality_score')
plt.title('Czas snu vs Jakość snu (wg klasy)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('wykres3_sen_jakosc.png', dpi=120)

# --- Wykres 4: Scatter — stres vs tętno ---
plt.figure(figsize=(7, 5))
for cls_idx, cls_name in enumerate(class_names):
    mask = y == cls_idx
    plt.scatter(
        data['stress_score'][mask],
        data['heart_rate_resting_bpm'][mask],
        s=15, alpha=0.6, label=cls_name, color=colors[cls_idx]
    )
plt.xlabel('stress_score')
plt.ylabel('heart_rate_resting_bpm')
plt.title('Stres vs Tętno spoczynkowe (wg klasy)')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('wykres4_stres_tetno.png', dpi=120)

# --- Wykres 5: Macierz korelacji ---
top_cols = [
    'sleep_duration_hrs', 'sleep_quality_score', 'stress_score',
    'heart_rate_resting_bpm', 'steps_that_day', 'age',
    'rem_percentage', 'deep_sleep_percentage'
]

corr = data[top_cols].corr()

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(len(top_cols)))
ax.set_yticks(range(len(top_cols)))
ax.set_xticklabels(top_cols, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(top_cols, fontsize=9)

for i in range(len(top_cols)):
    for j in range(len(top_cols)):
        ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                ha='center', va='center', fontsize=8,
                color='white' if abs(corr.iloc[i, j]) > 0.5 else 'black')

ax.set_title('Macierz korelacji wybranych cech')
plt.tight_layout()
plt.savefig('wykres5_korelacja.png', dpi=120)

# ============================================================
# EKSPERYMENT 1
# Wpływ stopniowego undersamplingu na klasyfikację
#
# Kroki balansowania (dynamicznie wg liczności klas):
#   Brak undersamp. — oryginalne liczności
#   Under k1     — najliczniejsza → poziom 2. najliczniejszej
#   Under k2     — 2 najliczniejsze → poziom 3. najliczniejszej
#   Under k3     — wszystkie → poziom najmniej licznej
# ============================================================
from imblearn.under_sampling import RandomUnderSampler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from scipy.stats import shapiro, ttest_rel

print("\n" + "="*60)
print("EKSPERYMENT 1: Stopniowy undersampling")
print("="*60)

unique_exp, counts_exp = np.unique(y, return_counts=True)
sorted_idx = np.argsort(counts_exp)[::-1]
sorted_classes = unique_exp[sorted_idx]
sorted_counts = counts_exp[sorted_idx]

print(f"\nKlasy posortowane wg liczności (malejąco):")
for cls, cnt in zip(sorted_classes, sorted_counts):
    print(f"  Klasa {cls} ({class_names[cls]}): {cnt} próbek")

# Definicja kroków undersamplingu jako słowniki {klasa: docelowa_liczność}
steps = {"Brak": dict(zip(sorted_classes, sorted_counts))}

for krok in range(1, 4):
    target = sorted_counts[krok]
    steps[f"Under k{krok}"] = {
        cls: min(cnt, target) for cls, cnt in zip(sorted_classes, sorted_counts)
    }

print(f"\nZdefiniowane kroki undersamplingu:")
for step_name, strategy in steps.items():
    total = sum(strategy.values())
    print(f"  {step_name}: łącznie {total} próbek → {strategy}")

clfs = {
    'GNB': GaussianNB(),
    'KNN': KNeighborsClassifier(),
    'DT':  DecisionTreeClassifier()
}

n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

print(f"\nKlasyfikatory: {list(clfs.keys())}")
print(f"Walidacja: RepeatedStratifiedKFold(n_splits={n_splits}, n_repeats={n_repeats})")

# results[krok][clf] = lista 10 wyników BAC
results = {step: {clf: [] for clf in clfs} for step in steps}

print(f"\n=== WYNIKI ===")
for step_name, sampling_strategy in steps.items():
    if step_name == "Brak":
        X_s, y_s = X, y
    else:
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
        X_s, y_s = rus.fit_resample(X, y)

    u_s, c_s = np.unique(y_s, return_counts=True)
    print(f"\n{step_name} (rozmiar zbioru: {len(y_s)}):")
    for u, c in zip(u_s, c_s):
        print(f"  Klasa {u} ({class_names[u]}): {c} próbek")

    for clf_name, clf in clfs.items():
        scores = []
        for train_idx, test_idx in rskf.split(X_s, y_s):
            X_train, X_test = X_s[train_idx], X_s[test_idx]
            y_train, y_test = y_s[train_idx], y_s[test_idx]
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            scores.append(balanced_accuracy_score(y_test, y_pred))
        results[step_name][clf_name] = scores
        print(f"  {clf_name}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}")

# --- Analiza statystyczna: DT i KNN — Brak vs każdy krok ---
print(f"\n=== ANALIZA STATYSTYCZNA ===")
alpha = 0.05
other_steps = [s for s in steps.keys() if s != "Brak"]

for clf_name in ['DT', 'KNN']:
    print(f"\n--- {clf_name}: Brak vs pozostałe kroki ---")
    sc_brak = np.array(results["Brak"][clf_name])
    _, p_norm = shapiro(sc_brak)
    print(f"Shapiro (Brak): p={p_norm:.3f} → rozkład normalny: {p_norm > alpha}")

    for step_name in other_steps:
        sc_step = np.array(results[step_name][clf_name])
        _, p_norm2 = shapiro(sc_step)
        t_stat, p_val = ttest_rel(sc_brak, sc_step)
        mean_brak = np.mean(sc_brak)
        mean_step = np.mean(sc_step)
        print(f"\n  Brak ({mean_brak:.3f}) vs {step_name} ({mean_step:.3f}):")
        print(f"    Shapiro ({step_name}): p={p_norm2:.3f} | rozkład normalny: {p_norm2 > alpha}")
        print(f"    t-test: t={t_stat:.3f}, p={p_val:.3f}")
        if p_val < alpha:
            winner = "Brak" if mean_brak > mean_step else step_name
            print(f"    Różnica istotna statystycznie (p < {alpha}). Lepszy wariant: {winner}")
        else:
            print(f"    Brak istotnej różnicy statystycznej (p >= {alpha}).")

# --- Wykres eksperymentu 1 ---
step_labels = list(steps.keys())
colors_clf = ['steelblue', 'tomato', 'mediumseagreen']
x_bar = np.arange(len(step_labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
for i, clf_name in enumerate(clfs):
    means = [np.mean(results[s][clf_name]) for s in step_labels]
    stds = [np.std(results[s][clf_name]) for s in step_labels]
    bars = ax.bar(x_bar + i * width, means, width, yerr=stds, label=clf_name,
                  color=colors_clf[i], edgecolor='black', capsize=4, alpha=0.85)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(x_bar + width)
ax.set_xticklabels(step_labels)
ax.set_ylabel('Balanced Accuracy Score')
ax.set_title('Eksperyment 1: Wpływ stopniowego undersamplingu na klasyfikację')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('exp1_undersampling.png', dpi=120)

# --- Wykres punktowy rozkładu danych dla każdego kroku undersamplingu ---
# Cechy o największej korelacji: sleep_duration_hrs vs sleep_quality_score
col_names = data.drop(columns=['sleep_disorder_risk']).columns.tolist()
idx_dur = col_names.index('sleep_duration_hrs')
idx_qual = col_names.index('sleep_quality_score')

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax_i, (step_name, sampling_strategy) in enumerate(steps.items()):
    if step_name == "Brak":
        X_plot, y_plot = X, y
    else:
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_plot, y_plot = rus.fit_resample(X, y)

    for cls_idx, cls_name in enumerate(class_names):
        mask = y_plot == cls_idx
        axes[ax_i].scatter(
            X_plot[mask, idx_dur], X_plot[mask, idx_qual],
            s=10, alpha=0.5, label=cls_name, color=colors[cls_idx]
        )

    axes[ax_i].set_title(f"{step_name}\n(n={len(y_plot)})")
    axes[ax_i].set_xlabel('sleep_duration_hrs')
    axes[ax_i].set_ylabel('sleep_quality_score')
    axes[ax_i].grid(alpha=0.3)

axes[0].legend(fontsize=8)
plt.suptitle('Rozkład próbek w poszczególnych krokach undersamplingu', fontsize=13)
plt.tight_layout()
plt.savefig('exp1_rozklad_scatter.png', dpi=120)

# ============================================================
# EKSPERYMENT 2
# Wpływ cech na klasyfikację: ablacja i selekcja wprzód
#
# Faza A — Drop-one: usuń po jednej cesze, porównaj z bazą (wszystkie cechy)
# Faza B — Forward selection: zacznij od pustego zbioru, dodawaj najlepszą cechę
# Klasyfikator: DT (jako reprezentatywny, wrażliwy na cechy)
# Miara: balanced_accuracy_score, RepeatedStratifiedKFold(2,5)
# ============================================================

print("\n" + "="*60)
print("EKSPERYMENT 2: Wpływ cech na klasyfikację")
print("="*60)

col_names_exp2 = data.drop(columns=['sleep_disorder_risk']).columns.tolist()
print(f"\nLista wszystkich cech ({len(col_names_exp2)}):")
for i, c in enumerate(col_names_exp2):
    print(f"  [{i}] {c}")

# Pomocnicza funkcja: ewaluacja klasyfikatora na podzbiorze cech
def evaluate_subset(X_full, y, feature_indices, clf, cv):
    """Zwraca listę wyników BAC dla podanego podzbioru cech."""
    X_sub = X_full[:, feature_indices]
    scores = []
    for train_idx, test_idx in cv.split(X_sub, y):
        clf.fit(X_sub[train_idx], y[train_idx])
        y_pred = clf.predict(X_sub[test_idx])
        scores.append(balanced_accuracy_score(y[test_idx], y_pred))
    return scores

rskf_exp2 = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=42)
dt_exp2 = DecisionTreeClassifier(random_state=42)
all_indices = list(range(len(col_names_exp2)))

# ── FAZA A: Drop-one ablation ────────────────────────────────

print("\n=== FAZA A: Drop-one ablation (klasyfikator: DT) ===")

# Wyniki bazowe (wszystkie cechy)
baseline_scores = evaluate_subset(X, y, all_indices, dt_exp2, rskf_exp2)
baseline_mean = np.mean(baseline_scores)
print(f"\nBaseline (wszystkie {len(all_indices)} cechy): mean={baseline_mean:.3f}, std={np.std(baseline_scores):.3f}")

ablation_results = {}   # {nazwa_cechy: lista_wyników}
ablation_deltas  = {}   # {nazwa_cechy: delta_vs_baseline}

print(f"\n{'Cecha':<35} {'Mean BAC':>9} {'Std':>7} {'Δ vs baseline':>14} {'Wpływ'}")
print("-" * 75)

for drop_i, col in enumerate(col_names_exp2):
    remaining = [i for i in all_indices if i != drop_i]
    scores = evaluate_subset(X, y, remaining, DecisionTreeClassifier(random_state=42), rskf_exp2)
    ablation_results[col] = scores
    delta = np.mean(scores) - baseline_mean
    ablation_deltas[col] = delta

    if delta > 0.005:
        impact = "✓ POPRAWA (usunięcie pomaga)"
    elif delta < -0.005:
        impact = "✗ POGORSZENIE (cecha ważna)"
    else:
        impact = "≈ neutralna"

    print(f"  bez {col:<30} {np.mean(scores):>9.3f} {np.std(scores):>7.3f} {delta:>+14.3f}  {impact}")

# Posortuj cechy wg wpływu na dokładność (rosnąco = usunięcie najbardziej szkodzi)
sorted_by_delta = sorted(ablation_deltas.items(), key=lambda x: x[1])
most_important = sorted_by_delta[0][0]   # usunięcie powoduje największy spadek
least_important = sorted_by_delta[-1][0] # usunięcie powoduje największy wzrost (lub neutralne)

print(f"\nNajważniejsza cecha (jej brak najbardziej szkodzi): {most_important} (Δ={ablation_deltas[most_important]:+.3f})")
print(f"Najsłabsza cecha   (jej brak najbardziej pomaga):  {least_important} (Δ={ablation_deltas[least_important]:+.3f})")

# Test statystyczny: baseline vs bez_najważniejszej_cechy
print(f"\n=== ANALIZA STATYSTYCZNA: baseline vs bez '{most_important}' ===")
sc_a = np.array(baseline_scores)
sc_b = np.array(ablation_results[most_important])

_, p_a = shapiro(sc_a)
_, p_b = shapiro(sc_b)
print(f"Shapiro-Wilk — baseline: p={p_a:.3f}, bez cechy: p={p_b:.3f}")

t_stat2, p_val2 = ttest_rel(sc_a, sc_b)
print(f"Sparowany t-test: t={t_stat2:.3f}, p={p_val2:.3f}")
if p_val2 < 0.05:
    winner2 = "baseline" if np.mean(sc_a) > np.mean(sc_b) else f"bez {most_important}"
    print(f"  Różnica istotna statystycznie (p < 0.05). Lepszy wariant: {winner2}")
else:
    print(f"  Brak istotnej różnicy statystycznej (p ≥ 0.05).")

# ── FAZA B: Forward selection ────────────────────────────────

print("\n=== FAZA B: Forward selection (klasyfikator: DT) ===")

selected_indices = []
remaining_indices = list(all_indices)
forward_history = []   # lista kroków: (nazwa_cechy, mean_bac, std_bac, lista_wyników)

print(f"\n{'Krok':<6} {'Dodana cecha':<35} {'Mean BAC':>9} {'Std':>7} {'Δ vs poprzedni':>15}")
print("-" * 75)

prev_mean = 0.0
for step in range(len(col_names_exp2)):
    best_score_mean = -1
    best_idx = None
    best_scores = None

    for candidate in remaining_indices:
        trial_indices = selected_indices + [candidate]
        scores = evaluate_subset(X, y, trial_indices, DecisionTreeClassifier(random_state=42), rskf_exp2)
        if np.mean(scores) > best_score_mean:
            best_score_mean = np.mean(scores)
            best_idx = candidate
            best_scores = scores

    selected_indices.append(best_idx)
    remaining_indices.remove(best_idx)

    delta_fwd = best_score_mean - prev_mean
    forward_history.append((col_names_exp2[best_idx], best_score_mean, np.std(best_scores), best_scores))
    print(f"  {step+1:<4} {col_names_exp2[best_idx]:<35} {best_score_mean:>9.3f} {np.std(best_scores):>7.3f} {delta_fwd:>+15.3f}")
    prev_mean = best_score_mean

# Optymalny podzbiór: maksymalne BAC w forward selection
best_step = int(np.argmax([h[1] for h in forward_history]))
print(f"\nOptymalny zbiór cech: {best_step+1} cech (step {best_step+1}), "
      f"mean BAC = {forward_history[best_step][1]:.3f}")
print("  Cechy:", [forward_history[i][0] for i in range(best_step+1)])

# ── Wykresy Eksperymentu 2 ────────────────────────────────────

# Wykres A: Drop-one — delta BAC (sortowany)
fig, ax = plt.subplots(figsize=(12, 5))
sorted_cols = [x[0] for x in sorted_by_delta]
sorted_deltas = [x[1] for x in sorted_by_delta]
bar_colors = ['tomato' if d < -0.005 else 'mediumseagreen' if d > 0.005 else 'steelblue'
              for d in sorted_deltas]
bars = ax.barh(sorted_cols, sorted_deltas, color=bar_colors, edgecolor='black', alpha=0.85)
ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
for bar, val in zip(bars, sorted_deltas):
    ax.text(val + (0.001 if val >= 0 else -0.001), bar.get_y() + bar.get_height() / 2,
            f'{val:+.3f}', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
ax.set_xlabel('Δ BAC (względem baseline ze wszystkimi cechami)')
ax.set_title('Eksperyment 2A: Zmiana dokładności po usunięciu jednej cechy (DT)\n'
             '  Czerwony = cecha ważna (jej brak szkodzi) | Zielony = cecha zbędna/szkodliwa')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('exp2a_drop_one.png', dpi=120)

# Wykres B: Forward selection — krzywa BAC
fwd_means = [h[1] for h in forward_history]
fwd_stds = [h[2] for h in forward_history]
fwd_labels = [f"{i+1}. {h[0]}" for i, h in enumerate(forward_history)]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(range(1, len(fwd_means)+1), fwd_means, marker='o', color='steelblue',
        linewidth=2, markersize=6, label='Mean BAC')
ax.fill_between(range(1, len(fwd_means)+1),
                np.array(fwd_means) - np.array(fwd_stds),
                np.array(fwd_means) + np.array(fwd_stds),
                alpha=0.2, color='steelblue', label='±1 std')
ax.axhline(baseline_mean, color='tomato', linestyle='--', linewidth=1.5,
           label=f'Baseline (wszystkie cechy): {baseline_mean:.3f}')
ax.axvline(best_step+1, color='mediumseagreen', linestyle=':', linewidth=1.5,
           label=f'Optimum: {best_step+1} cech')
ax.set_xticks(range(1, len(fwd_means)+1))
ax.set_xticklabels([h[0] for h in forward_history], rotation=45, ha='right', fontsize=8)
ax.set_xlabel('Kolejno dodawane cechy (forward selection)')
ax.set_ylabel('Mean Balanced Accuracy Score')
ax.set_title('Eksperyment 2B: Forward selection — wzrost dokładności wraz z dodawaniem cech (DT)')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('exp2b_forward_selection.png', dpi=120)

print("\nWykresy zapisane: exp2a_drop_one.png, exp2b_forward_selection.png")