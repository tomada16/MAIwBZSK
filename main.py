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