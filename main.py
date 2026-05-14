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
categorical_cols = data.select_dtypes(include='object').columns.tolist()
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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()

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
plt.show()