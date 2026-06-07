# wine_fraud_using_SVM
A machine learning project that detects **fraudulent wine** based on physicochemical
properties using a Support Vector Machine with class imbalance handling.

---

## Dataset

- **File:** `wine_fraud.csv`
- **Size:** 6,497 samples (red + white wine)
- **Features:** 11 physicochemical properties + wine type

| Feature | Description |
|---------|-------------|
| `fixed acidity` | Fixed acidity level |
| `volatile acidity` | Volatile acidity level |
| `citric acid` | Citric acid content |
| `residual sugar` | Residual sugar content |
| `chlorides` | Chloride content |
| `free sulfur dioxide` | Free SO₂ level |
| `total sulfur dioxide` | Total SO₂ level |
| `density` | Wine density |
| `pH` | pH level |
| `sulphates` | Sulphate content |
| `alcohol` | Alcohol percentage |
| `type` | Red or White (encoded) |

- **Target:** `quality` → `Legit` (0) / `Fraud` (1)

---

## Class Imbalance
Legit:  6,251  (96.2%)
Fraud:    246   (3.8%)

- Red wine fraud rate:   **3.94%**
- White wine fraud rate: **3.74%**
- Fraud rate is similar across both wine types

> This severe imbalance requires special handling — addressed with `class_weight='balanced'`.

---

## Exploratory Data Analysis

- **Count plot** of Legit vs Fraud overall
- **Count plot** of fraud split by wine type (red/white)
- **Bar chart** of feature correlations with the `Fraud` target
- **Clustermap** (hierarchical clustering of correlation matrix using `viridis`)

---

## Preprocessing

- Mapped target: `Legit → 0`, `Fraud → 1`
- One-hot encoded `type` column with `drop_first=True` (red=0, white=1)
- Split: **90% training / 10% test** (`test_size=0.1, random_state=101`)
- Feature scaling with **StandardScaler** (fit on train only)

---

## Model — SVC with Class Balancing + GridSearchCV

```python
SVC(class_weight='balanced')
```

`class_weight='balanced'` automatically adjusts weights inversely proportional
to class frequency — critical for imbalanced datasets like this one.

### Hyperparameter Search

```python
param_grid = {'C': [0.001, 0.01, 0.1, 0.5, 1]}
GridSearchCV(svc, param_grid, cv=5)
```

### Best Parameter
C = 1

---

## Results (10% test set — 650 samples)

### Confusion Matrix

|  | Predicted Fraud | Predicted Legit |
|--|----------------|----------------|
| **Actual Fraud** | 17 ✅ | 10 ❌ |
| **Actual Legit** | 92 ❌ | 531 ✅ |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Fraud | 0.16 | 0.63 | 0.25 | 27 |
| Legit | 0.98 | 0.85 | 0.91 | 623 |
| **Accuracy** | | | **0.84** | 650 |

---

## Interpreting the Results

| Metric | Value | What it means |
|--------|-------|----------------|
| Fraud Recall | **0.63** | Caught 63% of actual fraud cases |
| Fraud Precision | 0.16 | Many false alarms (legit flagged as fraud) |
| Legit Recall | 0.85 | Correctly cleared 85% of legit wines |
| Overall Accuracy | 0.84 | High, but misleading due to imbalance |

> **For fraud detection, Recall matters more than Precision.**
> Missing real fraud is more costly than a false alarm.
> The model catches **63% of fraud cases** — a solid result given only 3.8% fraud rate.

---

##  Libraries Used

- `numpy`, `pandas` — data handling
- `matplotlib`, `seaborn` — visualization (countplot, clustermap, bar chart)
- `scikit-learn`:
  - `SVC` with `class_weight='balanced'`
  - `StandardScaler`
  - `GridSearchCV`
  - `train_test_split`
  - `confusion_matrix`, `classification_report`, `ConfusionMatrixDisplay`


## Key Takeaways

| Topic | Detail |
|-------|--------|
| Dataset size | 6,497 wines |
| Fraud rate | ~3.8% (severely imbalanced) |
| Best C | 1 |
| Fraud recall | **63%** |
| Overall accuracy | 84% |

> **Lesson 1:** Accuracy is a misleading metric for imbalanced datasets —
> always check Precision, Recall, and F1 per class.

> **Lesson 2:** `class_weight='balanced'` is essential when one class
> heavily dominates — it prevents the model from simply predicting "Legit" every time.
