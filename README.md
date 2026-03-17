# Gradient Boosting — XGBoost, LightGBM & CatBoost on MNIST

## Project Summary

Built and compared three gradient boosting algorithms — XGBoost, LightGBM, and CatBoost — on the MNIST handwritten digit dataset. This project introduces the most widely used family of algorithms in production ML and Kaggle competitions.

**Dataset:** [ylecun/mnist](https://huggingface.co/datasets/ylecun/mnist)

**Business Problem:** Can we automatically recognize handwritten digits from images — powering applications like cheque processing, postal code recognition, and form digitization?

---

## What I Built

Three gradient boosting classifiers predicting digits 0-9 from 28×28 pixel images.

---

## Dataset Overview

```
Training images: 60,000
Test images:     10,000
Image size:      28 × 28 pixels = 784 features per image
Classes:         10 digits (0-9)
Class balance:   ~5,800-6,700 examples per digit — balanced ✅
```

Each image is a grayscale handwritten digit:
```
Pixel values: 0 (black) → 255 (white)
Normalized:   0.0 → 1.0 before training
```

---

## Key Learnings

### 1. What is Gradient Boosting?

Random Forest builds trees **in parallel** — each tree is independent.

Gradient Boosting builds trees **in sequence** — each tree learns from the mistakes of the previous one.

```
Tree 1 → makes predictions → has errors
Tree 2 → focuses on errors of Tree 1 → reduces them
Tree 3 → focuses on errors of Tree 2 → reduces them further
...
Tree 100 → final prediction = weighted sum of all trees
```

The "gradient" refers to mathematically following the direction that reduces error the fastest — exactly like gradient descent in neural networks, just applied to trees.

**Analogy:** Like a student who takes a test, reviews wrong answers, retakes the test, reviews again — each iteration getting better specifically on what they got wrong before.

---

### 2. XGBoost — Extreme Gradient Boosting

**How it works:**
XGBoost grows trees **level-wise** — completing each depth level before going deeper:

```
Level 1:        [root]
               /      \
Level 2:    [node]  [node]
            /  \    /  \
Level 3: [n]  [n]  [n]  [n]
```

Balanced, stable growth — every branch at the same depth before splitting further.

**Key parameters:**
```python
xgb.XGBClassifier(
    n_estimators=100,    # number of trees
    max_depth=6,         # how deep each tree grows
    learning_rate=0.1,   # contribution of each tree
    eval_metric='mlogloss'  # internal error metric for multiclass
)
```

**Learning rate explained:**
- High (0.3) → learns fast but might overshoot optimal solution
- Low (0.01) → learns slowly but more precisely, needs more trees
- Rule of thumb → lower learning rate + more trees = better but slower

**When to use XGBoost:**
- ✅ Small to medium datasets (< 100,000 rows)
- ✅ When stable, conservative results matter
- ✅ When interpretability and feature importance matter
- ✅ General purpose tabular data problems
- ❌ Very large datasets — LightGBM is faster
- ❌ Datasets with many categorical features — CatBoost handles these better
- ❌ Image or text data — deep learning models are better

**MNIST Result:**
```
10 trees  (depth=3) → 83.9%
100 trees (depth=6) → 97.05% ← final model
```
13% improvement from adding more trees and depth.

---

### 3. LightGBM — Light Gradient Boosting Machine

**How it works:**
LightGBM grows trees **leaf-wise** — always splitting the leaf that gives the highest error reduction:

```
         [root]
        /
     [node]
    /
 [node]
/
[leaf] ← keeps splitting the most promising branch
```

Instead of growing the whole tree evenly it focuses on the most important branch first — faster and often more accurate.

**Key parameters:**
```python
lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    verbose=-1           # suppress training output
)
```

**When to use LightGBM:**
- ✅ Large datasets (> 100,000 rows) — significantly faster than XGBoost
- ✅ Memory constrained environments
- ✅ Production systems requiring low latency
- ✅ When speed matters more than interpretability
- ✅ Numeric feature-heavy datasets
- ❌ Very small datasets (< 10,000 rows) — can overfit due to leaf-wise growth
- ❌ Datasets dominated by categorical features — CatBoost handles these better
- ❌ Image or text data — deep learning models are better

**MNIST Result:**
```
Accuracy: 97.68% — highest among all three models
```
Faster training than XGBoost and slightly better accuracy on this large dataset.

---

### 4. CatBoost — Categorical Boosting

**How it works:**
CatBoost uses a technique called **ordered boosting** — it processes training examples in random order and builds each tree using only the examples seen so far. This prevents overfitting on categorical features.

Its main innovation is handling categorical features **natively** without any preprocessing:

```python
# XGBoost/LightGBM — manual encoding required
df['city_enc'] = LabelEncoder().fit_transform(df['city'])

# CatBoost — pass categorical columns directly
model.fit(X_train, y_train, cat_features=['city', 'logistics_company'])
```

**Key parameters:**
```python
CatBoostClassifier(
    iterations=100,      # same as n_estimators
    depth=6,
    learning_rate=0.1,
    random_seed=42,      # note: random_seed not random_state
    verbose=0            # suppress training output
)
```

**When to use CatBoost:**
- ✅ Datasets with many categorical features (city names, product types, company names)
- ✅ When you want to skip label encoding preprocessing
- ✅ When overfitting on categorical features is a concern
- ✅ Datasets mixing numeric and categorical features
- ❌ Pure numeric datasets — XGBoost or LightGBM are faster
- ❌ Very large datasets — slower than LightGBM
- ❌ Image or text data — deep learning models are better

**MNIST Result:**
```
Accuracy: 95.36% — lowest among three
```
Expected — MNIST has no categorical features so CatBoost's main strength is completely unused here. On our shipment delay dataset with `logistics_company`, `origin_city`, `destination_city` — CatBoost would likely win.

---

### 5. Effect of n_estimators on Accuracy

```
n_estimators=10,  depth=3 → 83.9%
n_estimators=100, depth=6 → 97.05%
```

More trees + more depth = higher accuracy — but with diminishing returns and increasing training time. Finding the right balance is a hyperparameter tuning problem.

### 6. Pixel Values and Image Preprocessing

MNIST images store pixel brightness as integers 0-255 (8-bit grayscale). Before training:

```python
# Normalize to 0-1 range
X_train = X_train / 255.0
```

Normalization makes training faster and more stable — all features on the same scale prevents any single pixel from dominating.

### 7. Why Some Digits Are Harder Than Others

```
Digit 0 → 99% recall ✅ — very distinctive circle shape
Digit 1 → 99% recall ✅ — simple vertical line
Digit 5 → 96% recall   — improved significantly with more trees
Digit 8 → 96% recall   — complex shape, looks like 0, 3, 9
Digit 9 → 96% recall   — confused with 4 and 7
```

More trees specifically helped the hard digits — gradient boosting kept correcting errors on 5, 8, 9 with each new tree.

---

## Final Model Comparison

| Model | Accuracy | Macro F1 | Best For |
|---|---|---|---|
| XGBoost | 97.05% | 0.9703 | General purpose tabular data |
| LightGBM | 97.68% | 0.9767 | Large datasets, speed critical |
| CatBoost | 95.36% | 0.9532 | Categorical feature heavy data |

**Winner on MNIST: LightGBM** — fastest training, highest accuracy on this large numeric dataset.

**Note:** CatBoost's lower score is not a reflection of its quality — MNIST is simply not the right dataset to showcase its strengths.

---

## Algorithm Selection Guide

| Situation | Recommended Algorithm |
|---|---|
| General purpose, first attempt | XGBoost |
| Large dataset (>100k rows) | LightGBM |
| Many categorical features | CatBoost |
| Speed critical production system | LightGBM |
| Small dataset, risk of overfit | XGBoost |
| Mixed numeric + categorical | CatBoost |
| Kaggle competition on tabular data | Try all three, ensemble |

---

## Tools & Libraries

```python
datasets          # Hugging Face dataset loading
pandas            # Data manipulation
numpy             # Numerical operations
scikit-learn      # Evaluation metrics
xgboost           # XGBoost model
lightgbm          # LightGBM model
catboost          # CatBoost model
Pillow            # Image processing (PNG bytes to pixel arrays)
matplotlib        # Visualisation
python-dotenv     # Environment variable management
huggingface_hub   # Authentication
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/shambhavichaugule/gradient-boosting.git
cd gradient-boosting

# Activate virtual environment
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Add your Hugging Face token to .env
echo "HF_TOKEN=your_token_here" > .env

# Run the models
python gradient_boosting.py
```

---

## PM Perspective

This project simulates a real product decision: **which algorithm should power our digit recognition feature?**

### Real World Applications of Digit Recognition

| Application | Use Case |
|---|---|
| Banking | Cheque amount recognition |
| Logistics | Postal code reading on parcels |
| Healthcare | Handwritten prescription digitization |
| Government | Form digitization at scale |
| Retail | Handwritten order processing |

### Algorithm Selection is a Product Decision

**LightGBM for high volume production:**
- Fastest inference — critical when processing thousands of cheques per minute
- Lowest memory footprint — cheaper cloud infrastructure
- 97.68% accuracy — good enough for most banking applications

**XGBoost for regulated industries:**
- More interpretable — can explain why a digit was misclassified
- Stable, conservative — important for audit trails in banking
- Slightly lower accuracy acceptable for compliance reasons

**CatBoost for mixed data pipelines:**
- If digit recognition is part of a larger pipeline with categorical metadata (branch name, account type, form type) — CatBoost handles the full feature set natively

### What 97% Accuracy Means in Production

```
Processing 1,000,000 cheques/day
97% accuracy → 30,000 errors/day → need human review queue
99% accuracy → 10,000 errors/day → smaller review queue
99.9% accuracy → 1,000 errors/day → minimal human intervention
```

As a PM you'd define the acceptable error rate before choosing a model — not after. The business cost of a misread cheque determines your accuracy threshold.

### When NOT to Use Gradient Boosting

- **Real time image recognition at scale** → use a CNN (Convolutional Neural Network) — specifically designed for images, will get 99%+ on MNIST
- **Large language understanding** → use transformers
- **Sequential data (time series)** → use LSTM or temporal models
- **When you need < 1ms inference** → gradient boosting may be too slow, consider simpler models

Gradient boosting is the best algorithm for **tabular data** — structured rows and columns. For images, text, and audio — deep learning models are better. Knowing when NOT to use an algorithm is as important as knowing when to use it.

---

## Complete Classical ML Journey

| Project | Algorithm | Dataset | Best Accuracy |
|---|---|---|---|
| [Linear Regression](https://github.com/shambhavichaugule/linear-regression-project) | Linear Regression | Car prices | R²=0.09 |
| [Logistic Regression](https://github.com/shambhavichaugule/logistic-regression-project) | Logistic Regression | Shipment delays | 50% |
| [Decision Tree & RF](https://github.com/shambhavichaugule/decision-tree-random-forest) | DT + Random Forest | Shipment delays | 56% |
| [SVM](https://github.com/shambhavichaugule/svm) | SVM + TF-IDF | Twitter sentiment | 80% |
| [KNN & Naive Bayes](https://github.com/shambhavishinde/knn-naive-bayes) | KNN + NB | Twitter sentiment | 80% |
| This project | XGBoost + LightGBM + CatBoost | MNIST digits | 97.68% |

# Data Preprocessing — Label Encoding & Normalization

## Overview

Before feeding data into any machine learning model, raw data must be preprocessed. Two of the most fundamental preprocessing techniques are **Label Encoding** (converting text to numbers) and **Normalization** (scaling numbers to the same range).

This document explains both concepts with examples drawn from real projects.

---

## Why Preprocessing Matters

Machine learning models only understand numbers. Real world data contains:
- Text categories: city names, company names, delivery status
- Numbers on vastly different scales: shipping cost (₹1,000-₹100,000) vs quantity (1-50)
- Images: raw pixel bytes instead of usable arrays

Preprocessing converts messy real world data into clean numbers a model can learn from.

---

## Part 1 — Label Encoding

### What It Is

Converting text categories into integers.

```python
# Before encoding
city = ['Mumbai', 'Delhi', 'Bangalore', 'Mumbai', 'Delhi']

# After label encoding
city = [2, 1, 0, 2, 1]
```

sklearn assigns numbers alphabetically:
```
Bangalore → 0
Delhi     → 1
Mumbai    → 2
```

### Code

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['city_enc'] = le.fit_transform(df['city'])
```

---

### The Problem With Label Encoding

It implies an order that doesn't exist:
```
Bangalore = 0
Delhi     = 1
Mumbai    = 2
```

The model might think Mumbai (2) is "greater than" Bangalore (0) — which makes no sense for city names. This is called **ordinal assumption** — assuming categories have a natural rank.

---

### Fix — One Hot Encoding

Creates a separate binary column for each category:

```
Original:          One Hot Encoded:
city               Bangalore  Delhi  Mumbai
Mumbai      →          0        0      1
Delhi       →          0        1      0
Bangalore   →          1        0      0
```

```python
# Using pandas
df_encoded = pd.get_dummies(df['city'])

# Using sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoded = encoder.fit_transform(df[['city']])
```

No false ordering — each city is treated as completely independent.

---

### When to Use Which

| Situation | Use |
|---|---|
| Categories have natural order (small/medium/large) | Label Encoding |
| Categories have no order (city, color, company) | One Hot Encoding |
| Tree based models (Decision Tree, Random Forest, XGBoost) | Label Encoding is fine — trees don't assume order |
| Linear models (Linear/Logistic Regression, SVM) | One Hot Encoding — these assume numeric order |
| Many unique categories (100+ cities) | Label Encoding — one hot creates too many columns |
| Few unique categories (< 10) | One Hot Encoding |

---

### Real Project Example

From the shipment delay project — converting logistics company and city names to numbers:

```python
le = LabelEncoder()
df['logistics_company_enc'] = le.fit_transform(df['logistics_company'])
df['origin_city_enc'] = le.fit_transform(df['origin_city'])
df['destination_city_enc'] = le.fit_transform(df['destination_city'])
```

Used label encoding here because:
- Tree based models (Decision Tree, Random Forest) don't assume numeric order
- Cities had many unique values — one hot would create hundreds of columns

---

### CatBoost — Native Categorical Handling

CatBoost eliminates the need for label encoding entirely:

```python
# XGBoost/LightGBM — manual encoding required
df['city_enc'] = LabelEncoder().fit_transform(df['city'])
model.fit(X_train, y_train)

# CatBoost — pass text columns directly
model.fit(X_train, y_train, cat_features=['city', 'logistics_company'])
```

CatBoost handles the encoding internally using a technique called **target statistics** — more sophisticated than simple label encoding.

---

## Part 2 — Normalization

### What It Is

Scaling numeric features to the same range so no single feature dominates due to its magnitude.

### The Problem Without Normalization

```
Feature 1: shipping_cost → values between ₹1,000 and ₹100,000
Feature 2: quantity      → values between 1 and 50
```

The model sees shipping_cost as 2,000× more important just because the numbers are bigger — even if quantity is equally predictive. The scale of a feature should not determine its importance.

---

### Two Types of Normalization

#### Min-Max Scaling — scales to 0-1 range

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

```
Original: [1000, 50000, 100000]
Scaled:   [0.0,  0.49,  1.0]
```

Formula:
```
scaled = (value - min) / (max - min)
```

Best when you know the data has fixed bounds (like pixel values 0-255).

#### Standard Scaling — scales to mean=0, standard deviation=1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

```
Original: [1000, 50000, 100000]
Scaled:   [-1.2,  0.0,   1.2]
```

Formula:
```
scaled = (value - mean) / std_deviation
```

Best when data follows a normal distribution. Most commonly used in production.

---

### When to Normalize

| Algorithm | Normalize? | Why |
|---|---|---|
| SVM | ✅ Always | Distance based — scale affects margin calculation |
| KNN | ✅ Always | Distance based — large features dominate neighbor search |
| Neural Networks | ✅ Always | Gradient descent converges faster |
| Linear Regression | ✅ Recommended | Faster convergence, stable coefficients |
| Logistic Regression | ✅ Recommended | Faster convergence |
| Decision Tree | ❌ Not needed | Tree splits don't care about scale |
| Random Forest | ❌ Not needed | Ensemble of trees |
| XGBoost / LightGBM | ❌ Not needed | Tree based, scale invariant |
| CatBoost | ❌ Not needed | Tree based, scale invariant |

**Rule of thumb:** Normalize for distance based and gradient based algorithms. Skip for tree based algorithms.

---

### Real Project Examples

**SVM project — StandardScaler for text features:**
```python
# Not needed for TF-IDF since it already produces normalized vectors
# But for numeric features:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**MNIST project — pixel normalization:**
```python
# Divide by 255 to scale pixel values from 0-255 to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0
```

Simple division works here because pixel values have a known fixed range (0-255).

---

### The Critical Rule — Fit on Train, Transform Test

This is the most important rule in preprocessing:

```python
# ✅ Correct order — no data leakage
X_train, X_test = train_test_split(X)

scaler.fit(X_train)                      # learn scale from training data only
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test) # apply same scale to test

# ❌ Wrong — data leakage
scaler.fit(X)                            # test data influences the scale
X_train, X_test = train_test_split(X_scaled)
```

If you fit the scaler on the full dataset:
- Test data influences the mean and standard deviation
- The model indirectly "sees" test data during training
- Your evaluation metrics are optimistically biased

**Same rule applies to:**
- `StandardScaler` — fit on train only
- `MinMaxScaler` — fit on train only
- `TfidfVectorizer` — fit on train only
- `LabelEncoder` — fit on train only

---

## Summary Table

| Technique | What it does | When to use |
|---|---|---|
| Label Encoding | Text → integers | Tree models, many categories, ordered categories |
| One Hot Encoding | Text → binary columns | Linear models, few categories, unordered categories |
| Min-Max Scaling | Scale to 0-1 | Known bounds, Neural Networks, pixel values |
| Standard Scaling | Scale to mean=0 std=1 | Unknown bounds, SVM, KNN, Linear models |
| Divide by constant | Simple scaling | Pixel values (/255), known fixed ranges |

---

## Common Mistakes

**Mistake 1 — Fitting scaler on full dataset before split:**
```python
# ❌ Wrong
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)
```

**Mistake 2 — Not encoding categorical features for linear models:**
```python
# ❌ Wrong — passing text to logistic regression
model.fit(df[['city', 'company']], y)
```

**Mistake 3 — One hot encoding high cardinality features:**
```python
# ❌ Wrong — 500 unique cities creates 500 columns
pd.get_dummies(df['city'])  # when city has 500 unique values
```

**Mistake 4 — Normalizing tree based models unnecessarily:**
```python
# ❌ Unnecessary — XGBoost doesn't need this
scaler.fit_transform(X)
xgb_model.fit(X_scaled, y)  # wasted computation
```

---

## PM Perspective

Preprocessing decisions are product decisions — they affect model performance, engineering complexity, and maintenance cost.

**Label encoding vs one hot encoding:**
- One hot encoding doubles or triples your feature count — larger models, more memory, slower inference
- For a product with 500 cities, label encoding is the pragmatic choice even for linear models

**Normalization in production:**
- The scaler must be saved and deployed alongside the model
- When new data arrives with values outside the training range (new city, extreme price) — your scaler needs updating
- This is called **distribution shift** — a major source of model degradation in production

**What to monitor after launch:**
- Feature distributions — are new values within the training range?
- Encoding coverage — are new categories appearing that weren't in training?
- Scale drift — are feature scales changing over time (inflation affecting prices)?

**Key insight:** Preprocessing is not a one-time step — it's an ongoing maintenance responsibility. As a PM you need to plan for retraining pipelines and data validation before launching any ML feature.

