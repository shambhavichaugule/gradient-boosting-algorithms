import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import os
from huggingface_hub import login
load_dotenv()
login(token=os.getenv("HF_TOKEN"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Load MNIST from Hugging Face
dataset = load_dataset("ylecun/mnist")

# Convert to pandas
train_df = dataset['train'].to_pandas()
test_df = dataset['test'].to_pandas()

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nColumns:", train_df.columns.tolist())
print("\nFirst row:")
print(train_df.head(1))
print("\nLabel distribution:")
print(train_df['label'].value_counts().sort_index())

from PIL import Image
import io

def image_to_pixels(image_dict):
    # Convert PNG bytes to pixel array
    img = Image.open(io.BytesIO(image_dict['bytes']))
    return np.array(img).flatten()

print("Converting images to pixel arrays...")

# Convert all images
X_train = np.array([image_to_pixels(img) for img in train_df['image']])
X_test = np.array([image_to_pixels(img) for img in test_df['image']])

y_train = train_df['label'].values
y_test = test_df['label'].values

# Normalize pixel values to 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Pixel value range: {X_train.min()} to {X_train.max()}")

# Visualize first 10 digits
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.flatten()

for i in range(10):
    # Reshape 784 pixels back to 28x28 image
    img = X_train[i].reshape(28, 28)
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Label: {y_train[i]}")
    axes[i].axis('off')

plt.suptitle("Sample MNIST Digits")
plt.tight_layout()
plt.savefig("sample_digits.png")
print("Sample digits saved!")

print("\n" + "="*50)
print("XGBOOST RESULTS")
print("="*50)

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=2
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, xgb_pred))

print("\n" + "="*50)
print("LIGHTGBM RESULTS")
print("="*50)

lgb_model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=2,
    verbose=-1          # suppress training logs
)

lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, lgb_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, lgb_pred))

print("\n" + "="*50)
print("CATBOOST RESULTS")
print("="*50)

cat_model = CatBoostClassifier(
    iterations=100,       # same as n_estimators
    depth=6,
    learning_rate=0.1,
    random_seed=42,
    verbose=0             # suppress training logs
)

cat_model.fit(X_train, y_train)
cat_pred = cat_model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, cat_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, cat_pred))

print("\n" + "="*50)
print("FINAL MODEL COMPARISON")
print("="*50)

models = {
    'XGBoost': xgb_pred,
    'LightGBM': lgb_pred,
    'CatBoost': cat_pred
}

print(f"\n{'Model':<15} {'Accuracy':>10} {'Macro F1':>10}")
print("-" * 40)

for name, pred in models.items():
    from sklearn.metrics import f1_score
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    print(f"{name:<15} {acc:>10.4f} {f1:>10.4f}")

# Comparison chart
models = ['XGBoost', 'LightGBM', 'CatBoost']
accuracies = [0.9705, 0.9768, 0.9536]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color=['#2196F3', '#4CAF50', '#FF9800'])
plt.ylim(0.9, 1.0)
plt.title('Gradient Boosting Model Comparison — MNIST')
plt.ylabel('Accuracy')

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("model_comparison.png")
print("Chart saved!")