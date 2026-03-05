
---

### PROJECT 4: `diabetes-prediction-ml`

**File: `main.py`**
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')
os.makedirs('images', exist_ok=True)
np.random.seed(42)

# ── Load Pima Indians Diabetes Dataset ──
print("📊 Loading Pima Indians Diabetes dataset...")
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columns = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigree','Age','Outcome']
df = pd.read_csv(url, names=columns)
print(f"✅ Loaded {len(df)} records | Diabetic: {df['Outcome'].sum()} | Healthy: {(1-df['Outcome']).sum()}")

# ── EDA Plots ──
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, col in enumerate(columns[:-1]):
    ax = axes[i//4][i%4]
    df[df['Outcome']==0][col].hist(ax=ax, alpha=0.6, color='#2B7A78', label='Healthy', bins=20)
    df[df['Outcome']==1][col].hist(ax=ax, alpha=0.6, color='#E74C3C', label='Diabetic', bins=20)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=7)
plt.suptitle('Feature Distributions by Diabetes Status', fontsize=14)
plt.tight_layout(); plt.savefig('images/feature_distributions.png', dpi=150)
print("✅ Saved: images/feature_distributions.png")

# ── Correlation Heatmap ──
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
im = ax.imshow(corr, cmap='RdYlGn', vmin=-1, vmax=1)
ax.set_xticks(range(len(columns))); ax.set_yticks(range(len(columns)))
ax.set_xticklabels(columns, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(columns, fontsize=9)
for i in range(len(columns)):
    for j in range(len(columns)):
        ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im); plt.title('Correlation Heatmap')
plt.tight_layout(); plt.savefig('images/correlation_heatmap.png', dpi=150)
print("✅ Saved: images/correlation_heatmap.png")

# ── Train Models ──
X = df.drop('Outcome', axis=1); y = df['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train); X_test_sc = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

print("\n📈 Training models...")
results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    cv = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='roc_auc').mean()
    results[name] = {'auc': auc, 'cv_auc': cv, 'y_prob': y_prob}
    print(f"  {name}: Test AUC={auc:.4f}, CV AUC={cv:.4f}")

# ── ROC Curves ──
fig, ax = plt.subplots(figsize=(8, 6))
for (name, res), color in zip(results.items(), ['#2B7A78','#E74C3C','#3498DB']):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
    ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={res['auc']:.3f})")
ax.plot([0,1],[0,1],'k--'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
ax.set_title('ROC Curves — Diabetes Prediction'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('images/roc_curves.png', dpi=150)
print("✅ Saved: images/roc_curves.png")

# ── Feature Importance ──
rf = models['Random Forest']
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
importance.plot(kind='barh', color='#2B7A78', ax=ax)
ax.set_title('Feature Importance — Random Forest'); ax.grid(axis='x', alpha=0.3)
plt.tight_layout(); plt.savefig('images/feature_importance.png', dpi=150)
print("✅ Saved: images/feature_importance.png")

print("\n🏁 Done! All plots saved to /images/")
