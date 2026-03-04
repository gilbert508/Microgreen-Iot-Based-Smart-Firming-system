# ================================
# CSE 442 Lab Test Solution
# Semi-Supervised Learning + Ensemble
# ================================

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# Load Dataset
# ==========================================
iris = load_iris()
X = iris.data
y = iris.target

# Train-Test Split
X_train, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ==========================================
# Simulate Missing Labels (40% unlabeled)
# ==========================================
np.random.seed(42)
y_train = y_train_full.copy()

n_unlabeled = int(0.4 * len(y_train))
unlabeled_indices = np.random.choice(len(y_train), n_unlabeled, replace=False)

y_train[unlabeled_indices] = -1   # -1 means unlabeled

print("Total training samples:", len(y_train))
print("Unlabeled samples:", np.sum(y_train == -1))


# ==========================================
# Task 1: Baseline Supervised Learning
# Train RF only on labeled data
# ==========================================
X_labeled = X_train[y_train != -1]
y_labeled = y_train[y_train != -1]

rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_labeled, y_labeled)

y_pred_baseline = rf_baseline.predict(X_test)

print("\n===== Baseline Random Forest =====")
print("Accuracy:", accuracy_score(y_test, y_pred_baseline))
print("Precision:", precision_score(y_test, y_pred_baseline, average='macro'))
print("Recall:", recall_score(y_test, y_pred_baseline, average='macro'))
print("F1-score:", f1_score(y_test, y_pred_baseline, average='macro'))


# ==========================================
# Task 2: Label Propagation
# ==========================================
label_prop_model = LabelPropagation()
label_prop_model.fit(X_train, y_train)

# Get propagated labels
y_train_propagated = label_prop_model.transduction_

print("\nLabel Propagation completed.")


# ==========================================
# Task 3: Random Forest after Label Propagation
# ==========================================
rf_after_lp = RandomForestClassifier(random_state=42)
rf_after_lp.fit(X_train, y_train_propagated)

y_pred_lp = rf_after_lp.predict(X_test)

print("\n===== Random Forest After Label Propagation =====")
print("Accuracy:", accuracy_score(y_test, y_pred_lp))
print("Precision:", precision_score(y_test, y_pred_lp, average='macro'))
print("Recall:", recall_score(y_test, y_pred_lp, average='macro'))
print("F1-score:", f1_score(y_test, y_pred_lp, average='macro'))

print("\nComment:")
print("After using Label Propagation, the model uses more training data.")
print("Performance usually improves compared to baseline.")


# ==========================================
# Task 4: Train Five Supervised Models
# ==========================================
models = {
    "SVM": SVC(probability=True, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

model_scores = {}

print("\n===== Individual Model Performance =====")

for name, model in models.items():
    model.fit(X_train, y_train_propagated)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_scores[name] = acc
    
    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1-score:", f1_score(y_test, y_pred, average='macro'))


# ==========================================
# Task 5: Select Top 3 Models & Assign Weights
# ==========================================
sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

top3 = sorted_models[:3]
print("\nTop 3 Models:")
print(top3)

# Assign weights proportional to accuracy
weights = [score for _, score in top3]
model_names = [name for name, _ in top3]

selected_models = [(name, models[name]) for name in model_names]

print("Assigned Weights:", weights)


# ==========================================
# Task 6: Weighted Voting Ensemble
# ==========================================
ensemble = VotingClassifier(
    estimators=selected_models,
    voting='soft',
    weights=weights
)

ensemble.fit(X_train, y_train_propagated)
y_pred_ensemble = ensemble.predict(X_test)

print("\n===== Weighted Voting Ensemble Performance =====")
print("Accuracy:", accuracy_score(y_test, y_pred_ensemble))
print("Precision:", precision_score(y_test, y_pred_ensemble, average='macro'))
print("Recall:", recall_score(y_test, y_pred_ensemble, average='macro'))
print("F1-score:", f1_score(y_test, y_pred_ensemble, average='macro'))
