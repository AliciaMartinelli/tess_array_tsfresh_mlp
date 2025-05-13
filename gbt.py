
import os
import random
import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import f1_score, precision_recall_curve, confusion_matrix, roc_auc_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import joblib

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

X_train = np.load("X_train_tess.npy")
y_train = np.load("y_train_tess.npy")

X_val = np.load("X_val_tess.npy")
y_val = np.load("y_val_tess.npy")

X_test = np.load("X_test_tess.npy")
y_test = np.load("y_test_tess.npy")

def objective(trial):
    num_leaves = trial.suggest_int("num_leaves", 20, 150)
    learning_rate = trial.suggest_float("learning_rate", 0.005, 0.2, log=True)
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 12)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    f1_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val_inner = X_train[train_idx], X_train[val_idx]
        y_tr, y_val_inner = y_train[train_idx], y_train[val_idx]

        model = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=SEED,
            verbose=-1
        )

        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_val_inner)[:, 1]

        precision_vals, recall_vals, thresholds = precision_recall_curve(y_val_inner, y_pred_proba)
        f1_vals = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-8)
        optimal_idx = np.argmax(f1_vals)
        optimal_threshold = thresholds[optimal_idx]

        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        f1_scores.append(f1_score(y_val_inner, y_pred))

    return np.mean(f1_scores)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60)

X_final_train = np.concatenate([X_train, X_val], axis=0)
y_final_train = np.concatenate([y_train, y_val], axis=0)

best_params = study.best_trial.params

final_model = lgb.LGBMClassifier(
    num_leaves=best_params["num_leaves"],
    learning_rate=best_params["learning_rate"],
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    class_weight="balanced",
    random_state=SEED,
     verbose=-1
)

final_model.fit(X_final_train, y_final_train)

joblib.dump(final_model, "tess_gbt.pkl")
print("save model as 'tess_gbt.h5'.")

y_test_proba = final_model.predict_proba(X_test)[:, 1]

precision_tests, recall_tests, thresholds = precision_recall_curve(y_test, y_test_proba)
f1_tests = 2 * (precision_tests * recall_tests) / (precision_tests + recall_tests + 1e-8)
optimal_idx = np.argmax(f1_tests)
optimal_threshold = thresholds[optimal_idx]

y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

auc_score = roc_auc_score(y_test, y_test_proba)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

cm = confusion_matrix(y_test, y_test_pred)
p_misclass = cm[1, 0] / cm[1].sum() * 100 if cm[1].sum() != 0 else 0
n_misclass = cm[0, 1] / cm[0].sum() * 100 if cm[0].sum() != 0 else 0

print(f"\nFinal Test Set Evaluation:")
print(f"AUC: {auc_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
print(f"pMisclass (Positive Misclassification Rate): {p_misclass:.2f}%")
print(f"nMisclass (Negative Misclassification Rate): {n_misclass:.2f}%")

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_test_proba)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', label="No Skill Line")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
plt.axhline(y=np.mean(y_test), linestyle='--', color='gray', label="No Skill Line")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.tight_layout()
plt.show()