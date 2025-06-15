import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import ttest_1samp
from sklearn.metrics import accuracy_score


def simulate_meg_data(n_trials=120, n_features=200):
    """
    Simulate MEG data for demonstration.
    Each trial has 200 features representing MEG signals.
    """
    np.random.seed(42)
    data = np.random.randn(n_trials, n_features)
    labels = np.random.choice(['person', 'location', 'object'], size=n_trials)
    trial_types = np.random.choice(['closed_loop', 'open_loop'], size=n_trials)
    return data, labels, trial_types


def build_classifiers():
    """
    Initialize multiple machine learning classifiers.
    """
    return {
        'Lasso GLM': LogisticRegressionCV(cv=6, penalty='l1', solver='saga', max_iter=10000, multi_class='ovr'),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(kernel='linear', probability=True)
    }


def evaluate_classifier(X, y, clf, n_splits=6):
    """
    Perform cross-validation and return accuracy scores.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pipe = make_pipeline(StandardScaler(), clf)
    scores = cross_val_score(pipe, X, y, cv=skf)
    return scores


def compare_closed_open(X, y, trial_types, clf):
    """
    Compare classifier performance between closed- and open-loop trials.
    """
    closed_mask = trial_types == 'closed_loop'
    open_mask = trial_types == 'open_loop'

    closed_scores = evaluate_classifier(X[closed_mask], y[closed_mask], clf)
    open_scores = evaluate_classifier(X[open_mask], y[open_mask], clf)

    print("\n--- Trial-Type Comparison ---")
    print(f"Closed-loop accuracy: {np.mean(closed_scores):.2f} ± {np.std(closed_scores):.2f}")
    print(f"Open-loop accuracy:   {np.mean(open_scores):.2f} ± {np.std(open_scores):.2f}")

    t_stat, p_value = ttest_1samp(closed_scores - open_scores, 0)
    print(f"T-test (Closed > Open): t = {t_stat:.2f}, p = {p_value:.4f}")

    return closed_scores, open_scores


def plot_results(closed_scores, open_scores):
    """
    Plot bar chart comparing closed and open loop classifier accuracies.
    """
    labels = ['Closed Loop', 'Open Loop']
    means = [np.mean(closed_scores), np.mean(open_scores)]
    stds = [np.std(closed_scores), np.std(open_scores)]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, means, yerr=stds, capsize=5, color=['skyblue', 'lightcoral'])
    plt.ylabel('Accuracy')
    plt.title('Classifier Accuracy by Trial Type')
    plt.ylim(0, 1)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


# --- MAIN SCRIPT ---
if __name__ == "__main__":
    # Simulate MEG data (replace with real MEG array in practice)
    X, y, trial_types = simulate_meg_data()

    # Encode categorical variables
    from sklearn.preprocessing import LabelEncoder
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)
    trial_types = np.array(trial_types)

    # Train and evaluate classifiers
    classifiers = build_classifiers()
    for name, clf in classifiers.items():
        print(f"\n>>> Evaluating: {name}")
        scores = evaluate_classifier(X, y_encoded, clf)
        print(f"Mean Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")

    # Compare trial types with best model (e.g., SVM)
    best_clf = classifiers['SVM']
    closed_scores, open_scores = compare_closed_open(X, y_encoded, trial_types, best_clf)

    # Visualize results
    plot_results(closed_scores, open_scores)
