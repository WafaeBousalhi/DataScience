# 🧠 MEG-Based Classifier for Episodic Memory Retrieval

**Goal**: Build and evaluate machine learning classifiers (GLM, KNN, SVM) that decode neural MEG activity associated with non-target elements during episodic memory retrieval, distinguishing between closed-loop and open-loop associative structures.

---

## 🧪 Research Context

Based on experiments conducted in the lab of Dr. Daniel Bush (UCL), participants performed an associative memory task involving encoding and retrieval of event pairs (e.g., \[location, person, object/animal]). Closed-loop trials form all-to-all associations, allowing pattern completion during retrieval. Open-loop trials do not.

Multivariate pattern analysis (MVPA) was applied to MEG data to detect if the classifier could decode the semantic category of non-target elements during retrieval—indicating holistic recollection.

---

## 📊 Data Description

Each trial is represented as:

* MEG signals from 0.5s before to 3s after cue word onset
* Labeled by category: `person`, `location`, `object/animal`
* Label for trial type: `closed_loop` or `open_loop`

---

## 🧠 Classification Pipeline

```python
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Simulated example data
data = np.random.rand(100, 200)   # 100 trials, 200 features (e.g. MEG channels x time bins)
labels = np.random.choice(['person', 'location', 'object'], size=100)
trial_type = np.random.choice(['closed_loop', 'open_loop'], size=100)

# Define classifiers
classifiers = {
    'Lasso GLM': LogisticRegressionCV(cv=6, penalty='l1', solver='saga', max_iter=5000, multi_class='ovr'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='linear')
}

# Evaluate each classifier with 6-fold CV
for name, clf in classifiers.items():
    pipe = make_pipeline(StandardScaler(), clf)
    scores = cross_val_score(pipe, data, labels, cv=StratifiedKFold(n_splits=6))
    print(f"{name} Accuracy: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
```

---

## 🔍 Analysis Focus

* Compare decoding accuracy between **closed-loop** and **open-loop** trials.
* Use non-parametric randomization test to evaluate if accuracy is above chance (33%).
* Time-lock decoding performance to specific post-stimulus time windows (e.g., 100–250ms).

---

## 📈 Visualization Ideas

* Line plot of classifier accuracy over time (sliding window)
* Bar plot comparing accuracy (closed-loop vs open-loop)
* Heatmaps of classifier weights (brain topographies)

---

## 🔬 Scientific Relevance

This project is a novel contribution to cognitive neuroscience and machine learning, showing how the brain’s ability to complete memory patterns can be quantified with classifiers trained on MEG activity.

---

🧾 *Data is anonymised. Classifiers trained on high-dimensional MEG vectors per trial using temporal and spatial features.*

👩‍💻 *Project developed as part of my MSci Neuroscience degree at UCL with the Bush Lab.*
