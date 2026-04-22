---
name: classify-feature
description: Use when the user wants to classify an SAE feature's semantic category, train the random forest classifier, or evaluate classifier performance. Wraps the TF-IDF + layer + frequency random forest trained on 80 manually annotated GEMMA bottleneck features.
---

# Semantic Feature Classification (Random Forest)

Classify SAE features into semantic categories (SEMANTICS:CODE, SEMANTICS:CONCEPT, SYNTAX, POLYSEMANTIC, UNKNOWN) using a random forest trained on manually annotated bottleneck features.

## Instructions

1. **Train the classifier (first run only, or after annotations update):**

```bash
python scripts/semantic_classifier_rf.py --train
```

Loads `semantic_taxonomy_annotations_auto.csv` (80 labeled features), builds TF-IDF vectors from explanations plus numeric features (layer, activation count, cross-circuit frequency, non-latin script flag), trains `RandomForestClassifier(n_estimators=100, max_depth=8)`, and saves the bundle to `data/classifier/rf_classifier.joblib`.

2. **Run leave-one-out cross-validation:**

```bash
python scripts/semantic_classifier_rf.py --evaluate
```

Writes a full classification report (per-class precision/recall/F1 + confusion matrix) to `data/classifier/eval_report.txt`.

3. **Classify a single feature:**

```bash
python scripts/semantic_classifier_rf.py --predict "HTML formatting tags and structural markup" --layer 3
```

Optional args: `--activation-count N`, `--frequency N` (cross-circuit frequency from the bottleneck library).

4. **Use from Python:**

```python
from semantic_classifier_rf import classify_feature
category = classify_feature(
    explanation="HTML formatting tags and structural markup",
    layer=3,
    activation_count=10,
    frequency=3,
)
# Returns: 'SEMANTICS:CODE'
```

If the trained model is missing, the function falls back to `annotate_features_v2.classify_from_explanation` (keyword-based classifier).

## Output

- `data/classifier/rf_classifier.joblib` — serialized model bundle (model, TF-IDF vectorizer, category labels, feature column order)
- `data/classifier/eval_report.txt` — LOO evaluation report with classification metrics and confusion matrix
- Single-feature predict: prints the category to stdout

## Example Interaction

**User:** Train the semantic classifier and tell me how accurate it is

**Commands:**

```bash
python scripts/semantic_classifier_rf.py --train
python scripts/semantic_classifier_rf.py --evaluate
```

**Expected output:**

```
INFO: Loaded 80 labeled training rows.
Training accuracy: 0.9500
Class distribution:
  SEMANTICS:CODE      : 47
  POLYSEMANTIC        : 18
  SEMANTICS:CONCEPT   : 9
  SYNTAX              : 4
  UNKNOWN             : 2

Saved model bundle to: data/classifier/rf_classifier.joblib

--- Evaluation ---
Overall accuracy: 0.5250 (LOO cross-validation)
Classification report:
                   precision    recall  f1-score  support
  SEMANTICS:CODE       0.60      0.81      0.69       47
  POLYSEMANTIC         0.50      0.06      0.10       18
  ...
```

## Common Issues

- **No labeled training rows found:** The script expects `manual_category` column filled in `semantic_taxonomy_annotations_auto.csv`. Check the file exists and has labels.
- **Low LOO accuracy (~52%):** Expected with 80 training samples. The dataset is heavily imbalanced (47 CODE, 2 UNKNOWN) so minority classes get 0% recall. Accuracy is roughly tied with the keyword classifier (51.2%). Upgrade path: collect more labels or switch to sentence embeddings.
- **Joblib file missing on predict:** The `classify_feature()` function falls back to the keyword classifier automatically — no error, but predictions may differ.
- **Module import error for sklearn:** Install via `pip install scikit-learn>=1.3 joblib>=1.3`.

## When to Use

- Classifying a newly discovered bottleneck feature
- Evaluating whether the random forest has improved after collecting more labels
- Running batch semantic classification as part of a larger analysis pipeline
- Checking semantic purity of a feature before including it in a steering experiment

## Related Skills

- `/neuronpedia-fetch` — fetch the feature explanation from Neuronpedia before classifying
- `/steering-validate` — run steering experiments on classified features

## Prerequisites

- `scikit-learn>=1.3.0`, `scipy>=1.11.0`, `joblib>=1.3.0` (all in `config/requirements.txt`)
- `semantic_taxonomy_annotations_auto.csv` in the project root (80 manually labeled features)
- Optional: `data/stage_1_5_bottleneck_library.json` for cross-circuit frequency lookups

## Performance Notes

| Classifier | Accuracy | Notes |
|-----------|----------|-------|
| Random Forest (LOO) | 52.5% | Infrastructure ready, limited by training data size |
| Keyword (v2) | 51.2% | Baseline regex matching |

Neither classifier is production-grade yet. Use with judgment and verify edge cases manually. A sentence-embedding-based upgrade is planned but requires more labels.
