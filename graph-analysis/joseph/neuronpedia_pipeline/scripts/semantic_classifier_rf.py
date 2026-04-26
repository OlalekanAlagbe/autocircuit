#!/usr/bin/env python3
"""
Random Forest Semantic Classifier for SAE features.

A supervised replacement for the keyword-based v2 classifier
(annotate_features_v2.py). Trains a RandomForestClassifier on manually
labeled features from ``semantic_taxonomy_annotations.csv`` using a feature
vector that combines:

  1. TF-IDF of the explanation text (from ``activation_examples`` or an
     explicit ``explanation`` column when present).
  2. Layer number.
  3. Activation count.
  4. Cross-circuit frequency (optional lookup from
     ``data/stage_1_5_bottleneck_library.json``).
  5. Non-Latin script flag (1 if text contains any non-latin alpha chars).

Usage
-----
    # Train + save model
    python semantic_classifier_rf.py --train

    # Leave-one-out cross-validation eval
    python semantic_classifier_rf.py --evaluate

    # Predict a single feature
    python semantic_classifier_rf.py --predict "sky/heaven direction" \\
        --layer 5 --activation-count 42 --frequency 3

If the trained joblib model is missing, :func:`classify_feature` falls back
to ``classify_from_explanation`` from ``annotate_features_v2``.
"""

import argparse
import csv
import io
import json
import logging
import sys
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Windows UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneOut
import joblib

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
TRAIN_CSV = BASE / "semantic_taxonomy_annotations_auto.csv"
BOTTLENECK_LIB = BASE / "data" / "stage_1_5_bottleneck_library.json"
CLASSIFIER_DIR = BASE / "data" / "classifier"
MODEL_PATH = CLASSIFIER_DIR / "rf_classifier.joblib"
EVAL_REPORT_PATH = CLASSIFIER_DIR / "eval_report.txt"

# In-process cache for the loaded model bundle
_MODEL_CACHE: Optional[Dict] = None


# ──────────────────────────────────────────────────────────────────────────
# Feature helpers
# ──────────────────────────────────────────────────────────────────────────
def has_non_latin(text: str) -> int:
    """Return 1 if ``text`` has any non-latin alphabetic characters."""
    if not text:
        return 0
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch, '')
        except ValueError:
            continue
        if name and 'LATIN' not in name:
            return 1
    return 0


def load_bottleneck_frequencies(path: Path = BOTTLENECK_LIB) -> Dict[str, int]:
    """Load per-feature cross-circuit frequencies from the bottleneck library.

    Returns a dict keyed by ``feature_label`` (e.g. ``"L5_F7993995"``).
    Returns an empty dict if the file doesn't exist or can't be parsed.
    """
    if not path.exists():
        logger.info("Bottleneck library not found at %s; frequencies default to 1.", path)
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse %s: %s", path, exc)
        return {}

    freq: Dict[str, int] = {}

    def _walk(obj):
        if isinstance(obj, dict):
            label = obj.get('feature_label') or obj.get('label')
            count = (
                obj.get('cross_circuit_count')
                or obj.get('frequency')
                or obj.get('circuit_count')
                or obj.get('count')
            )
            if label and isinstance(count, (int, float)):
                freq[label] = int(count)
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for item in obj:
                _walk(item)

    _walk(data)
    return freq


# ──────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────
def load_training_data(
    csv_path: Path = TRAIN_CSV,
    freq_lookup: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[int], List[int], List[int], List[int], List[str], List[str]]:
    """Load training rows from the annotations CSV.

    Returns a tuple of parallel lists:
        (texts, layers, activation_counts, frequencies, non_latin_flags,
         labels, feature_labels)
    Only rows with a non-empty ``manual_category`` are included.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    freq_lookup = freq_lookup or {}

    texts: List[str] = []
    layers: List[int] = []
    counts: List[int] = []
    frequencies: List[int] = []
    non_latin: List[int] = []
    labels: List[str] = []
    feature_labels: List[str] = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # Prefer a dedicated "explanation" column if it exists; otherwise use
        # "activation_examples".
        text_col = 'explanation' if 'explanation' in cols else 'activation_examples'

        for row in reader:
            label = (row.get('manual_category') or '').strip()
            if not label:
                continue

            text = (row.get(text_col) or '').strip()
            if not text and text_col != 'activation_examples':
                text = (row.get('activation_examples') or '').strip()

            try:
                layer = int(row.get('layer') or 0)
            except ValueError:
                layer = 0
            try:
                count = int(row.get('activation_count') or 0)
            except ValueError:
                count = 0

            feat_label = (row.get('feature_label') or '').strip()
            frequency = int(freq_lookup.get(feat_label, 1))

            texts.append(text)
            layers.append(layer)
            counts.append(count)
            frequencies.append(frequency)
            non_latin.append(has_non_latin(text))
            labels.append(label)
            feature_labels.append(feat_label)

    return texts, layers, counts, frequencies, non_latin, labels, feature_labels


# ──────────────────────────────────────────────────────────────────────────
# Vectorization
# ──────────────────────────────────────────────────────────────────────────
def build_vectorizer() -> TfidfVectorizer:
    """Create the TF-IDF vectorizer used for training and inference."""
    return TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True,
    )


def stack_features(
    tfidf_matrix,
    layers: List[int],
    counts: List[int],
    frequencies: List[int],
    non_latin: List[int],
):
    """Concatenate TF-IDF with numeric features into a single sparse matrix."""
    numeric = np.array(
        [layers, counts, frequencies, non_latin],
        dtype=float,
    ).T  # shape (n_samples, 4)
    numeric_sparse = csr_matrix(numeric)
    return hstack([tfidf_matrix, numeric_sparse]).tocsr()


# ──────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────
def train(csv_path: Path = TRAIN_CSV) -> Dict:
    """Fit the random forest and save a joblib bundle to ``MODEL_PATH``."""
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)

    freq_lookup = load_bottleneck_frequencies()
    texts, layers, counts, frequencies, non_latin, labels, _ = load_training_data(
        csv_path, freq_lookup=freq_lookup,
    )

    n = len(labels)
    if n == 0:
        raise RuntimeError("No labeled training rows found.")

    logger.info("Loaded %d labeled training rows.", n)

    vectorizer = build_vectorizer()
    tfidf = vectorizer.fit_transform(texts)
    X = stack_features(tfidf, layers, counts, frequencies, non_latin)
    y = np.array(labels)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced',
    )
    model.fit(X, y)

    train_pred = model.predict(X)
    train_acc = accuracy_score(y, train_pred)

    # Class distribution
    unique, cnt = np.unique(y, return_counts=True)
    dist = dict(zip(unique.tolist(), cnt.tolist()))

    print(f"Training accuracy: {train_acc:.4f}")
    print("Class distribution:")
    for cls, c in sorted(dist.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:30s}: {c}")

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "categories": sorted(unique.tolist()),
        "numeric_feature_order": ["layer", "activation_count", "frequency", "non_latin"],
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved model bundle to: {MODEL_PATH}")

    return bundle


# ──────────────────────────────────────────────────────────────────────────
# Evaluation (leave-one-out)
# ──────────────────────────────────────────────────────────────────────────
def evaluate(csv_path: Path = TRAIN_CSV) -> None:
    """Run leave-one-out cross-validation and write a report to disk."""
    CLASSIFIER_DIR.mkdir(parents=True, exist_ok=True)

    freq_lookup = load_bottleneck_frequencies()
    texts, layers, counts, frequencies, non_latin, labels, _ = load_training_data(
        csv_path, freq_lookup=freq_lookup,
    )

    n = len(labels)
    if n < 2:
        raise RuntimeError("Need at least 2 labeled rows for LOO evaluation.")

    logger.info("Running leave-one-out CV on %d samples.", n)

    y = np.array(labels)
    indices = np.arange(n)
    loo = LeaveOneOut()

    y_true: List[str] = []
    y_pred: List[str] = []
    skipped = 0

    for train_idx, test_idx in loo.split(indices):
        try:
            train_texts = [texts[i] for i in train_idx]
            train_layers = [layers[i] for i in train_idx]
            train_counts = [counts[i] for i in train_idx]
            train_freqs = [frequencies[i] for i in train_idx]
            train_latin = [non_latin[i] for i in train_idx]
            train_y = y[train_idx]

            vect = build_vectorizer()
            train_tfidf = vect.fit_transform(train_texts)
            X_train = stack_features(
                train_tfidf, train_layers, train_counts, train_freqs, train_latin,
            )

            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                class_weight='balanced',
            )
            model.fit(X_train, train_y)

            # Test fold (single sample)
            ti = test_idx[0]
            test_tfidf = vect.transform([texts[ti]])
            X_test = stack_features(
                test_tfidf,
                [layers[ti]],
                [counts[ti]],
                [frequencies[ti]],
                [non_latin[ti]],
            )
            pred = model.predict(X_test)[0]

            y_true.append(labels[ti])
            y_pred.append(pred)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.debug("Skipped fold %s: %s", test_idx, exc)
            continue

    if not y_true:
        raise RuntimeError("No successful LOO folds; cannot produce a report.")

    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0)
    all_classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    cm_lines = ["Confusion matrix (rows=true, cols=pred):"]
    header = "              " + " ".join(f"{c[:10]:>10s}" for c in all_classes)
    cm_lines.append(header)
    for cls, row in zip(all_classes, cm):
        cm_lines.append(
            f"{cls[:12]:12s}  " + " ".join(f"{v:10d}" for v in row)
        )
    cm_text = "\n".join(cm_lines)

    lines = [
        "Random Forest Semantic Classifier - LOO Evaluation",
        "=" * 60,
        f"Samples evaluated: {len(y_true)}",
        f"Folds skipped:     {skipped}",
        f"Overall accuracy:  {acc:.4f}",
        "",
        "Classification report:",
        report,
        "",
        cm_text,
        "",
    ]
    text = "\n".join(lines)

    print(text)
    with open(EVAL_REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"Wrote eval report to: {EVAL_REPORT_PATH}")


# ──────────────────────────────────────────────────────────────────────────
# Single-feature prediction
# ──────────────────────────────────────────────────────────────────────────
def _load_model_cached() -> Optional[Dict]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    if not MODEL_PATH.exists():
        return None
    try:
        _MODEL_CACHE = joblib.load(MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load joblib model %s: %s", MODEL_PATH, exc)
        return None
    return _MODEL_CACHE


def classify_feature(
    explanation: str,
    layer: int,
    activation_count: int = 0,
    frequency: int = 1,
) -> str:
    """Classify a single feature with the trained RF model.

    Falls back to the keyword classifier from ``annotate_features_v2`` if the
    joblib model hasn't been trained yet.
    """
    bundle = _load_model_cached()
    if bundle is None:
        logger.warning(
            "RF model not found at %s; falling back to keyword classifier.",
            MODEL_PATH,
        )
        try:
            from annotate_features_v2 import classify_from_explanation
        except ImportError:
            # Try absolute import if invoked outside the scripts dir
            sys.path.insert(0, str(Path(__file__).parent))
            from annotate_features_v2 import classify_from_explanation  # noqa
        return classify_from_explanation(explanation, layer)

    vectorizer = bundle["vectorizer"]
    model = bundle["model"]

    tfidf = vectorizer.transform([explanation or ""])
    X = stack_features(
        tfidf,
        [int(layer)],
        [int(activation_count)],
        [int(frequency)],
        [has_non_latin(explanation or "")],
    )
    return str(model.predict(X)[0])


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random forest semantic classifier for SAE features.",
    )
    parser.add_argument('--train', action='store_true', help='Train and save the model.')
    parser.add_argument('--evaluate', action='store_true', help='Run LOO cross-validation.')
    parser.add_argument('--predict', type=str, default=None,
                        help='Explanation text to classify.')
    parser.add_argument('--layer', type=int, default=0, help='Layer number for --predict.')
    parser.add_argument('--activation-count', type=int, default=0,
                        help='Activation count for --predict.')
    parser.add_argument('--frequency', type=int, default=1,
                        help='Cross-circuit frequency for --predict.')

    args = parser.parse_args()

    if not any([args.train, args.evaluate, args.predict]):
        parser.print_help()
        return

    if args.train:
        train()

    if args.evaluate:
        evaluate()

    if args.predict is not None:
        category = classify_feature(
            args.predict,
            layer=args.layer,
            activation_count=args.activation_count,
            frequency=args.frequency,
        )
        print(f"Predicted category: {category}")


if __name__ == '__main__':
    main()
