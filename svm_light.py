"""Train and evaluate the project's classifier with the SVMlight command-line tools."""

from pathlib import Path
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd


PART2_CV_ACCURACIES = {
    "Naive Bayes": 87.62,
    "KNN (k=3)": 56.73,
    "Decision Tree (depth=6)": 76.53,
}
SVM_LIGHT_CITATION = (
    "T. Joachims, Making Large-Scale SVM Learning Practical, "
    "Advances in Kernel Methods - Support Vector Learning, MIT Press, 1999."
)


def _find_executable(environment_variable, default_name):
    configured_name = os.environ.get(environment_variable, default_name)
    executable = shutil.which(configured_name)
    if executable:
        return executable
    configured_path = Path(configured_name)
    if configured_path.is_file():
        return str(configured_path)
    raise FileNotFoundError(
        f"Could not find {default_name}. Install SVMlight or use Docker; "
        f"{environment_variable} can point to the executable."
    )


def _write_svmlight_file(file_path, features, labels):
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=int)
    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("SVMlight labels must be binary values 0 or 1.")

    with file_path.open("w", encoding="utf-8") as output:
        for row, label in zip(features, labels):
            target = 1 if label == 1 else -1
            feature_tokens = [
                f"{index}:{value:g}"
                for index, value in enumerate(row, start=1)
                if value != 0
            ]
            output.write(" ".join([str(target), *feature_tokens]) + "\n")


def _run_command(command):
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.rstrip())
    if result.stderr.strip():
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(command)}"
        )


def run_svmlight(training_features, training_labels, testing_features, testing_labels, output_directory):
    """Train a linear SVM with svm_learn, then classify the held-out wines."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    training_file = output_directory / "training.svmlight"
    testing_file = output_directory / "testing.svmlight"
    model_file = output_directory / "svm_light_model.txt"
    scores_file = output_directory / "decision_scores.txt"
    _write_svmlight_file(training_file, training_features, training_labels)
    _write_svmlight_file(testing_file, testing_features, testing_labels)

    svm_learn = _find_executable("SVM_LEARN", "svm_learn")
    svm_classify = _find_executable("SVM_CLASSIFY", "svm_classify")
    _run_command([svm_learn, "-t", "0", str(training_file), str(model_file)])
    _run_command([svm_classify, str(testing_file), str(model_file), str(scores_file)])

    decision_scores = np.loadtxt(scores_file, dtype=float, ndmin=1)
    if len(decision_scores) != len(testing_labels):
        raise ValueError("SVMlight returned a different number of scores than test rows.")
    if not np.isfinite(decision_scores).all():
        raise ValueError("SVMlight returned a non-finite decision score.")

    predictions = (decision_scores >= 0).astype(int)
    return predictions, decision_scores


def build_part2_comparison(svm_accuracy):
    """Compare the held-out SVM accuracy with the Part 2 fold averages."""

    rows = []
    for algorithm, part2_accuracy in PART2_CV_ACCURACIES.items():
        rows.append(
            {
                "Part 2 algorithm": algorithm,
                "Part 2 mean CV accuracy (%)": part2_accuracy,
                "SVM holdout accuracy (%)": round(svm_accuracy * 100, 2),
                "Difference (percentage points)": round(
                    svm_accuracy * 100 - part2_accuracy, 2
                ),
                "Part 2 evaluation": "5-fold CV, 1,010 wines",
            }
        )
    return pd.DataFrame(rows)
