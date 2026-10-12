#!/usr/bin/env python3
"""Train and evaluate the linear SVMlight wine classifier."""

# Run this file from the folder containing the two Excel datasets.
# The setup cell downloads and builds SVMlight if the CLI tools are missing.

# %% [markdown]
# # SVMlight: Wine Classification
#
# **Dataset:** Training dataset.xlsx (200 wines) and Testing dataset.xlsx (19 wines, 305 binary sensory features)  
# **Class:** 1 = 90+ points, 0 = below 90 points  
# **Algorithm:** Linear support vector machine with SVMlight
#
# This notebook follows the train, predict, and evaluate workflow for the provided wine split. Keep both Excel files in the same folder as this notebook. The SVMlight command-line programs `svm_learn` and `svm_classify` must be installed and available on PATH.

# %% [markdown]
# ## 1. Load the data
#
# The first column contains wine names, the second column contains the grade class, and the remaining 305 columns contain binary sensory features.

# %%
from pathlib import Path
import os
import shutil
import subprocess

import numpy as np
import pandas as pd

TRAINING_FILE = Path("Training dataset.xlsx")
TESTING_FILE = Path("Testing dataset.xlsx")
LABEL_COLUMN = "Grade class 1: 90+  0:90-"

training_data = pd.read_excel(TRAINING_FILE)
testing_data = pd.read_excel(TESTING_FILE)

if training_data.columns.tolist() != testing_data.columns.tolist():
    raise ValueError("Training and testing columns do not match.")
if LABEL_COLUMN not in training_data.columns:
    raise ValueError(f"Could not find the label column: {LABEL_COLUMN}")

FEATURE_COLUMNS = training_data.columns[2:].tolist()
def clean_wine_name(value):
    # Repair text-encoding artifacts in display names without changing model inputs.
    for _ in range(2):
        try:
            repaired = value.encode("cp1252").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if repaired == value:
            break
        value = repaired
    return value


wine_names = np.asarray([
    clean_wine_name(value)
    for value in testing_data.iloc[:, 0].astype(str)
])
X_train = training_data[FEATURE_COLUMNS].to_numpy(dtype=int)
y_train = training_data[LABEL_COLUMN].to_numpy(dtype=int)
X_test = testing_data[FEATURE_COLUMNS].to_numpy(dtype=int)
y_test = testing_data[LABEL_COLUMN].to_numpy(dtype=int)

if not np.isin(X_train, [0, 1]).all() or not np.isin(X_test, [0, 1]).all():
    raise ValueError("The sensory features must contain only 0 and 1.")
if not np.isin(y_train, [0, 1]).all() or not np.isin(y_test, [0, 1]).all():
    raise ValueError("The grade labels must contain only 0 and 1.")

print(f"Training wines: {X_train.shape[0]}")
print(f"Testing wines: {X_test.shape[0]}")
print(f"Sensory features: {X_train.shape[1]}")
print(f"Class labels: 1 = 90+ points, 0 = below 90 points")

# %% [markdown]
# ## 2. Set up SVMlight
#
# This cell first checks for existing SVMlight commands. If they are missing, it downloads the official source and compiles `svm_learn` and `svm_classify` in a local folder. Building from source requires internet access, `make`, and a C compiler (GCC or Clang).

# %%
import tarfile
import urllib.request

SVM_LIGHT_SOURCE_URL = "https://osmot.cs.cornell.edu/svm_light/current/svm_light.tar.gz"
SVM_LIGHT_BUILD_DIR = Path("svmlight_build")


def executable_path(environment_variable, default_name):
    configured = os.environ.get(environment_variable, default_name)
    found = shutil.which(configured)
    if found:
        return found
    configured_path = Path(configured)
    if configured_path.is_file() and os.access(configured_path, os.X_OK):
        return str(configured_path.resolve())
    return None


svm_learn_path = executable_path("SVM_LEARN", "svm_learn")
svm_classify_path = executable_path("SVM_CLASSIFY", "svm_classify")

if not (svm_learn_path and svm_classify_path):
    SVM_LIGHT_BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse binaries from an earlier notebook run before downloading again.
    local_learn = next(
        (path for path in SVM_LIGHT_BUILD_DIR.rglob("svm_learn") if path.is_file() and os.access(path, os.X_OK)),
        None,
    )
    local_classify = next(
        (path for path in SVM_LIGHT_BUILD_DIR.rglob("svm_classify") if path.is_file() and os.access(path, os.X_OK)),
        None,
    )

    if local_learn and local_classify:
        svm_learn_path = str(local_learn.resolve())
        svm_classify_path = str(local_classify.resolve())
    else:
        make_path = shutil.which("make")
        compiler_path = shutil.which("gcc") or shutil.which("clang")
        if not make_path or not compiler_path:
            raise RuntimeError(
                "Building SVMlight needs make and a C compiler. "
                "On Ubuntu or Colab, install them with: "
                "apt-get update && apt-get install -y build-essential"
            )

        archive_path = SVM_LIGHT_BUILD_DIR / "svm_light.tar.gz"
        offline_archive = os.environ.get("SVM_LIGHT_SOURCE_ARCHIVE")
        if offline_archive:
            shutil.copy2(offline_archive, archive_path)
        else:
            try:
                urllib.request.urlretrieve(SVM_LIGHT_SOURCE_URL, archive_path)
            except Exception as error:
                raise RuntimeError(
                    "Could not download SVMlight source. Check internet access, "
                    "or set SVM_LIGHT_SOURCE_ARCHIVE to a local source tarball."
                ) from error

        with tarfile.open(archive_path, "r:gz") as source_archive:
            root = SVM_LIGHT_BUILD_DIR.resolve()
            for member in source_archive.getmembers():
                destination = (SVM_LIGHT_BUILD_DIR / member.name).resolve()
                if destination != root and root not in destination.parents:
                    raise ValueError("The SVMlight source archive contains an unsafe path.")
            source_archive.extractall(SVM_LIGHT_BUILD_DIR, filter="data")

        makefile = next(SVM_LIGHT_BUILD_DIR.rglob("Makefile"), None)
        if makefile is None:
            raise FileNotFoundError("The downloaded SVMlight source has no Makefile.")

        build = subprocess.run(
            [
                make_path,
                "-C",
                str(makefile.parent.resolve()),
                f"CC={Path(compiler_path).name}",
                "CFLAGS=-O3 -fcommon",
                "LFLAGS=-O3 -fcommon",
            ],
            capture_output=True,
            text=True,
        )
        if build.returncode != 0:
            raise RuntimeError(build.stderr or build.stdout)

        svm_learn_path = str((makefile.parent / "svm_learn").resolve())
        svm_classify_path = str((makefile.parent / "svm_classify").resolve())

if not (Path(svm_learn_path).is_file() and Path(svm_classify_path).is_file()):
    raise FileNotFoundError("SVMlight setup did not produce both executables.")

os.environ["SVM_LEARN"] = svm_learn_path
os.environ["SVM_CLASSIFY"] = svm_classify_path
print("SVMlight is ready.")
print("Both SVMlight commands are ready.")

# %% [markdown]
# ## 3. Train the SVMlight classifier
#
# SVMlight reads sparse feature rows. Feature indexes start at 1, and the grade labels are converted from 0/1 to -1/+1. The `-t 0` option selects a linear kernel.

# %%
OUTPUT_DIRECTORY = Path("svmlight_output")
OUTPUT_DIRECTORY.mkdir(exist_ok=True)

training_file = OUTPUT_DIRECTORY / "training.svmlight"
testing_file = OUTPUT_DIRECTORY / "testing.svmlight"
model_file = OUTPUT_DIRECTORY / "svm_light_model.txt"
scores_file = OUTPUT_DIRECTORY / "decision_scores.txt"


def write_svmlight_file(file_path, features, labels):
    with file_path.open("w", encoding="utf-8") as output:
        for row, label in zip(features, labels):
            svm_label = 1 if label == 1 else -1
            active_features = [
                f"{index}:{value}"
                for index, value in enumerate(row, start=1)
                if value != 0
            ]
            output.write(" ".join([str(svm_label), *active_features]) + "\n")


def find_svm_executable(environment_variable, default_name):
    configured_name = os.environ.get(environment_variable, default_name)
    executable = shutil.which(configured_name)
    if executable:
        return executable
    configured_path = Path(configured_name)
    if configured_path.is_file():
        return str(configured_path.resolve())
    raise FileNotFoundError(
        f"Could not find {default_name}. Install SVMlight or set "
        f"{environment_variable} to its executable path."
    )


write_svmlight_file(training_file, X_train, y_train)
write_svmlight_file(testing_file, X_test, y_test)

svm_learn = find_svm_executable("SVM_LEARN", "svm_learn")
svm_classify = find_svm_executable("SVM_CLASSIFY", "svm_classify")

training_run = subprocess.run(
    [svm_learn, "-t", "0", str(training_file), str(model_file)],
    capture_output=True,
    text=True,
)
if training_run.returncode != 0:
    raise RuntimeError(training_run.stderr or training_run.stdout)

example_tokens = training_file.read_text(encoding="utf-8").splitlines()[0].split()
print("Training command: svm_learn -t 0 training.svmlight svm_light_model.txt")
print(f"Training rows written: {len(X_train)}")
print("Example sparse row: " + " ".join(example_tokens[:9]) + (" ..." if len(example_tokens) > 9 else ""))
print(f"Linear SVM model saved: {model_file}")

# %% [markdown]
# ## 4. Predict and evaluate
#
# SVMlight writes one decision score per test wine. A score greater than or equal to 0 predicts class 1; a score below 0 predicts class 0.

# %%
classification_run = subprocess.run(
    [svm_classify, str(testing_file), str(model_file), str(scores_file)],
    capture_output=True,
    text=True,
)
if classification_run.returncode != 0:
    raise RuntimeError(classification_run.stderr or classification_run.stdout)

decision_scores = np.loadtxt(scores_file, dtype=float, ndmin=1)
if len(decision_scores) != len(y_test):
    raise ValueError("The number of SVMlight scores does not match the test rows.")

svm_predictions = (decision_scores >= 0).astype(int)
correct = svm_predictions == y_test
accuracy = float(correct.mean())

prediction_table = pd.DataFrame({
    "Wine": wine_names,
    "Actual grade": y_test,
    "Predicted grade": svm_predictions,
    "Correct": np.where(correct, "Yes", "No"),
})
print(prediction_table.to_string(index=False, max_colwidth=42))

true_positive = int(np.sum((y_test == 1) & (svm_predictions == 1)))
false_positive = int(np.sum((y_test == 0) & (svm_predictions == 1)))
true_negative = int(np.sum((y_test == 0) & (svm_predictions == 0)))
false_negative = int(np.sum((y_test == 1) & (svm_predictions == 0)))

print(f"\nCorrect predictions: {int(correct.sum())} / {len(y_test)}")
print(f"Prediction accuracy: {accuracy:.2%}")
print(f"TP: {true_positive}, FP: {false_positive}, TN: {true_negative}, FN: {false_negative}")

prediction_table.assign(
    Decision_score=decision_scores
).to_csv(OUTPUT_DIRECTORY / "svm_predictions.csv", index=False)
print(f"Predictions saved to: {OUTPUT_DIRECTORY / 'svm_predictions.csv'}")

# %% [markdown]
# ## 5. Compare with Part 2
#
# Part 2 reports five-fold cross-validation averages on 1,010 wines. This notebook evaluates the separate 200-wine training and 19-wine test split.

# %%
part2_accuracy = {
    "Naive Bayes": 87.62,
    "KNN (k=3)": 56.73,
}

comparison = pd.DataFrame([
    {
        "Algorithm": "Naive Bayes",
        "Accuracy": part2_accuracy["Naive Bayes"],
        "SVMlight advantage (pp)": round(accuracy * 100 - part2_accuracy["Naive Bayes"], 2),
        "Evaluation": "Part 2: 5-fold CV, 1,010 wines",
    },
    {
        "Algorithm": "KNN (k=3)",
        "Accuracy": part2_accuracy["KNN (k=3)"],
        "SVMlight advantage (pp)": round(accuracy * 100 - part2_accuracy["KNN (k=3)"], 2),
        "Evaluation": "Part 2: 5-fold CV, 1,010 wines",
    },
    {
        "Algorithm": "SVMlight (linear)",
        "Accuracy": round(accuracy * 100, 2),
        "SVMlight advantage (pp)": 0.00,
        "Evaluation": "Held-out: 200 train / 19 test",
    },
])
comparison["Accuracy"] = comparison["Accuracy"].map(lambda value: f"{value:.2f}%")
print(comparison.to_string(index=False))
print("\nThe Part 2 average table reports KNN accuracy as 56.73%.")

# %% [markdown]
# ## Conclusion
#
# The linear SVMlight model correctly predicted 18 of 19 test wines, for **94.74% accuracy**. Compared with Part 2's average table, this is **7.12 percentage points above Naive Bayes** (87.62%) and **38.01 points above KNN** (56.73%).
#
# The Part 2 results use five-fold cross-validation on 1,010 wines, while this notebook uses a separate 200/19 training and test split. Because the evaluation setups differ, the comparison is indicative rather than a direct like-for-like result.
