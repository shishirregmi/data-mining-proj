"""Classify wines as 90+ (1) or below 90 (0) using three algorithms."""

from pathlib import Path
from datetime import datetime
import os
import numpy as np
import pandas as pd

PROJECT_FOLDER = Path(__file__).resolve().parent
TRAINING_FILE = PROJECT_FOLDER / "Training dataset.xlsx"
TESTING_FILE = PROJECT_FOLDER / "Testing dataset.xlsx"
OUTPUT_DIRECTORY = Path(os.environ.get("OUTPUT_DIRECTORY", PROJECT_FOLDER))
LABEL_COLUMN = "Grade class 1: 90+  0:90-"
NAME_COLUMN_INDEX = 0
RANDOM_SEED = 4370


def load_datasets():
    training_data = pd.read_excel(TRAINING_FILE, sheet_name=0)
    testing_data = pd.read_excel(TESTING_FILE, sheet_name=0)
    if LABEL_COLUMN not in training_data.columns or LABEL_COLUMN not in testing_data.columns:
        raise ValueError(f"Both datasets must contain '{LABEL_COLUMN}'.")
    if training_data.columns.tolist() != testing_data.columns.tolist():
        raise ValueError("Training and testing columns do not match.")
    feature_columns = training_data.columns[2:].tolist()
    training_features = training_data[feature_columns].to_numpy(dtype=float)
    training_labels = training_data[LABEL_COLUMN].to_numpy(dtype=int)
    testing_features = testing_data[feature_columns].to_numpy(dtype=float)
    testing_labels = testing_data[LABEL_COLUMN].to_numpy(dtype=int)
    wine_names = testing_data.iloc[:, NAME_COLUMN_INDEX].astype(str).to_numpy()
    if not np.isin(training_features, [0, 1]).all() or not np.isin(testing_features, [0, 1]).all():
        raise ValueError("All sensory feature values must be 0 or 1.")
    return training_features, training_labels, testing_features, testing_labels, wine_names


class BernoulliNaiveBayes:
    """Classify binary review features with Bernoulli probabilities."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_log_prior = None
        self.feature_probability = None

    def fit(self, features, labels):
        self.classes = np.unique(labels)
        prior_values = []
        probability_values = []
        for class_value in self.classes:
            class_features = features[labels == class_value]
            # The prior represents how common this grade is in the training data.
            prior_values.append(np.log(len(class_features) / len(features)))
            feature_counts = class_features.sum(axis=0)
            # Laplace smoothing prevents unseen sensory words from creating zero probabilities.
            probabilities = (feature_counts + self.alpha) / (len(class_features) + 2 * self.alpha)
            probability_values.append(probabilities)
        self.class_log_prior = np.asarray(prior_values)
        self.feature_probability = np.vstack(probability_values)
        return self

    def predict(self, features):
        # Bernoulli NB uses evidence from both present and absent sensory words.
        present_scores = features @ np.log(self.feature_probability).T
        absent_scores = (1 - features) @ np.log(1 - self.feature_probability).T
        total_scores = present_scores + absent_scores + self.class_log_prior
        return self.classes[np.argmax(total_scores, axis=1)]


class KNearestNeighbors:
    """Classify a wine by majority vote among its nearest training wines."""

    def __init__(self, number_of_neighbors=5):
        self.number_of_neighbors = number_of_neighbors
        self.training_features = None
        self.training_labels = None

    def fit(self, features, labels):
        self.training_features = features
        self.training_labels = labels
        return self

    def predict(self, features):
        predictions = []
        for testing_wine in features:
            # Euclidean distance measures similarity across all binary sensory features.
            distances = np.sqrt(np.sum((self.training_features - testing_wine) ** 2, axis=1))
            neighbor_indexes = np.argsort(distances)[: self.number_of_neighbors]
            neighbor_labels = self.training_labels[neighbor_indexes]
            # The class receiving the most votes from the k nearest wines is predicted.
            vote_counts = np.bincount(neighbor_labels, minlength=2)
            predictions.append(int(np.argmax(vote_counts)))
        return np.asarray(predictions)


class DecisionTreeNode:
    """Represent either a binary feature decision or a predicted-class leaf."""

    def __init__(self, feature_index=None, left_child=None, right_child=None, predicted_class=None):
        self.feature_index = feature_index
        self.left_child = left_child
        self.right_child = right_child
        self.predicted_class = predicted_class


class DecisionTreeClassifier:
    """Build binary feature splits that maximize reduction in Gini impurity."""

    def __init__(self, maximum_depth=4, minimum_samples_split=2):
        self.maximum_depth = maximum_depth
        self.minimum_samples_split = minimum_samples_split
        self.root = None

    def _gini_impurity(self, labels):
        if len(labels) == 0:
            return 0.0
        class_counts = np.bincount(labels, minlength=2)
        class_proportions = class_counts / len(labels)
        # Impurity is zero when every wine at a node belongs to the same class.
        return float(1.0 - np.sum(class_proportions**2))

    def _find_best_feature(self, features, labels):
        parent_impurity = self._gini_impurity(labels)
        best_feature_index = None
        best_gain = 0.0
        for feature_index in range(features.shape[1]):
            left_mask = features[:, feature_index] == 0
            right_mask = features[:, feature_index] == 1
            if not left_mask.any() or not right_mask.any():
                continue
            left_weight = float(left_mask.mean())
            right_weight = float(right_mask.mean())
            child_impurity = left_weight * self._gini_impurity(labels[left_mask]) + right_weight * self._gini_impurity(labels[right_mask])
            # Information gain measures how much purer the two child groups become.
            information_gain = parent_impurity - child_impurity
            if information_gain > best_gain:
                best_gain = information_gain
                best_feature_index = feature_index
        return best_feature_index, best_gain

    def _build_tree(self, features, labels, current_depth):
        class_counts = np.bincount(labels, minlength=2)
        majority_class = int(np.argmax(class_counts))
        labels_are_pure = len(np.unique(labels)) == 1
        depth_limit_reached = current_depth >= self.maximum_depth
        too_few_samples = len(labels) < self.minimum_samples_split
        # A leaf predicts the majority class when further splitting should stop.
        if labels_are_pure or depth_limit_reached or too_few_samples:
            return DecisionTreeNode(predicted_class=majority_class)
        feature_index, information_gain = self._find_best_feature(features, labels)
        if feature_index is None or information_gain <= 0.0:
            return DecisionTreeNode(predicted_class=majority_class)
        left_mask = features[:, feature_index] == 0
        right_mask = features[:, feature_index] == 1
        # Each branch repeats the split search using only wines that reached that branch.
        left_child = self._build_tree(features[left_mask], labels[left_mask], current_depth + 1)
        right_child = self._build_tree(features[right_mask], labels[right_mask], current_depth + 1)
        return DecisionTreeNode(feature_index, left_child, right_child)

    def fit(self, features, labels):
        self.root = self._build_tree(features, labels, current_depth=0)
        return self

    def _predict_one(self, wine_features):
        current_node = self.root
        while current_node.predicted_class is None:
            if wine_features[current_node.feature_index] == 0:
                current_node = current_node.left_child
            else:
                current_node = current_node.right_child
        return current_node.predicted_class

    def predict(self, features):
        return np.asarray([self._predict_one(wine) for wine in features], dtype=int)


def make_stratified_folds(labels, fold_count=5):
    """Create reproducible folds that preserve the balance of both grade classes."""

    generator = np.random.default_rng(RANDOM_SEED)
    folds = [[] for _ in range(fold_count)]
    for class_value in np.unique(labels):
        class_indexes = np.where(labels == class_value)[0]
        shuffled_indexes = generator.permutation(class_indexes)
        class_parts = np.array_split(shuffled_indexes, fold_count)
        for fold_index, class_part in enumerate(class_parts):
            folds[fold_index].extend(class_part.tolist())
    return [np.asarray(fold, dtype=int) for fold in folds]


def select_best_k(features, labels, candidate_values=(3, 5, 7, 9, 11)):
    """Choose k using training-only cross-validation to avoid test-data leakage."""

    folds = make_stratified_folds(labels)
    scores = {}
    for candidate_k in candidate_values:
        fold_scores = []
        for validation_indexes in folds:
            training_mask = np.ones(len(labels), dtype=bool)
            training_mask[validation_indexes] = False
            model = KNearestNeighbors(candidate_k).fit(features[training_mask], labels[training_mask])
            validation_predictions = model.predict(features[validation_indexes])
            fold_scores.append(float(np.mean(validation_predictions == labels[validation_indexes])))
        scores[candidate_k] = float(np.mean(fold_scores))
    best_k = max(scores, key=lambda value: (scores[value], -value))
    return best_k, scores


def select_best_tree_depth(features, labels, candidate_depths=(2, 3, 4, 5, 6)):
    """Choose tree depth using training-only cross-validation."""

    folds = make_stratified_folds(labels)
    scores = {}
    for candidate_depth in candidate_depths:
        fold_scores = []
        for validation_indexes in folds:
            training_mask = np.ones(len(labels), dtype=bool)
            training_mask[validation_indexes] = False
            model = DecisionTreeClassifier(candidate_depth).fit(features[training_mask], labels[training_mask])
            validation_predictions = model.predict(features[validation_indexes])
            fold_scores.append(float(np.mean(validation_predictions == labels[validation_indexes])))
        scores[candidate_depth] = float(np.mean(fold_scores))
    best_depth = max(scores, key=lambda value: (scores[value], -value))
    return best_depth, scores


def calculate_accuracy(real_labels, predicted_labels):
    """Return correct predictions divided by total testing wines."""

    return float(np.mean(real_labels == predicted_labels))


def main():
    output_timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    output_file = OUTPUT_DIRECTORY / f"prediction_results_{output_timestamp}.csv"
    train_x, train_y, test_x, test_y, wine_names = load_datasets()
    best_k, validation_scores = select_best_k(train_x, train_y)
    best_depth, tree_validation_scores = select_best_tree_depth(train_x, train_y)
    naive_bayes_model = BernoulliNaiveBayes().fit(train_x, train_y)
    knn_model = KNearestNeighbors(best_k).fit(train_x, train_y)
    decision_tree_model = DecisionTreeClassifier(best_depth).fit(train_x, train_y)
    naive_bayes_predictions = naive_bayes_model.predict(test_x)
    knn_predictions = knn_model.predict(test_x)
    decision_tree_predictions = decision_tree_model.predict(test_x)
    naive_bayes_accuracy = calculate_accuracy(test_y, naive_bayes_predictions)
    knn_accuracy = calculate_accuracy(test_y, knn_predictions)
    decision_tree_accuracy = calculate_accuracy(test_y, decision_tree_predictions)
    result_table = pd.DataFrame({
        "Wine": wine_names,
        "Real grade": test_y,
        "Naive Bayes predicted grade": naive_bayes_predictions,
        "KNN predicted grade": knn_predictions,
        "Decision Tree predicted grade": decision_tree_predictions,
    })
    output_file.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(output_file, index=False)
    print(f"Training wines: {len(train_y)} | Testing wines: {len(test_y)}")
    print("KNN validation accuracy by k:", {k: f"{score:.2%}" for k, score in validation_scores.items()})
    print(f"Selected KNN k: {best_k}")
    print("Decision Tree validation accuracy by depth:", {depth: f"{score:.2%}" for depth, score in tree_validation_scores.items()})
    print(f"Selected Decision Tree maximum depth: {best_depth}")
    print(f"Naive Bayes: {(naive_bayes_predictions == test_y).sum()}/{len(test_y)} correct = {naive_bayes_accuracy:.2%}")
    print(f"KNN: {(knn_predictions == test_y).sum()}/{len(test_y)} correct = {knn_accuracy:.2%}")
    print(f"Decision Tree: {(decision_tree_predictions == test_y).sum()}/{len(test_y)} correct = {decision_tree_accuracy:.2%}")
    print(result_table[["Real grade", "Naive Bayes predicted grade", "KNN predicted grade", "Decision Tree predicted grade"]].to_string(index=False))
    print(f"\nSaved predictions to: {output_file}")


if __name__ == "__main__":
    main()
