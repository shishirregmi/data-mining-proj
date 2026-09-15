"""Classify wines as 90+ (1) or below 90 (0) using three algorithms."""

# Import Path so dataset paths work from any current directory.
from pathlib import Path
# Import datetime so every output filename receives a unique timestamp.
from datetime import datetime
# Import os so Docker can choose a mounted output location.
import os
# Import NumPy for arrays, distances, probabilities, and reproducible shuffling.
import numpy as np
# Import pandas for reading Excel files and writing prediction tables.
import pandas as pd

# Store the folder containing this Python file.
PROJECT_FOLDER = Path(__file__).resolve().parent
# Store the training workbook path.
TRAINING_FILE = PROJECT_FOLDER / "Training dataset.xlsx"
# Store the testing workbook path.
TESTING_FILE = PROJECT_FOLDER / "Testing dataset.xlsx"
# Read an optional output directory from Docker or use the project folder by default.
OUTPUT_DIRECTORY = Path(os.environ.get("OUTPUT_DIRECTORY", PROJECT_FOLDER))
# Store the label column exactly as it appears in both workbooks.
LABEL_COLUMN = "Grade class 1: 90+  0:90-"
# Store the first column's position because its header is unnamed.
NAME_COLUMN_INDEX = 0
# Store a seed so cross-validation produces the same result every run.
RANDOM_SEED = 4370


# Define a function that loads and validates both datasets.
def load_datasets():
    # Read the first worksheet of the training workbook.
    training_data = pd.read_excel(TRAINING_FILE, sheet_name=0)
    # Read the first worksheet of the testing workbook.
    testing_data = pd.read_excel(TESTING_FILE, sheet_name=0)
    # Stop with a clear message if the expected label column is absent.
    if LABEL_COLUMN not in training_data.columns or LABEL_COLUMN not in testing_data.columns:
        # Raise an error that identifies the required label column.
        raise ValueError(f"Both datasets must contain '{LABEL_COLUMN}'.")
    # Stop if the two workbooks do not use the same columns in the same order.
    if training_data.columns.tolist() != testing_data.columns.tolist():
        # Raise an error because prediction requires matching features.
        raise ValueError("Training and testing columns do not match.")
    # Select every sensory feature after the name and label columns.
    feature_columns = training_data.columns[2:].tolist()
    # Convert the training sensory features to a numeric NumPy array.
    training_features = training_data[feature_columns].to_numpy(dtype=float)
    # Convert the training labels to a one-dimensional integer array.
    training_labels = training_data[LABEL_COLUMN].to_numpy(dtype=int)
    # Convert the testing sensory features to a numeric NumPy array.
    testing_features = testing_data[feature_columns].to_numpy(dtype=float)
    # Convert the hidden testing labels to a one-dimensional integer array.
    testing_labels = testing_data[LABEL_COLUMN].to_numpy(dtype=int)
    # Copy the wine names so they can be included in the output.
    wine_names = testing_data.iloc[:, NAME_COLUMN_INDEX].astype(str).to_numpy()
    # Verify that every feature value is binary as required by these datasets.
    if not np.isin(training_features, [0, 1]).all() or not np.isin(testing_features, [0, 1]).all():
        # Raise an error instead of silently using an unsuitable calculation.
        raise ValueError("All sensory feature values must be 0 or 1.")
    # Return all arrays required for training, prediction, and reporting.
    return training_features, training_labels, testing_features, testing_labels, wine_names


# Define a Bernoulli Naïve Bayes classifier for binary sensory features.
class BernoulliNaiveBayes:
    # Initialize the model with Laplace smoothing.
    def __init__(self, alpha=1.0):
        # Save the positive smoothing value used to avoid zero probabilities.
        self.alpha = alpha
        # Prepare a place for the two class labels learned during training.
        self.classes = None
        # Prepare a place for each class's prior probability.
        self.class_log_prior = None
        # Prepare a place for P(feature=1 | class).
        self.feature_probability = None

    # Train the classifier from binary features and known labels.
    def fit(self, features, labels):
        # Find the unique class labels in sorted order.
        self.classes = np.unique(labels)
        # Create a list that will hold each class's log prior.
        prior_values = []
        # Create a list that will hold each class's feature probabilities.
        probability_values = []
        # Process one class at a time.
        for class_value in self.classes:
            # Select the training rows that belong to the current class.
            class_features = features[labels == class_value]
            # Calculate and save the natural logarithm of the class prior.
            prior_values.append(np.log(len(class_features) / len(features)))
            # Count how many wines in this class contain each sensory word.
            feature_counts = class_features.sum(axis=0)
            # Apply Laplace smoothing to each Bernoulli probability.
            probabilities = (feature_counts + self.alpha) / (len(class_features) + 2 * self.alpha)
            # Save the current class's smoothed feature probabilities.
            probability_values.append(probabilities)
        # Convert the prior list into a NumPy array.
        self.class_log_prior = np.asarray(prior_values)
        # Stack the feature probabilities into one row per class.
        self.feature_probability = np.vstack(probability_values)
        # Return this trained object to support method chaining.
        return self

    # Predict a class label for every supplied wine.
    def predict(self, features):
        # Calculate log probabilities for features that are present.
        present_scores = features @ np.log(self.feature_probability).T
        # Calculate log probabilities for features that are absent.
        absent_scores = (1 - features) @ np.log(1 - self.feature_probability).T
        # Add feature evidence and class priors for each possible class.
        total_scores = present_scores + absent_scores + self.class_log_prior
        # Select the class with the largest posterior log score for each wine.
        return self.classes[np.argmax(total_scores, axis=1)]


# Define a K-Nearest Neighbors classifier using Euclidean distance.
class KNearestNeighbors:
    # Initialize the model with the requested number of neighbors.
    def __init__(self, number_of_neighbors=5):
        # Save k so prediction knows how many neighbors to use.
        self.number_of_neighbors = number_of_neighbors
        # Prepare a place for the training features.
        self.training_features = None
        # Prepare a place for the training labels.
        self.training_labels = None

    # Store the labeled examples because KNN learns by remembering them.
    def fit(self, features, labels):
        # Save the training feature matrix.
        self.training_features = features
        # Save the matching training labels.
        self.training_labels = labels
        # Return this trained object to support method chaining.
        return self

    # Predict a class label for every supplied wine.
    def predict(self, features):
        # Create a list for the predicted labels.
        predictions = []
        # Process one testing wine at a time.
        for testing_wine in features:
            # Calculate Euclidean distance from this wine to every training wine.
            distances = np.sqrt(np.sum((self.training_features - testing_wine) ** 2, axis=1))
            # Find the indexes of the k smallest distances.
            neighbor_indexes = np.argsort(distances)[: self.number_of_neighbors]
            # Retrieve the known labels of those nearest wines.
            neighbor_labels = self.training_labels[neighbor_indexes]
            # Count votes for label 0 and label 1.
            vote_counts = np.bincount(neighbor_labels, minlength=2)
            # Choose the label with the most votes and append it to the result.
            predictions.append(int(np.argmax(vote_counts)))
        # Convert the completed prediction list to a NumPy array.
        return np.asarray(predictions)


# Define one node used by the binary decision tree.
class DecisionTreeNode:
    # Initialize either a decision node or a leaf node.
    def __init__(self, feature_index=None, left_child=None, right_child=None, predicted_class=None):
        # Store the sensory feature tested at this node.
        self.feature_index = feature_index
        # Store the child followed when the feature value is zero.
        self.left_child = left_child
        # Store the child followed when the feature value is one.
        self.right_child = right_child
        # Store the predicted class when this node is a leaf.
        self.predicted_class = predicted_class


# Define a binary decision tree classifier using Gini impurity.
class DecisionTreeClassifier:
    # Initialize the tree with controls that prevent overfitting.
    def __init__(self, maximum_depth=4, minimum_samples_split=2):
        # Save the greatest number of decisions allowed from root to leaf.
        self.maximum_depth = maximum_depth
        # Save the fewest rows required before a node may split.
        self.minimum_samples_split = minimum_samples_split
        # Prepare a place for the root node learned during training.
        self.root = None

    # Calculate the Gini impurity of a group of labels.
    def _gini_impurity(self, labels):
        # Return zero impurity for an empty group.
        if len(labels) == 0:
            # Use zero because an empty child contributes no weighted impurity.
            return 0.0
        # Count the number of rows belonging to each binary class.
        class_counts = np.bincount(labels, minlength=2)
        # Convert the class counts to class proportions.
        class_proportions = class_counts / len(labels)
        # Subtract the sum of squared proportions from one.
        return float(1.0 - np.sum(class_proportions**2))

    # Find the feature that produces the largest reduction in Gini impurity.
    def _find_best_feature(self, features, labels):
        # Calculate impurity before performing any split.
        parent_impurity = self._gini_impurity(labels)
        # Start with no selected feature.
        best_feature_index = None
        # Start with no impurity improvement.
        best_gain = 0.0
        # Examine every sensory feature as a possible decision.
        for feature_index in range(features.shape[1]):
            # Mark rows where the current binary feature is absent.
            left_mask = features[:, feature_index] == 0
            # Mark rows where the current binary feature is present.
            right_mask = features[:, feature_index] == 1
            # Skip a feature that would create an empty child.
            if not left_mask.any() or not right_mask.any():
                # Continue directly to the next sensory feature.
                continue
            # Calculate the fraction of rows sent to the left child.
            left_weight = float(left_mask.mean())
            # Calculate the fraction of rows sent to the right child.
            right_weight = float(right_mask.mean())
            # Calculate the weighted impurity after this possible split.
            child_impurity = left_weight * self._gini_impurity(labels[left_mask]) + right_weight * self._gini_impurity(labels[right_mask])
            # Calculate how much this split reduces impurity.
            information_gain = parent_impurity - child_impurity
            # Keep the feature when it improves upon the best split found so far.
            if information_gain > best_gain:
                # Save the improved Gini gain.
                best_gain = information_gain
                # Save the feature responsible for the improvement.
                best_feature_index = feature_index
        # Return the best feature and its impurity reduction.
        return best_feature_index, best_gain

    # Recursively construct a tree from the supplied training subset.
    def _build_tree(self, features, labels, current_depth):
        # Count the rows belonging to each class.
        class_counts = np.bincount(labels, minlength=2)
        # Select the majority class for a possible leaf prediction.
        majority_class = int(np.argmax(class_counts))
        # Stop when every row has the same label.
        labels_are_pure = len(np.unique(labels)) == 1
        # Stop when the configured depth limit has been reached.
        depth_limit_reached = current_depth >= self.maximum_depth
        # Stop when too few rows remain to make a reliable split.
        too_few_samples = len(labels) < self.minimum_samples_split
        # Create a leaf immediately when any stopping condition is true.
        if labels_are_pure or depth_limit_reached or too_few_samples:
            # Return a leaf that predicts the current majority class.
            return DecisionTreeNode(predicted_class=majority_class)
        # Search for the strongest available binary feature split.
        feature_index, information_gain = self._find_best_feature(features, labels)
        # Stop when no feature can reduce impurity.
        if feature_index is None or information_gain <= 0.0:
            # Return a leaf that predicts the current majority class.
            return DecisionTreeNode(predicted_class=majority_class)
        # Select rows where the chosen feature is absent.
        left_mask = features[:, feature_index] == 0
        # Select rows where the chosen feature is present.
        right_mask = features[:, feature_index] == 1
        # Build the absent-feature branch one level deeper.
        left_child = self._build_tree(features[left_mask], labels[left_mask], current_depth + 1)
        # Build the present-feature branch one level deeper.
        right_child = self._build_tree(features[right_mask], labels[right_mask], current_depth + 1)
        # Return a decision node connected to both completed children.
        return DecisionTreeNode(feature_index, left_child, right_child)

    # Train the decision tree from binary features and known labels.
    def fit(self, features, labels):
        # Build the complete tree beginning at depth zero.
        self.root = self._build_tree(features, labels, current_depth=0)
        # Return this trained object to support method chaining.
        return self

    # Follow the tree to predict one wine's class.
    def _predict_one(self, wine_features):
        # Begin traversal at the root node.
        current_node = self.root
        # Continue until traversal reaches a leaf prediction.
        while current_node.predicted_class is None:
            # Follow the left child when the selected sensory word is absent.
            if wine_features[current_node.feature_index] == 0:
                # Move to the absent-feature child.
                current_node = current_node.left_child
            # Follow the right child when the selected sensory word is present.
            else:
                # Move to the present-feature child.
                current_node = current_node.right_child
        # Return the class stored in the reached leaf.
        return current_node.predicted_class

    # Predict a class label for every supplied wine.
    def predict(self, features):
        # Predict each row independently and return an integer array.
        return np.asarray([self._predict_one(wine) for wine in features], dtype=int)


# Define a function that creates reproducible stratified folds.
def make_stratified_folds(labels, fold_count=5):
    # Create a reproducible random-number generator.
    generator = np.random.default_rng(RANDOM_SEED)
    # Create an empty list of indexes for each fold.
    folds = [[] for _ in range(fold_count)]
    # Split each class separately so every fold remains balanced.
    for class_value in np.unique(labels):
        # Find all row indexes belonging to the current class.
        class_indexes = np.where(labels == class_value)[0]
        # Shuffle a copy of the indexes to avoid order bias.
        shuffled_indexes = generator.permutation(class_indexes)
        # Divide the shuffled indexes as evenly as possible among the folds.
        class_parts = np.array_split(shuffled_indexes, fold_count)
        # Add each class part to its corresponding fold.
        for fold_index, class_part in enumerate(class_parts):
            # Extend the current fold with the current class's indexes.
            folds[fold_index].extend(class_part.tolist())
    # Convert every completed fold to an integer NumPy array.
    return [np.asarray(fold, dtype=int) for fold in folds]


# Define a function that selects k without examining testing labels.
def select_best_k(features, labels, candidate_values=(3, 5, 7, 9, 11)):
    # Build five balanced validation folds from the training labels.
    folds = make_stratified_folds(labels)
    # Create a dictionary that will store mean validation accuracy for each k.
    scores = {}
    # Evaluate every candidate neighbor count.
    for candidate_k in candidate_values:
        # Create a list for this candidate's five accuracy values.
        fold_scores = []
        # Use each fold once for validation.
        for validation_indexes in folds:
            # Mark all rows as available for training initially.
            training_mask = np.ones(len(labels), dtype=bool)
            # Remove the current validation rows from the training subset.
            training_mask[validation_indexes] = False
            # Train KNN using only the current training subset.
            model = KNearestNeighbors(candidate_k).fit(features[training_mask], labels[training_mask])
            # Predict labels for the held-out validation subset.
            validation_predictions = model.predict(features[validation_indexes])
            # Calculate and save the current fold's accuracy.
            fold_scores.append(float(np.mean(validation_predictions == labels[validation_indexes])))
        # Save the candidate's mean accuracy across all folds.
        scores[candidate_k] = float(np.mean(fold_scores))
    # Select the highest-scoring k and prefer the smaller k when scores tie.
    best_k = max(scores, key=lambda value: (scores[value], -value))
    # Return both the chosen k and all validation scores for transparency.
    return best_k, scores


# Define a function that selects tree depth without examining testing labels.
def select_best_tree_depth(features, labels, candidate_depths=(2, 3, 4, 5, 6)):
    # Build five balanced validation folds from the training labels.
    folds = make_stratified_folds(labels)
    # Create a dictionary that will store mean validation accuracy for each depth.
    scores = {}
    # Evaluate every candidate maximum depth.
    for candidate_depth in candidate_depths:
        # Create a list for this candidate's five accuracy values.
        fold_scores = []
        # Use each fold once for validation.
        for validation_indexes in folds:
            # Mark all rows as available for training initially.
            training_mask = np.ones(len(labels), dtype=bool)
            # Remove the current validation rows from the training subset.
            training_mask[validation_indexes] = False
            # Train a tree using only the current training subset.
            model = DecisionTreeClassifier(candidate_depth).fit(features[training_mask], labels[training_mask])
            # Predict labels for the held-out validation subset.
            validation_predictions = model.predict(features[validation_indexes])
            # Calculate and save the current fold's accuracy.
            fold_scores.append(float(np.mean(validation_predictions == labels[validation_indexes])))
        # Save the candidate's mean accuracy across all folds.
        scores[candidate_depth] = float(np.mean(fold_scores))
    # Select the highest-scoring depth and prefer the shallower tree when scores tie.
    best_depth = max(scores, key=lambda value: (scores[value], -value))
    # Return both the chosen depth and all validation scores for transparency.
    return best_depth, scores


# Define a small function for computing prediction accuracy.
def calculate_accuracy(real_labels, predicted_labels):
    # Count correct predictions and divide by the total number of wines.
    return float(np.mean(real_labels == predicted_labels))


# Define the complete training, prediction, evaluation, and export workflow.
def main():
    # Create a timestamp in day-month-year and hour-minute-second order.
    output_timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    # Build a unique CSV filename from the timestamp.
    output_file = OUTPUT_DIRECTORY / f"prediction_results_{output_timestamp}.csv"
    # Load the training and testing data from the supplied workbooks.
    train_x, train_y, test_x, test_y, wine_names = load_datasets()
    # Select k using only the training dataset.
    best_k, validation_scores = select_best_k(train_x, train_y)
    # Select maximum tree depth using only the training dataset.
    best_depth, tree_validation_scores = select_best_tree_depth(train_x, train_y)
    # Train Bernoulli Naïve Bayes on every training wine.
    naive_bayes_model = BernoulliNaiveBayes().fit(train_x, train_y)
    # Train KNN with the selected k on every training wine.
    knn_model = KNearestNeighbors(best_k).fit(train_x, train_y)
    # Train the decision tree with the selected depth on every training wine.
    decision_tree_model = DecisionTreeClassifier(best_depth).fit(train_x, train_y)
    # Predict the testing labels with Naïve Bayes.
    naive_bayes_predictions = naive_bayes_model.predict(test_x)
    # Predict the testing labels with KNN.
    knn_predictions = knn_model.predict(test_x)
    # Predict the testing labels with the decision tree.
    decision_tree_predictions = decision_tree_model.predict(test_x)
    # Calculate Naïve Bayes accuracy using the true testing labels.
    naive_bayes_accuracy = calculate_accuracy(test_y, naive_bayes_predictions)
    # Calculate KNN accuracy using the true testing labels.
    knn_accuracy = calculate_accuracy(test_y, knn_predictions)
    # Calculate decision-tree accuracy using the true testing labels.
    decision_tree_accuracy = calculate_accuracy(test_y, decision_tree_predictions)
    # Build the professor-requested real-versus-predicted result table.
    result_table = pd.DataFrame({
        # Include each wine name so incorrect predictions can be investigated.
        "Wine": wine_names,
        # Include the true grade label from the testing workbook.
        "Real grade": test_y,
        # Include the Naïve Bayes predicted grade.
        "Naive Bayes predicted grade": naive_bayes_predictions,
        # Include the KNN predicted grade.
        "KNN predicted grade": knn_predictions,
        # Include the decision tree predicted grade.
        "Decision Tree predicted grade": decision_tree_predictions,
    })
    # Create the output folder when Docker writes into a mounted directory.
    output_file.parent.mkdir(parents=True, exist_ok=True)
    # Save the complete table as a CSV file for submission and discussion.
    result_table.to_csv(output_file, index=False)
    # Print the dataset sizes for a quick correctness check.
    print(f"Training wines: {len(train_y)} | Testing wines: {len(test_y)}")
    # Print every cross-validation score used to choose k.
    print("KNN validation accuracy by k:", {k: f"{score:.2%}" for k, score in validation_scores.items()})
    # Print the selected neighbor count.
    print(f"Selected KNN k: {best_k}")
    # Print every cross-validation score used to choose tree depth.
    print("Decision Tree validation accuracy by depth:", {depth: f"{score:.2%}" for depth, score in tree_validation_scores.items()})
    # Print the selected maximum tree depth.
    print(f"Selected Decision Tree maximum depth: {best_depth}")
    # Print Naïve Bayes's number correct, total, and accuracy.
    print(f"Naive Bayes: {(naive_bayes_predictions == test_y).sum()}/{len(test_y)} correct = {naive_bayes_accuracy:.2%}")
    # Print KNN's number correct, total, and accuracy.
    print(f"KNN: {(knn_predictions == test_y).sum()}/{len(test_y)} correct = {knn_accuracy:.2%}")
    # Print the decision tree's number correct, total, and accuracy.
    print(f"Decision Tree: {(decision_tree_predictions == test_y).sum()}/{len(test_y)} correct = {decision_tree_accuracy:.2%}")
    # Print the exact real-versus-predicted columns requested by the professor.
    print(result_table[["Real grade", "Naive Bayes predicted grade", "KNN predicted grade", "Decision Tree predicted grade"]].to_string(index=False))
    # Print the output location so the user can find the saved table.
    print(f"\nSaved predictions to: {output_file}")


# Run the workflow only when this file is executed directly.
if __name__ == "__main__":
    # Call the program's main function.
    main()
