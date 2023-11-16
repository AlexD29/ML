import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Preprocessing

df1 = pd.read_csv("WFOvsWFH.csv")
discrete_attributes = [
    "Age",
    "Gender",
    "Occupation",
    "Same_ofiice_home_location",
    "kids",
    "RM_save_money",
    "RM_quality_time",
    "RM_better_sleep",
    "calmer_stressed",
    "digital_connect_sufficient",
    "RM_job_opportunities",
    "Target"
]
continuous_attributes = [
    "RM_professional_growth",
    "RM_lazy",
    "RM_productive",
    "RM_better_work_life_balance",
    "RM_improved_skillset"
]
discrete_attributes_without_target = [
    "Age",
    "Gender",
    "Occupation",
    "Same_ofiice_home_location",
    "kids",
    "RM_save_money",
    "RM_quality_time",
    "RM_better_sleep",
    "calmer_stressed",
    "digital_connect_sufficient",
    "RM_job_opportunities"
]
attributes_to_analyze = [
    "Age",
    "Gender",
    "Occupation",
    "Same_ofiice_home_location",
    "kids",
    "RM_save_money",
    "RM_quality_time",
    "RM_better_sleep",
    "calmer_stressed",
    "digital_connect_sufficient",
    "RM_job_opportunities",
    "RM_professional_growth",
    "RM_lazy",
    "RM_productive",
    "RM_better_work_life_balance",
    "RM_improved_skillset"
]
target_attribute = "Target"


# a.
def remove_nan_values_from_csv(df, output_csv_file_path):
    nan_values = df.isna()
    df = df.dropna()
    df.to_csv(output_csv_file_path, index=False)

#remove_nan_values_from_csv(df1,"WFOvsWFHCleaned.csv")

# b.
df = pd.read_csv("WFOvsWFHCleaned.csv")

def calculate_mean_and_variance(df, continuous_attributes):
    mean_and_variance = {}
    for attribute in continuous_attributes:
        mean = df[attribute].mean()
        variance = df[attribute].var()
        mean_and_variance[attribute] = {
            "mean": round(mean, 2),
            "variance": round(variance, 2)
        }
    return mean_and_variance

# mean_and_variance = calculate_mean_and_variance(df, continuous_attributes)
# print(mean_and_variance)


# 2. Probabilities, Information Theory

# a.
def compute_probabilities(df, attribute):
    probabilities = {}

    for value in df[attribute].unique():
        probability = df[attribute].value_counts()[value] / len(df)
        probabilities[value] = probability

    return probabilities


# for attribute in discrete_attributes:
#   probabilities = compute_probabilities(df, attribute)
#   print(f"Probability mass function of {attribute}:")
#   print(probabilities)


# b.
def calculate_entropy(probabilities):
    entropy = 0.0

    for value, probability in probabilities.items():
        entropy += -probability * math.log2(probability)

    return entropy


def calculate_entropy_for_discrete_attributes(df):
    entropy_dict = {}

    for attribute in discrete_attributes:
        probabilities = compute_probabilities(df, attribute)
        entropy = calculate_entropy(probabilities)

        entropy_dict[attribute] = entropy

    return entropy_dict


# entropy_dict = calculate_entropy_for_discrete_attributes(df)
# print("Entropy for each discrete attribute:")
# print(entropy_dict)


# c.
def calculate_conditional_entropy(df, target_attribute, given_attribute):
    # P(Y|X)
    conditional_probabilities = df.groupby(given_attribute)[target_attribute].value_counts(normalize=True).unstack()

    # H(Y|X)
    prob_x = df[given_attribute].value_counts(normalize=True)
    prob_y_given_x = conditional_probabilities.apply(lambda row: row * np.log2(row)).sum(axis=1)

    conditional_entropy = -(prob_x * prob_y_given_x).sum()

    return conditional_entropy

# conditional_entropy = calculate_conditional_entropy(df, target_attribute, "calmer_stressed")
# print(f"Conditional Entropy: {conditional_entropy}")

# d.
def calculate_information_gain(df, target_attribute, given_attribute):
    entropy_before_split = calculate_entropy(compute_probabilities(df, target_attribute))
    conditional_entropy = calculate_conditional_entropy(df, target_attribute, given_attribute)
    information_gain = entropy_before_split - conditional_entropy
    return information_gain


# information_gain = calculate_information_gain(df, "Target", "calmer_stressed")
# print(f"Information Gain: {information_gain}")


# 3. ID3

# a.
def find_root_node(df, target_attribute, attributes):
    max_information_gain = -1
    best_attribute = None

    for attribute in attributes:
        information_gain = calculate_information_gain(df, target_attribute, attribute)

        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attribute = attribute

    return (best_attribute, max_information_gain)


# root_node = find_root_node(df, target_attribute, attributes_to_analyze)
# print(f"The root node is: {root_node[0]} with Information Gain: {root_node[1]}")


# b.
def id3_discrete(df, target_attribute, attributes):
    if df[target_attribute].nunique() == 1:
        return {target_attribute: df[target_attribute].iloc[0]}

    if not attributes:
        majority_class = df[target_attribute].mode().iloc[0]
        return {target_attribute: majority_class}

    best_attribute, max_information_gain = find_root_node(df, target_attribute, attributes)

    node = {
        best_attribute: {
            "n_observations": df[target_attribute].count(),
            "information_gain": max_information_gain,
            "values": {}
        }
    }

    for value in df[best_attribute].unique():
        subset = df[df[best_attribute] == value].copy()
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]
        subtree = id3_discrete(subset, target_attribute, remaining_attributes)
        node[best_attribute]["values"][value] = subtree

    return node


# tree = id3_discrete(df, target_attribute, discrete_attributes_without_target)
# print(tree)


# c.
# label_encoder = LabelEncoder()
# for column in discrete_attributes_without_target:
#     if column in df.columns:
#         df[column] = label_encoder.fit_transform(df[column])
#
# X = df[discrete_attributes_without_target]
# y = df[target_attribute]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# sklearn_tree = DecisionTreeClassifier(random_state=42)
# sklearn_tree.fit(X_train, y_train)
#
# y_pred_sklearn = sklearn_tree.predict(X_test)
#
# accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
# print(f"Accuracy using scikit-learn DecisionTreeClassifier: {accuracy_sklearn:.2f}")


# tree = id3_discrete(df, target_attribute, discrete_attributes_without_target)
def predict_id3(tree, instances):
    predictions = []

    for _, instance in instances.iterrows():
        prediction = predict_single_id3(tree, instance)
        predictions.append(prediction)

    return predictions


def predict_single_id3(tree, instance, default_prediction="Unknown"):
    if isinstance(tree, dict):
        for attribute, subtree in tree.items():
            value = instance.get(attribute, None)
            if value is not None and value in subtree.get("values", {}):
                return predict_single_id3(subtree["values"][value], instance, default_prediction)

        # Handle missing values or unseen values in the tree
        return default_prediction
    else:
        # Leaf node
        return tree

# test_instances = pd.DataFrame({
#     "Age": [45, 25, 35],
#     "Occupation": ["Tutor", "HR", "Engineer"],
#     "Gender": ["Female", "Female", "Male"],
#     "Same_ofiice_home_location": ["Yes", "No", "Yes"],
#     "kids": ["Yes", "Yes", "No"],
#     "RM_save_money": ["Yes", "No", "Yes"],
#     "RM_quality_time": ["Yes", "No", "Yes"],
#     "RM_better_sleep": ["Yes", "No", "Yes"],
#     "calmer_stressed": ["CALMER", "STRESSED", "CALMER"],
#     "RM_professional_growth": [5, 2, 5],
#     "RM_lazy": [1, 4, 1],
#     "RM_productive": [5, 2, 4],
#     "digital_connect_sufficient": ["Yes", "No", "Yes"],
#     "RM_better_work_life_balance": [5, 1, 4],
#     "RM_improved_skillset": [5, 3, 4],
#     "RM_job_opportunities": ["Yes", "No", "Yes"],
# })
#
# # Make predictions using your ID3 algorithm
# predictions = predict_id3(tree, test_instances)
#
# # Display the predictions
# print("Predictions:")
# print(predictions)


# d.
def get_splits(attribute_values, labels):
    data = pd.DataFrame({'attribute': attribute_values, 'label': labels})
    sorted_data = data.sort_values(by='attribute')
    unique_values = sorted_data['attribute'].unique()
    midpoints = [(unique_values[i] + unique_values[i + 1]) / 2 for i in range(len(unique_values) - 1)]
    return midpoints


# attribute_values_professional_growth = df["RM_professional_growth"].values
# labels = df["Target"].values
# splits_professional_growth = get_splits(attribute_values_professional_growth, labels)
# print("Potential split points for RM_professional_growth:", splits_professional_growth)


# e.
def id3(df, target_attribute, attributes):
    # Base case: If all examples have the same class, return a leaf node
    if df[target_attribute].nunique() == 1:
        return {target_attribute: df[target_attribute].iloc[0]}

    # Base case: If there are no attributes left, return the majority class
    if not attributes:
        majority_class = df[target_attribute].mode().iloc[0]
        return {target_attribute: majority_class}

    # Find the best attribute for splitting
    best_attribute, max_information_gain = find_root_node(df, target_attribute, attributes)

    # Create the tree node
    node = {
        best_attribute: {
            "n_observations": df[target_attribute].count(),
            "information_gain": max_information_gain,
            "values": {}
        }
    }

    # Check if the best attribute is continuous
    if best_attribute in continuous_attributes:
        # Discretize the continuous attribute using potential split points
        splits = get_splits(df[best_attribute].values, df[target_attribute].values)
        for split in splits:
            # Create subsets for each split point
            subset = df[df[best_attribute] <= split].copy()
            value = f"<= {round(split, 2)}"
            remaining_attributes = [attr for attr in attributes if attr != best_attribute]
            # Recursively build the subtree
            subtree = id3(subset, target_attribute, remaining_attributes)
            # Add the subtree to the current node
            node[best_attribute]["values"][value] = subtree

    else:
        # Recursively build the tree for each value of the best discrete attribute
        for value in df[best_attribute].unique():
            subset = df[df[best_attribute] == value].copy()
            remaining_attributes = [attr for attr in attributes if attr != best_attribute]
            subtree = id3(subset, target_attribute, remaining_attributes)
            node[best_attribute]["values"][value] = subtree

    return node

# tree = id3(df, target_attribute, attributes_to_analyze)
# print(tree)

#SKLEARN
# label_encoder = LabelEncoder()
# for column in discrete_attributes_without_target:
#     if column in df.columns:
#         df[column] = label_encoder.fit_transform(df[column])
#
# X = df[discrete_attributes_without_target + continuous_attributes]
# y = df["Target"]  # Target variable
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Train a Decision Tree Classifier using scikit-learn
# sklearn_tree = DecisionTreeClassifier()
# sklearn_tree.fit(X_train, y_train)
#
# # Predict using the scikit-learn model
# y_pred_sklearn = sklearn_tree.predict(X_test)
#
# # Calculate accuracy using scikit-learn
# accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
# print(f"Accuracy using scikit-learn DecisionTreeClassifier: {accuracy_sklearn:.2f}")


#f.
def id3_discrete_pruned(df, target_attribute, attributes, max_depth=None):
    # Base case: If all examples have the same class, return a leaf node
    if df[target_attribute].nunique() == 1:
        return {target_attribute: df[target_attribute].iloc[0]}

    # Base case: If there are no attributes left, return the majority class
    if not attributes:
        majority_class = df[target_attribute].mode().iloc[0]
        return {target_attribute: majority_class}

    # Find the best attribute for splitting
    best_attribute, max_information_gain = find_root_node(df, target_attribute, attributes)

    # Check if we've reached the maximum depth
    if max_depth is not None and max_depth == 0:
        majority_class = df[target_attribute].mode().iloc[0]
        return {target_attribute: majority_class}

    # Create the tree node
    node = {
        best_attribute: {
            "n_observations": df[target_attribute].count(),
            "information_gain": max_information_gain,
            "values": {}
        }
    }

    # Recursively build the tree for each value of the best attribute
    for value in df[best_attribute].unique():
        # Filter the data for the current attribute value
        subset = df[df[best_attribute] == value].copy()

        # Remove the best attribute from the set of attributes
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        # Recursively build the subtree with reduced max_depth
        subtree = id3_discrete_pruned(subset, target_attribute, remaining_attributes, max_depth=max_depth - 1 if max_depth is not None else None)

        # Add the subtree to the current node
        node[best_attribute]["values"][value] = subtree

    return node

# discrete_tree_pruned = id3_discrete_pruned(df, target_attribute, discrete_attributes_without_target, max_depth=3)
# print(discrete_tree_pruned)


def id3_pruned_simple(df, target_attribute, attributes, max_depth=None):
    # Combine continuous and discrete attributes
    continuous_attributes = ["RM_professional_growth", "RM_lazy", "RM_productive", "RM_better_work_life_balance",
                              "RM_improved_skillset"]
    # Separate continuous and discrete attributes
    discrete_attributes_without_target = ["Age", "Gender", "Same_ofiice_home_location", "kids", "RM_save_money",
                                          "RM_quality_time", "RM_better_sleep", "calmer_stressed",
                                          "digital_connect_sufficient", "RM_job_opportunities"]
    continuous_attributes = ["RM_professional_growth", "RM_lazy", "RM_productive", "RM_better_work_life_balance",
                              "RM_improved_skillset"]

    # Assuming X and y are already defined for the entire dataset
    X = df[discrete_attributes_without_target + continuous_attributes]
    y = df[target_attribute]

    # Combine discrete and continuous attributes
    attributes_to_analyze = discrete_attributes_without_target + continuous_attributes

    # Train the tree
    tree = id3_pruned_simple_helper(df, target_attribute, attributes_to_analyze, max_depth=max_depth)

    return tree


def id3_pruned_simple_helper(df, target_attribute, attributes, depth=0, max_depth=None):
    # Base case: If all examples have the same class, return a leaf node
    if df[target_attribute].nunique() == 1:
        return {target_attribute: df[target_attribute].iloc[0]}

    # Base case: If there are no attributes left, return the majority class
    if not attributes:
        majority_class = df[target_attribute].mode().iloc[0]
        return {target_attribute: majority_class}

    # Check if we've reached the maximum depth
    if max_depth is not None and depth >= max_depth:
        majority_class = df[target_attribute].mode().iloc[0]
        return {target_attribute: majority_class}

    # Find the best attribute for splitting
    best_attribute, max_information_gain = find_root_node(df, target_attribute, attributes)

    # Create the tree node
    node = {
        best_attribute: {
            "n_observations": df[target_attribute].count(),
            "information_gain": max_information_gain,
            "values": {}
        }
    }

    # Recursively build the tree for each value of the best attribute
    for value in df[best_attribute].unique():
        # Filter the data for the current attribute value
        subset = df[df[best_attribute] == value].copy()

        # Remove the best attribute from the set of attributes
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        # Recursively build the subtree with increased depth
        subtree = id3_pruned_simple_helper(subset, target_attribute, remaining_attributes, depth=depth + 1, max_depth=max_depth)

        # Add the subtree to the current node
        node[best_attribute]["values"][value] = subtree

    return node

# tree_pruned_simple = id3_pruned_simple(df, target_attribute="Target", attributes=attributes_to_analyze, max_depth=3)
# print(tree_pruned_simple)


#g.
# def id3_pruned_cv(df, target_attribute, attributes, max_depth_values=None, cv=5):
#     if max_depth_values is None:
#         max_depth_values = [None] + list(range(1, 11))  # Including None for no pruning
#
#     # Combine discrete and continuous attributes
#     attributes_to_analyze = ["Age", "Gender", "Same_ofiice_home_location", "kids", "RM_save_money",
#                              "RM_quality_time", "RM_better_sleep", "calmer_stressed",
#                              "digital_connect_sufficient", "RM_job_opportunities",
#                              "RM_professional_growth", "RM_lazy", "RM_productive", "RM_better_work_life_balance",
#                              "RM_improved_skillset"]
#
#     # Assuming X and y are already defined for the entire dataset
#     X = df[attributes_to_analyze]
#     y = df[target_attribute]
#
#     best_score = float('-inf')
#     best_max_depth = None
#
#     for max_depth in max_depth_values:
#         scores = []  # Store scores for each fold
#
#         for _ in range(cv):
#             # Split the data into training and validation sets (you need to implement this)
#             X_train, X_valid, y_train, y_valid = custom_train_test_split(X, y)
#
#             # Train the model on the training set (you need to implement this)
#             tree = id3_pruned_simple(X_train, y_train, attributes, max_depth=max_depth)
#
#             # Evaluate the model on the validation set (you need to implement this)
#             accuracy = custom_evaluate(tree, X_valid, y_valid)
#
#             scores.append(accuracy)
#
#         average_score = sum(scores) / cv
#
#         print(f"Max Depth: {max_depth}, Average Accuracy: {average_score}")
#
#         if average_score > best_score:
#             best_score = average_score
#             best_max_depth = max_depth
#
#     print(f"Best Max Depth: {best_max_depth}, Best Average Accuracy: {best_score}")
#
#     # Train the final model with the best max depth on the entire dataset
#     final_tree = id3_pruned_simple(X, y, attributes, max_depth=best_max_depth)
#
#     return final_tree
#
#
# def custom_train_test_split(X, y, test_size=0.2, random_state=None):
#     """
#     Split the dataset into training and testing sets without using scikit-learn.
#
#     Parameters:
#     - X: Features
#     - y: Labels
#     - test_size: The proportion of the dataset to include in the test split
#     - random_state: Seed for reproducibility
#
#     Returns:
#     - X_train: Training features
#     - X_test: Testing features
#     - y_train: Training labels
#     - y_test: Testing labels
#     """
#     if random_state is not None:
#         np.random.seed(random_state)
#
#     # Shuffle indices
#     indices = np.arange(len(X))
#     np.random.shuffle(indices)
#
#     # Calculate the split index
#     split_index = int((1 - test_size) * len(X))
#
#     # Split the dataset
#     X_train, X_test = X.iloc[indices[:split_index]], X.iloc[indices[split_index:]]
#     y_train, y_test = y.iloc[indices[:split_index]], y.iloc[indices[split_index:]]
#
#     return X_train, X_test, y_train, y_test
#
# def custom_evaluate(tree, X, y):
#     """
#     Evaluate the model without using scikit-learn.
#
#     Parameters:
#     - tree: The decision tree model
#     - X: Features
#     - y: Labels
#
#     Returns:
#     - accuracy: The accuracy of the model on the given dataset
#     """
#     correct_predictions = 0
#
#     for i in range(len(X)):
#         instance = X.iloc[i]
#         prediction = predict_id3(tree, instance)
#
#         if prediction == y.iloc[i]:
#             correct_predictions += 1
#
#     accuracy = correct_predictions / len(X)
#
#     return accuracy


