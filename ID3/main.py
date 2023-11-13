import math
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. Preprocessing

#a.
df1 = pd.read_csv("SalaryPrediction.csv")
def remove_nan_values_from_csv(df, output_csv_file_path):
  """Removes NaN values from a CSV file and saves the cleaned file to the specified output path.

  Args:
    csv_file_path: The path to the CSV file to be cleaned.
    output_csv_file_path: The path to the output CSV file.
  """

  # Identify the NaN values in the DataFrame.
  nan_values = df.isna()

  # Remove the rows that contain NaN values.
  df = df.dropna()

  # Save the cleaned DataFrame to the output CSV file.
  df.to_csv(output_csv_file_path, index=False)
#remove_nan_values_from_csv(df1,"SalaryPredictionCleaned.csv")

#b.
df = pd.read_csv("SalaryPredictionCleaned.csv")
def calculate_mean_and_variance(df, numerical_attributes):
  """Calculates the mean and variance for each numerical attribute in a Pandas DataFrame and stores them in a dictionary.

  Args:
    df: A Pandas DataFrame.
    numerical_attributes: A list of the numerical attributes to calculate the mean and variance for.

  Returns:
    A dictionary containing the mean and variance for each numerical attribute.
  """

  mean_and_variance = {}

  for attribute in numerical_attributes:
    mean = df[attribute].mean()
    variance = df[attribute].var()

    mean_and_variance[attribute] = {
      "mean": mean,
      "variance": variance
    }

  return mean_and_variance
numerical_attributes = ["Age", "Wage"]
mean_and_variance = calculate_mean_and_variance(df, numerical_attributes)
#print(mean_and_variance)


#2. Probabilities, Information Theory

#a.
def compute_probabilities(df, attribute):
  """Calculates the probability mass function of a discrete attribute.

  Args:
    df: A Pandas DataFrame.
    attribute: The name of the discrete attribute.

  Returns:
    A dictionary containing the probability of each value of the discrete attribute.
  """

  probabilities = {}

  for value in df[attribute].unique():
    probability = df[attribute].value_counts()[value] / len(df)
    probabilities[value] = probability

  return probabilities


# Apply the function to the discrete attributes from your dataset
df = pd.read_csv("SalaryPredictionCleaned.csv")
discrete_attributes = ["Club", "League", "Nation", "Position", "Apps", "Caps"]

# for attribute in discrete_attributes:
#   probabilities = compute_probabilities(df, attribute)
#   print(f"Probability mass function of {attribute}:")
#   print(probabilities)


#b.
def calculate_entropy(probabilities):
  """Calculates the entropy of a random variable given its probability distribution.

  Args:
      probabilities: A dictionary containing the probability of each value of the random variable.

  Returns:
      The entropy of the random variable.
  """
  entropy = 0.0

  for value, probability in probabilities.items():
    entropy += -probability * math.log2(probability)

  return entropy


def calculate_entropy_for_discrete_attributes(df):
  """Calculates the entropy for each discrete attribute in a Pandas DataFrame.

  Args:
    df: A Pandas DataFrame.

  Returns:
    A dictionary containing the entropy of each discrete attribute.
  """

  discrete_attributes = ["Club", "League", "Nation", "Position", "Apps", "Caps"]
  entropy_dict = {}

  for attribute in discrete_attributes:
    probabilities = compute_probabilities(df, attribute)
    entropy = calculate_entropy(probabilities)

    entropy_dict[attribute] = entropy

  return entropy_dict


# entropy_dict = calculate_entropy_for_discrete_attributes(df)
#
# print("Entropy for each discrete attribute:")
# print(entropy_dict)


#c.
def calculate_conditional_entropy(df, target_attribute, given_attribute):
  """Calculates the conditional entropy of the target attribute given a specific attribute.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute (Y).
      given_attribute: The name of the attribute to condition on (X).

  Returns:
      The conditional entropy H(Y|X).
  """
  # Calculate the conditional probabilities P(Y|X)
  conditional_probabilities = df.groupby(given_attribute)[target_attribute].value_counts(normalize=True).unstack()

  # Calculate the conditional entropy H(Y|X)
  prob_x = df[given_attribute].value_counts(normalize=True)
  prob_y_given_x = conditional_probabilities.apply(lambda row: row * np.log2(row)).sum(axis=1)

  conditional_entropy = -(prob_x * prob_y_given_x).sum()

  return conditional_entropy


#d.
def calculate_information_gain(df, target_attribute, given_attribute):
  """Calculates the information gain of splitting on a specific attribute.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute.
      given_attribute: The name of the attribute to split on.

  Returns:
      The information gain.
  """
  # Calculate the entropy of the target variable before the split
  entropy_before_split = calculate_entropy(compute_probabilities(df, target_attribute))

  # Calculate the conditional entropy of the target variable given the attribute
  conditional_entropy = calculate_conditional_entropy(df, target_attribute, given_attribute)

  # Calculate the information gain
  information_gain = entropy_before_split - conditional_entropy

  return information_gain

# information_gain = calculate_information_gain(df, "Wage", "Caps")
# print(f"Information Gain by splitting on Position: {information_gain}")


#3. ID3

#a.
def find_root_node(df, target_attribute, attributes):
  """Finds the attribute picked by ID3 as the root.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute.
      attributes: A list of attribute names to consider.

  Returns:
      A tuple with the name of the attribute and the information gain.
  """
  max_information_gain = -1
  best_attribute = None

  for attribute in attributes:
    information_gain = calculate_information_gain(df, target_attribute, attribute)

    if information_gain > max_information_gain:
      max_information_gain = information_gain
      best_attribute = attribute

  return (best_attribute, max_information_gain)


# Assuming df is your DataFrame and target_attribute is the target variable
# attributes_to_consider = ["Club", "League", "Nation", "Position", "Apps", "Caps"]
# root_node = find_root_node(df, "Wage", attributes_to_consider)
#
# print(f"The root node is: {root_node[0]} with Information Gain: {root_node[1]}")


#b.
def id3_discrete(df, target_attribute, attributes):
  """Implements the ID3 algorithm for discrete attributes.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute.
      attributes: A list of attribute names to consider.

  Returns:
      A dictionary representing the decision tree structure.
  """
  # If all examples have the same target value, create a leaf node
  unique_values = df[target_attribute].unique()
  if len(unique_values) == 1:
    return {attributes[0]: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": 0.0,
      "values": {unique_values[0]: {}}
    }}

  # If there are no attributes left, create a leaf node with the majority target value
  if len(attributes) == 0:
    majority_value = df[target_attribute].mode().iloc[0]
    return {target_attribute: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": 0.0,
      "values": {majority_value: {}}
    }}

  # Find the best attribute to split on
  root_attribute, information_gain = find_root_node(df, target_attribute, attributes)

  # Create the tree structure
  tree = {
    root_attribute: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": information_gain,
      "values": {}
    }
  }

  # Recursively build the tree for each value of the root attribute
  for value in df[root_attribute].unique():
    subset = df[df[root_attribute] == value]
    subtree = id3_discrete(subset, target_attribute, [attr for attr in attributes if attr != root_attribute])
    tree[root_attribute]["values"][value] = subtree

  return tree


# Assuming df is your DataFrame and target_attribute is the target variable
# attributes_to_consider = ["Club", "League", "Nation", "Position", "Apps", "Caps"]
# decision_tree = id3_discrete(df, "Wage", attributes_to_consider)
# print(decision_tree)


#c.
# Sample dataset with only discrete attributes
# Assuming df is your original DataFrame with discrete and continuous attributes
# and target_attribute is the target variable

# Identify discrete attributes (you may need to adjust this based on your specific dataset)
discrete_attributes = ["Club", "League", "Nation", "Position", "Apps", "Caps"]
target_attribute = "Wage"
# Filter the DataFrame to include only discrete attributes and the target variable
discrete_df = df[discrete_attributes + [target_attribute]]

# decision_tree_id3 = id3_discrete(discrete_df, target_attribute, discrete_attributes)

# Print the decision tree structure from id3_discrete
# print("Decision Tree from id3_discrete:")
# print(decision_tree_id3)


# Encode categorical attributes
discrete_df_encoded = pd.get_dummies(discrete_df, columns=discrete_attributes, drop_first=True)

# Split the data into features (X) and target variable (y)
X = discrete_df_encoded.drop(target_attribute, axis=1)
y = discrete_df_encoded[target_attribute]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the DecisionTreeClassifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Print the decision tree structure from scikit-learn
# print("\nDecision Tree from scikit-learn:")
# print(dt_classifier.tree_.__getstate__()['nodes'])

# Evaluate the accuracy of the scikit-learn model
accuracy = accuracy_score(y_test, y_pred)
# print(f"\nAccuracy from scikit-learn: {accuracy}")


#d.

def get_splits(attribute_values):
  """Identifies potential splits for discretization of a continuous attribute.

  Args:
      attribute_values: A list or array of continuous attribute values.

  Returns:
      A list of potential split points.
  """
  unique_values = sorted(set(attribute_values))
  splits = [(unique_values[i] + unique_values[i + 1]) / 2.0 for i in range(len(unique_values) - 1)]
  return splits


# Example usage:
# Assuming 'Age' is a continuous attribute in your dataset
age_values = df['Age'].tolist()
age_splits = get_splits(age_values)

# print("Potential split points for Age:")
# print(age_splits)


#e.

def id3(df, target_attribute, attributes):
  """Implements the ID3 algorithm for a dataset with both continuous and discrete attributes.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute.
      attributes: A list of attribute names to consider.

  Returns:
      A dictionary representing the decision tree structure.
  """
  # If all examples have the same target value, create a leaf node
  unique_values = df[target_attribute].unique()
  if len(unique_values) == 1:
    return {attributes[0]: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": 0.0,
      "values": {unique_values[0]: {}}
    }}

  # If there are no attributes left, create a leaf node with the majority target value
  if len(attributes) == 0:
    majority_value = df[target_attribute].mode().iloc[0]
    return {target_attribute: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": 0.0,
      "values": {majority_value: {}}
    }}

  # Determine the type of each attribute (continuous or discrete)
  attribute_types = {attr: 'continuous' if df[attr].dtype == 'float64' else 'discrete' for attr in attributes}

  # Find the best attribute to split on
  root_attribute, information_gain, split_point = find_best_split(df, target_attribute, attributes, attribute_types)

  # Create the tree structure
  tree = {
    root_attribute: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": information_gain,
      "split_point": split_point,
      "values": {}
    }
  }

  # Recursively build the tree for each value of the root attribute
  if attribute_types[root_attribute] == 'continuous':
    # For continuous attribute, split the data into two subsets
    subset1 = df[df[root_attribute] <= split_point]
    subset2 = df[df[root_attribute] > split_point]

    # Recursively build the tree for each subset
    tree[root_attribute]["values"]["<= " + str(split_point)] = id3(subset1, target_attribute,
                                                                   [attr for attr in attributes if
                                                                    attr != root_attribute])
    tree[root_attribute]["values"]["> " + str(split_point)] = id3(subset2, target_attribute,
                                                                  [attr for attr in attributes if
                                                                   attr != root_attribute])
  else:
    # For discrete attribute, build the tree for each value of the attribute
    for value in df[root_attribute].unique():
      subset = df[df[root_attribute] == value]
      subtree = id3(subset, target_attribute, [attr for attr in attributes if attr != root_attribute])
      tree[root_attribute]["values"][value] = subtree

  return tree


def find_best_split(df, target_attribute, attributes, attribute_types):
  """Finds the best attribute and split point for continuous attributes.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute.
      attributes: A list of attribute names to consider.
      attribute_types: A dictionary specifying the type (continuous or discrete) of each attribute.

  Returns:
      A tuple with the best attribute, information gain, and split point.
  """
  max_information_gain = -1
  best_attribute = None
  best_split_point = None

  for attribute in attributes:
    if attribute_types[attribute] == 'continuous':
      # For continuous attributes, find potential split points
      splits = get_splits(df[attribute])

      for split_point in splits:
        subset1 = df[df[attribute] <= split_point]
        subset2 = df[df[attribute] > split_point]

        information_gain = calculate_information_gain_continuous(df, target_attribute, subset1, subset2)

        if information_gain > max_information_gain:
          max_information_gain = information_gain
          best_attribute = attribute
          best_split_point = split_point
    else:
      # For discrete attributes, calculate information gain without splitting
      information_gain = calculate_information_gain(df, target_attribute, attribute)

      if information_gain > max_information_gain:
        max_information_gain = information_gain
        best_attribute = attribute
        best_split_point = None

  return best_attribute, max_information_gain, best_split_point


def calculate_information_gain_continuous(df, target_attribute, subset1, subset2):
  """Calculates the information gain for continuous attributes.

  Args:
      df: The original DataFrame.
      target_attribute: The name of the target attribute.
      subset1: A subset of the DataFrame where the attribute values are less than or equal to the split point.
      subset2: A subset of the DataFrame where the attribute values are greater than the split point.

  Returns:
      The information gain.
  """
  total_entropy = calculate_entropy(compute_probabilities(df, target_attribute))

  weight1 = len(subset1) / len(df)
  weight2 = len(subset2) / len(df)

  entropy1 = calculate_entropy(compute_probabilities(subset1, target_attribute))
  entropy2 = calculate_entropy(compute_probabilities(subset2, target_attribute))

  information_gain = total_entropy - (weight1 * entropy1 + weight2 * entropy2)
  return information_gain


# Example usage:
# Assuming df is your original DataFrame and target_attribute is the target variable
# all_attributes = df.columns.drop(target_attribute).tolist()
# decision_tree_id3_combined = id3(df, target_attribute, all_attributes)
#
# # Print or analyze the decision tree structure from id3
# print("Decision Tree from id3:")
# print(decision_tree_id3_combined)


# Assume df is your original DataFrame and target_attribute is the target variable
all_attributes = df.columns.drop(target_attribute).tolist()

# Split the data into features (X) and target variable (y)
X = df[all_attributes]
y = df[target_attribute]

# Encode categorical attributes
X_encoded = pd.get_dummies(X, drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and fit the DecisionTreeClassifier from scikit-learn
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set using scikit-learn's decision tree
y_pred_dt = dt_classifier.predict(X_test)

# Calculate accuracy from scikit-learn
# accuracy_dt = accuracy_score(y_test, y_pred_dt)
# print(f"Accuracy from scikit-learn: {accuracy_dt}")


#f.
def id3_pruning(df, target_attribute, attributes, max_depth=None, min_samples_split=2):
  """Implements the ID3 algorithm for a dataset with both continuous and discrete attributes, with pruning.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute.
      attributes: A list of attribute names to consider.
      max_depth: Maximum depth of the tree for pruning.
      min_samples_split: Minimum number of samples required to split further.

  Returns:
      A dictionary representing the decision tree structure.
  """
  # If all examples have the same target value, create a leaf node
  unique_values = df[target_attribute].unique()
  if len(unique_values) == 1:
    return {attributes[0]: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": 0.0,
      "values": {unique_values[0]: {}}
    }}

  # If there are no attributes left, create a leaf node with the majority target value
  if len(attributes) == 0:
    majority_value = df[target_attribute].mode().iloc[0]
    return {target_attribute: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": 0.0,
      "values": {majority_value: {}}
    }}

  # Determine the type of each attribute (continuous or discrete)
  attribute_types = {attr: 'continuous' if df[attr].dtype == 'float64' else 'discrete' for attr in attributes}

  # Find the best attribute to split on
  root_attribute, information_gain, split_point = find_best_split(df, target_attribute, attributes, attribute_types)

  # Create the tree structure
  tree = {
    root_attribute: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": information_gain,
      "split_point": split_point,
      "values": {}
    }
  }

  # Recursively build the tree for each value of the root attribute
  if attribute_types[root_attribute] == 'continuous':
    # For continuous attribute, split the data into two subsets
    subset1 = df[df[root_attribute] <= split_point]
    subset2 = df[df[root_attribute] > split_point]

    # Check for pruning conditions
    if max_depth is not None and max_depth <= 1:
      tree[root_attribute]["values"]["<= " + str(split_point)] = create_leaf_node(subset1, target_attribute)
    else:
      tree[root_attribute]["values"]["<= " + str(split_point)] = id3_pruning(
        subset1, target_attribute, [attr for attr in attributes if attr != root_attribute],
        max_depth=max_depth - 1 if max_depth is not None else None,
        min_samples_split=min_samples_split
      )

    if max_depth is not None and max_depth <= 1:
      tree[root_attribute]["values"]["> " + str(split_point)] = create_leaf_node(subset2, target_attribute)
    else:
      tree[root_attribute]["values"]["> " + str(split_point)] = id3_pruning(
        subset2, target_attribute, [attr for attr in attributes if attr != root_attribute],
        max_depth=max_depth - 1 if max_depth is not None else None,
        min_samples_split=min_samples_split
      )
  else:
    # For discrete attribute, build the tree for each value of the attribute
    for value in df[root_attribute].unique():
      subset = df[df[root_attribute] == value]
      subtree = id3_pruning(
        subset, target_attribute, [attr for attr in attributes if attr != root_attribute],
        max_depth=max_depth - 1 if max_depth is not None else None,
        min_samples_split=min_samples_split
      )
      tree[root_attribute]["values"][value] = subtree

  return tree


def create_leaf_node(df, target_attribute):
  """Creates a leaf node with the majority target value.

  Args:
      df: A subset of the original DataFrame.
      target_attribute: The name of the target attribute.

  Returns:
      A leaf node.
  """
  majority_value = df[target_attribute].mode().iloc[0]
  return {
    target_attribute: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": 0.0,
      "values": {majority_value: {}}
    }
  }

# all_attributes = df.columns.drop(target_attribute).tolist()
# decision_tree_id3_combined_pruning = id3_pruning(df, target_attribute, all_attributes)
#
# # Print or analyze the decision tree structure from id3
# print("Decision Tree from id3_pruning:")
# print(decision_tree_id3_combined_pruning)


def id3_discrete_pruning(df, target_attribute, attributes, max_depth=None, min_samples_split=2):
  """Implements the ID3 algorithm for a dataset with discrete attributes, with pruning.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute.
      attributes: A list of attribute names to consider.
      max_depth: Maximum depth of the tree for pruning.
      min_samples_split: Minimum number of samples required to split further.

  Returns:
      A dictionary representing the decision tree structure.
  """
  # If all examples have the same target value, create a leaf node
  unique_values = df[target_attribute].unique()
  if len(unique_values) == 1:
    return {attributes[0]: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": 0.0,
      "values": {unique_values[0]: {}}
    }}

  # If there are no attributes left, create a leaf node with the majority target value
  if len(attributes) == 0:
    majority_value = df[target_attribute].mode().iloc[0]
    return {target_attribute: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": 0.0,
      "values": {majority_value: {}}
    }}

  # Find the best attribute to split on
  root_attribute, information_gain = find_best_attribute_pruning(df, target_attribute, attributes)

  # Create the tree structure
  tree = {
    root_attribute: {
      "n_observations": df[target_attribute].value_counts().to_dict(),
      "information_gain": information_gain,
      "values": {}
    }
  }

  # Recursively build the tree for each value of the root attribute
  for value in df[root_attribute].unique():
    subset = df[df[root_attribute] == value]

    # Check for pruning conditions
    if max_depth is not None and max_depth <= 1:
      tree[root_attribute]["values"][value] = create_leaf_node(subset, target_attribute)
    elif len(subset) < min_samples_split:
      tree[root_attribute]["values"][value] = create_leaf_node(subset, target_attribute)
    else:
      tree[root_attribute]["values"][value] = id3_discrete_pruning(
        subset, target_attribute, [attr for attr in attributes if attr != root_attribute],
        max_depth=max_depth - 1 if max_depth is not None else None,
        min_samples_split=min_samples_split
      )

  return tree


# Helper function to find the best attribute for id3_discrete_pruning
def find_best_attribute_pruning(df, target_attribute, attributes):
  """Finds the best attribute for splitting in the ID3 algorithm for discrete attributes.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute.
      attributes: A list of attribute names to consider.

  Returns:
      A tuple with the best attribute and its information gain.
  """
  max_information_gain = -1
  best_attribute = None

  for attribute in attributes:
    information_gain = calculate_information_gain_pruning(df, target_attribute, attribute)

    if information_gain > max_information_gain:
      max_information_gain = information_gain
      best_attribute = attribute

  return best_attribute, max_information_gain


# Example of a helper function to calculate information gain for discrete attributes
def calculate_information_gain_pruning(df, target_attribute, attribute):
  """Calculates the information gain for a discrete attribute in the ID3 algorithm.

  Args:
      df: A Pandas DataFrame.
      target_attribute: The name of the target attribute.
      attribute: The name of the discrete attribute.

  Returns:
      The information gain for the attribute.
  """
  total_entropy = calculate_entropy(compute_probabilities_pruning(df, target_attribute))

  weighted_entropy = 0.0
  for value in df[attribute].unique():
    subset = df[df[attribute] == value]
    weight = len(subset) / len(df)
    entropy = calculate_entropy(compute_probabilities_pruning(subset, target_attribute))
    weighted_entropy += weight * entropy

  information_gain = total_entropy - weighted_entropy
  return information_gain


# Example of a helper function to compute probabilities for discrete attributes
def compute_probabilities_pruning(df, attribute):
  """Calculates the probability mass function of a discrete attribute.

  Args:
      df: A Pandas DataFrame.
      attribute: The name of the discrete attribute.

  Returns:
      A dictionary containing the probability of each value of the discrete attribute.
  """
  probabilities = {}

  for value in df[attribute].unique():
    probability = df[attribute].value_counts()[value] / len(df)
    probabilities[value] = probability

  return probabilities


# all_attributes = df.columns.drop(target_attribute).tolist()
# decision_tree_id3_discrete_pruning = id3_discrete_pruning(df, target_attribute, all_attributes)
#
# # Print or analyze the decision tree structure from id3
# print("Decision Tree from id3_pruning:")
# print(decision_tree_id3_discrete_pruning)


#f.

import numpy as np

def predict_id3_combined(tree, sample):
  """Predicts the target variable using the id3 decision tree.

  Args:
      tree: The decision tree produced by the id3 function.
      sample: A sample (row) from the dataset.

  Returns:
      The predicted value for the target variable.
  """
  current_node = list(tree.keys())[0]

  if current_node in sample:
    attribute_value = sample[current_node]

    if isinstance(tree[current_node]["values"][attribute_value], dict):
      return predict_id3_combined(tree[current_node]["values"][attribute_value], sample)
    else:
      # Reached a leaf node, return the predicted value
      return list(tree[current_node]["values"][attribute_value].keys())[0]
  else:
    # Attribute not present in the sample, return a default value (can be adjusted based on your needs)
    return None


# Assume df is your original DataFrame and target_attribute is the target variable
all_attributes = df.columns.drop(target_attribute).tolist()

# Encode categorical attributes
X_encoded = pd.get_dummies(df[all_attributes], drop_first=True)
y = df[target_attribute]

# Define a range of pruning parameters
max_depth_values = [None, 5, 10, 15]
min_samples_split_values = [2, 5, 10, 20]

best_accuracy = 0.0
best_hyperparameters = {}

# Perform cross-validation for each combination of hyperparameters
for max_depth in max_depth_values:
    for min_samples_split in min_samples_split_values:
        accuracy_scores = []

        # Perform k-fold cross-validation manually
        k = 5
        fold_size = len(X_encoded) // k

        for i in range(k):
            # Split the data into training and validation sets
            validation_start = i * fold_size
            validation_end = (i + 1) * fold_size
            X_val = X_encoded.iloc[validation_start:validation_end]
            y_val = y.iloc[validation_start:validation_end]
            X_train = pd.concat([X_encoded.iloc[:validation_start], X_encoded.iloc[validation_end:]])
            y_train = pd.concat([y.iloc[:validation_start], y.iloc[validation_end:]])

            # Train the decision tree
            tree = id3_pruning(X_train, target_attribute, all_attributes, max_depth=max_depth, min_samples_split=min_samples_split)

            # Convert the decision tree structure into a function for prediction
            predict_tree = lambda sample: predict_id3_combined(tree, sample)

            # Evaluate accuracy on the validation set
            y_pred_val = [predict_tree(sample) for _, sample in X_val.iterrows()]
            accuracy = np.sum(y_pred_val == y_val) / len(y_val)
            accuracy_scores.append(accuracy)

        # Calculate the average accuracy for the current hyperparameters
        average_accuracy = np.mean(accuracy_scores)

        # Update the best hyperparameters if the current combination is better
        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            best_hyperparameters = {'max_depth': max_depth, 'min_samples_split': min_samples_split}

        print(f"Hyperparameters: max_depth={max_depth}, min_samples_split={min_samples_split}, Average Accuracy: {average_accuracy}")

print("Best Hyperparameters:", best_hyperparameters)
