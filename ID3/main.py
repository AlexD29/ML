import math
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. Preprocessing

df1 = pd.read_csv("WFOvsWFH.csv")
discrete_attributes = [
    "Age",
    "Gender",
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
attributes_to_analyze = [
    "Age",
    "Gender",
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

#a.
def remove_nan_values_from_csv(df, output_csv_file_path):
  nan_values = df.isna()
  df = df.dropna()
  df.to_csv(output_csv_file_path, index=False)

# remove_nan_values_from_csv(df1,"WFOvsWFHCleaned.csv")

#b.
df = pd.read_csv("WFOvsWFHCleaned.csv")
def calculate_mean_and_variance(df, continuous_attributes):
  mean_and_variance = {}
  for attribute in continuous_attributes:
    mean = df[attribute].mean()
    variance = df[attribute].var()
    mean_and_variance[attribute] = {
      "mean": round(mean,2),
      "variance": round(variance,2)
    }
  return mean_and_variance

# mean_and_variance = calculate_mean_and_variance(df, continuous_attributes)
# print(mean_and_variance)


#2. Probabilities, Information Theory

#a.
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


#b.
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


#c.
def calculate_conditional_entropy(df, target_attribute, given_attribute):
  #P(Y|X)
  conditional_probabilities = df.groupby(given_attribute)[target_attribute].value_counts(normalize=True).unstack()

  #H(Y|X)
  prob_x = df[given_attribute].value_counts(normalize=True)
  prob_y_given_x = conditional_probabilities.apply(lambda row: row * np.log2(row)).sum(axis=1)

  conditional_entropy = -(prob_x * prob_y_given_x).sum()

  return conditional_entropy


#d.
def calculate_information_gain(df, target_attribute, given_attribute):
  entropy_before_split = calculate_entropy(compute_probabilities(df, target_attribute))
  conditional_entropy = calculate_conditional_entropy(df, target_attribute, given_attribute)
  information_gain = entropy_before_split - conditional_entropy
  return information_gain

# information_gain = calculate_information_gain(df, "Target", "calmer_stressed")
# print(f"Information Gain: {information_gain}")


#3. ID3

#a.
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


#b.
def id3_discrete(df, target_attribute, attributes):
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

    # Recursively build the tree for each value of the best attribute
    for value in df[best_attribute].unique():
        # Filter the data for the current attribute value
        subset = df[df[best_attribute] == value].copy()

        # Remove the best attribute from the set of attributes
        remaining_attributes = [attr for attr in attributes if attr != best_attribute]

        # Recursively build the subtree
        subtree = id3_discrete(subset, target_attribute, remaining_attributes)

        # Add the subtree to the current node
        node[best_attribute]["values"][value] = subtree

    return node

tree = id3_discrete(df, target_attribute, attributes_to_analyze)
print(tree)


