import csv
import re
import pandas as pd

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


