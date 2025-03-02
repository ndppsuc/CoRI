
import pandas as pd
from sklearn.metrics import f1_score

# Load the CSV file
file_path = r"results2.csv"  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Convert columns to strings
data['P'] = data['P'].astype(str)
data['stance_label'] = data['stance_label'].astype(str)

# Ensure only valid classes are considered
valid_labels = {"Against", "Favor", "Neutral"}  # Three categories
data = data[data['P'].isin(valid_labels) & data['stance_label'].isin(valid_labels)]

# Extract predictions and true labels
predicted_labels = data['P']
true_labels = data['stance_label']

# Calculate Micro-F1 score for three categories
micro_f1 = f1_score(true_labels, predicted_labels, average='macro')


print("Micro-F1 Score for Three Categories:", micro_f1)
