import pandas as pd

# Load the CSV files
test_df = pd.read_csv()  # File with stance labels
results_df = pd.read_csv()  # File with results (A, B, C)

# Limit the number of rows in test_df to match the number of rows in results_df
test_df = test_df.head(len(results_df))  # Read only as many rows as are in results.csv

# Map results in results.csv to the corresponding stance labels
label_map = {'B': 'Favor', 'A': 'Against', 'C': 'Neutral'}

# Assuming the results are stored starting from column "0" in results.csv (i.e., index 0)
# Extract the results column (e.g., column 0 of results.csv)
results_column = results_df.iloc[:, 0]

for idx,row in results_column.iterrows():

# Map results to stance labels
mapped_results = results_column.map(label_map)

# Now compare the stance labels in 'test.csv' with the mapped results
correctness = (test_df['stance_label'] == mapped_results)

# Optionally, calculate the accuracy or print the comparison
accuracy = correctness.mean()  # This gives the proportion of correct matches
print(f'Accuracy: {accuracy * 100:.2f}%')

# If you want to see the rows where the labels don't match
incorrect_rows = test_df[~correctness]
print("Incorrect rows:")
print(incorrect_rows)
