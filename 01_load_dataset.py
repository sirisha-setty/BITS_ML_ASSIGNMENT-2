# Import pandas library
import pandas as pd

# Specify the location of our dataset
file_path = "dataset/wdbc.data"

# Read the dataset
data = pd.read_csv(file_path, header=None)

# Display the number of rows and columns
print("Dataset shape:")
print(data.shape)

# Display the first five records
print("\nFirst five records:")
print(data.head())

# Display the column information
print("\nDataset information:")
print(data.info())

# Display the number of records in each diagnosis category
print("\nDiagnosis distribution:")
print(data[1].value_counts())
