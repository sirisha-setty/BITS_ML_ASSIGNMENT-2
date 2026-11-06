# Import pandas library
import pandas as pd

# Define meaningful column names
column_names = [
    "ID",
    "Diagnosis",

    # Mean measurements
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    # Standard error measurements
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    # Worst measurements
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

# Load dataset with meaningful column names
data = pd.read_csv(
    "dataset/wdbc.data",
    header=None,
    names=column_names
)

# Display column names
print("Column names:")
print(data.columns.tolist())

# Display first five records
print("\nFirst five records:")
print(data.head())

# Display dataset dimensions
print("\nDataset shape:")
print(data.shape)
# -----------------------------------------
# STEP 12: DATA QUALITY CHECKING
# -----------------------------------------

# Check missing values in each column
print("\nMissing values in each column:")
print(data.isnull().sum())

# Calculate total missing values
total_missing = data.isnull().sum().sum()

print("\nTotal missing values:")
print(total_missing)

# Check duplicate records
total_duplicates = data.duplicated().sum()

print("\nTotal duplicate records:")
print(total_duplicates)

# STEP 13: SEPARATE FEATURES AND TARGET

# Remove ID and Diagnosis columns from input features
X = data.drop(columns=["ID", "Diagnosis"])

# Store Diagnosis separately as target
y = data["Diagnosis"]

# Display results
print("\nFeatures (X) shape:")
print(X.shape)

print("\nTarget (y) shape:")
print(y.shape)

print("\nDiagnosis distribution:")
print(y.value_counts())


# -----------------------------------------
# STEP 14: ENCODE TARGET LABELS
# -----------------------------------------

# Convert diagnosis labels into numerical values
y = y.map({
    "B": 0,
    "M": 1
})

# Display the converted diagnosis values
print("\nEncoded diagnosis distribution:")
print(y.value_counts())

# Check the data type
print("\nTarget data type:")
print(y.dtype)

# Check for missing values after encoding
print("\nMissing values in target:")
print(y.isnull().sum())

# -----------------------------------------
# STEP 15: TRAIN-TEST SPLIT
# -----------------------------------------

# Import train_test_split function
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,                  # Input features
    y,                  # Target labels
    test_size=0.20,     # 20% testing data
    random_state=42,    # Ensures reproducible results
    stratify=y          # Maintains class proportions
)

# Display the shapes of the datasets
print("\nTraining and Testing Dataset Shapes:")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Display diagnosis distribution in training data
print("\nTraining Diagnosis Distribution:")
print(y_train.value_counts())

# Display diagnosis distribution in testing data
print("\nTesting Diagnosis Distribution:")
print(y_test.value_counts())

# -----------------------------------------
# STEP 16: FEATURE SCALING
# -----------------------------------------

# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Create the scaler object
scaler = StandardScaler()

# Calculate mean and standard deviation from training data
# Then transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Use the same training parameters to transform test data
X_test_scaled = scaler.transform(X_test)

# Display shapes
print("\nScaled Training Data Shape:")
print(X_train_scaled.shape)

print("\nScaled Testing Data Shape:")
print(X_test_scaled.shape)

# Display first five scaled records
print("\nFirst five scaled training records:")
print(X_train_scaled[:5])

