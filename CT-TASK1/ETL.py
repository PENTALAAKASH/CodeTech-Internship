import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Step 1: Extract - Load raw data
def extract_data(input_path):
    """
    Extract data from the source file.
    
    """
    return pd.read_csv(input_path)

# Step 2: Transform - Data preprocessing
def transform_data(df):
    """
    Transform the raw data: handle missing values, scale numerical features,
    and encode categorical features.
   
    """
    # Separate features into numerical and categorical
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = df.select_dtypes(include=['object']).columns

    # Preprocessing for numerical features
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
        ('scaler', StandardScaler())                 # Standardize the values
    ])

    # Preprocessing for categorical features
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values
        ('encoder', OneHotEncoder(handle_unknown='ignore'))    # One-hot encode
    ])

    # Combine pipelines into a column transformer
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    # Apply the transformations
    transformed_data = preprocessor.fit_transform(df)

    # Convert transformed data back into a DataFrame
    transformed_df = pd.DataFrame(
        transformed_data.toarray() if hasattr(transformed_data, 'toarray') else transformed_data,
        columns=numerical_features.tolist() + 
                list(preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features))
    )

    return transformed_df

# Step 3: Load - Save the transformed data
def load_data(df, output_path):
    """
    Save the transformed data to a specified output path.
    
    """
    df.to_csv(output_path, index=False)
    print(f"Transformed data saved to {output_path}")

# Main function to run the ETL pipeline
def main(input_path, output_path):
    print("Starting the ETL pipeline...")
    raw_data = extract_data(input_path)
    print("Data extraction completed.")
    
    processed_data = transform_data(raw_data)
    print("Data transformation completed.")
    
    load_data(processed_data, output_path)
    print("ETL pipeline completed successfully.")

# Replace with file paths
if __name__ == "__main__":
    input_csv = "tips.csv"  # Replace with input file path
    output_csv = "processed_data2.csv"  # Replace with output file path
    main(input_csv, output_csv)
