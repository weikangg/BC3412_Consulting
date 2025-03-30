import pandas as pd
import numpy as np

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a wide-format DataFrame containing sustainability metrics.
    
    Cleaning steps:
      1. Identify meta columns (e.g., 'Metric', 'Units', 'Unit', 'Financial statement', 'SASB Topic')
      2. Identify year columns (headers that are entirely digits)
      3. Trim whitespace in meta columns and column headers.
      4. Replace "not reported" in year columns with np.nan and convert values to numeric.
      5. Drop rows where all year columns are missing.
    
    Returns:
      A cleaned wide-format DataFrame.
    """
    # Define meta columns (only keep those that exist)
    meta_columns = ['Metric', 'Units', "Unit", "Financial statement", 'SASB Topic']
    meta_columns = [col for col in meta_columns if col in df.columns]
    
    # Identify year columns: headers that are all digits
    year_columns = [col for col in df.columns if col.isdigit()]
    
    # Trim whitespace in meta columns
    for col in meta_columns:
        df[col] = df[col].astype(str).str.strip()
    
    # Process year columns: replace "not reported" with NaN and convert to numeric
    for col in year_columns:
        df[col] = df[col].replace("not reported", np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove extraneous whitespace from all column headers
    df.rename(columns=lambda x: x.strip() if isinstance(x, str) else x, inplace=True)
    
    # Drop rows where all year columns are missing
    df_cleaned = df.dropna(subset=year_columns, how='all').reset_index(drop=True)
    
    return df_cleaned

def pivot_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot a wide-format DataFrame to long (tidy) format.
    
    Assumes the DataFrame contains:
      - Meta columns (e.g., 'Metric', 'Units', 'Financial statement', 'SASB Topic', etc.)
      - Year columns (headers that are digits)
    
    Returns:
      A long-format DataFrame with the following columns:
         - All meta columns
         - 'Year' (originally the column names for years, now as values)
         - 'Value' (the corresponding numeric values)
    """
    # Define meta columns (only keep those that exist)
    meta_columns = ['Metric', 'Units', "Unit", "Financial statement", 'SASB Topic']
    meta_columns = [col for col in meta_columns if col in df.columns]
    
    # Identify year columns: headers that are digits
    year_columns = [col for col in df.columns if col.isdigit()]
    
    # Pivot using melt: convert year columns into a 'Year' and 'Value' pair.
    df_long = df.melt(id_vars=meta_columns,
                      value_vars=year_columns,
                      var_name="Year",
                      value_name="Value")
    
    # Convert 'Year' to numeric (optional but often useful)
    df_long["Year"] = pd.to_numeric(df_long["Year"], errors="coerce")
    
    return df_long

def clean_and_pivot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean a wide-format DataFrame and then convert it to long format.
    
    This function:
      1. Cleans the DataFrame (using clean_dataframe)
      2. Pivots the cleaned wide DataFrame to long format (using pivot_to_long)
    
    Returns:
      A long-format (tidy) DataFrame ready for further analysis.
    """
    df_cleaned = clean_dataframe(df)
    df_long = pivot_to_long(df_cleaned)
    return df_long

def extract_company_from_key(key: str) -> str:
    """
    Given a composite key (e.g. "nextera_energy_SASB_Metrics" or "nextera_energy_Social"),
    extract the company name by finding the first occurrence of one of the known aspects.
    Known aspects are: "Environment", "Financial", "Governance", "SASB_Metrics", "Social".
    Everything before the underscore preceding one of these aspects is considered the company name.
    """
    aspects = ["Environment", "Financial", "Governance", "SASB_Metrics", "Social"]
    for aspect in aspects:
        marker = "_" + aspect
        if marker in key:
            return key.split(marker)[0]
    return key  # fallback if no marker is found

def combine_cleaned_data(cleaned_data_long: dict) -> pd.DataFrame:
    """
    Combine multiple long-format DataFrames from different aspects of the company.
    
    Each DataFrame is augmented with:
      - An 'Aspect' column (derived from the composite key, e.g. "nextera_energy_SASB_Metrics", "tesla_Social", etc.)
      - A 'Company' column, extracted from the composite key using known aspects.
      - A 'UniqueMetric' column, which concatenates the Aspect and the original Metric to ensure uniqueness.
    
    Returns:
      A combined long-format DataFrame.
    """
    combined = []
    for key, df_long in cleaned_data_long.items():
        # key is composite, e.g. "nextera_energy_SASB_Metrics"
        aspect = key  
        company = extract_company_from_key(key)
        df_long_copy = df_long.copy()
        df_long_copy["Aspect"] = aspect
        df_long_copy["Company"] = company
        df_long_copy["UniqueMetric"] = df_long_copy["Aspect"] + "_" + df_long_copy["Metric"]
        combined.append(df_long_copy)
    combined_df = pd.concat(combined, ignore_index=True)
    return combined_df

def pivot_combined_data(combined_df: pd.DataFrame, index_cols: list = ["Year"]) -> pd.DataFrame:
    """
    Pivot the combined long-format DataFrame into wide format for modeling.
    
    The UniqueMetric column is used as the column names.
    
    Parameters:
      combined_df: The combined long-format DataFrame (from combine_cleaned_data).
      index_cols : Columns to use as the index for pivoting (e.g., "Year").
    
    Returns:
      A wide-format DataFrame with each unique metric as a separate column.
    """
    df_wide = combined_df.pivot_table(index=index_cols,
                                      columns="UniqueMetric",
                                      values="Value").reset_index()
    df_wide.columns.name = None
    return df_wide

# For testing the cleaning, pivoting, and combining process independently:
if __name__ == "__main__":
    sample_file = "data/nextera_energy/nextera energy - Financial.csv"  # Adjust as needed
    try:
        df = pd.read_csv(sample_file)
        print("Original DataFrame (tail):")
        print(df.tail(), "\n")
        
        df_long = clean_and_pivot_dataframe(df)
        print("Cleaned & Pivoted (Long Format) DataFrame (tail):")
        print(df_long.tail(), "\n")
        
        # Simulate combining multiple files by using the same df_long twice.
        combined_df = combine_cleaned_data({"Financial": df_long, "Environment": df_long})
        print("Combined Long Format DataFrame (tail):")
        print(combined_df.tail(), "\n")
        
        df_wide = pivot_combined_data(combined_df, index_cols=["Year"])
        print("Prepared Model Data (Wide Format) DataFrame (tail):")
        print(df_wide.tail())
    except Exception as e:
        print(f"Error loading or cleaning file: {e}")
