from groundwork.data_import import load_all_csvs
from groundwork.data_cleaning import (
    clean_and_pivot_dataframe,
    combine_cleaned_data,
    pivot_combined_data
)
from groundwork.important_metrics_analyzer import (
    fit_mle_model,
    calculate_vif,
    extract_importance_weights,
    check_residuals
)
from collections import Counter
import os

def main():
    print("\n========== LOADING CSV FILES ==========")
    data_frames = load_all_csvs()

    print("\n========== LOADED DATAFRAMES ==========")
    for key in data_frames:
        print(f"✔ Loaded: {key} ({len(data_frames[key])} rows)")
   
    print("\n========== CLEANING & PIVOTING DATAFRAMES ==========")
    # Process each file to a long-format DataFrame.
    cleaned_data_long = {}
    for key, df in data_frames.items():
        print(f"➡️  Cleaning & Pivoting: {key} ...", end=" ")
        original_rows = len(df)
        df_long = clean_and_pivot_dataframe(df)
        cleaned_rows = len(df_long)
        print(f"✔ Converted: {original_rows} rows -> {cleaned_rows} rows (long format)")
        cleaned_data_long[key] = df_long

    # Combine all long-format DataFrames from various aspects.
    print("\n========== COMBINING CLEANED DATA ==========")
    combined_long = combine_cleaned_data(cleaned_data_long)
    print(f"Combined long-format data has {combined_long.shape[0]} rows.")

    # Process data for each company.
    companies = combined_long["Company"].unique()
    print("\nFound companies:", companies)
    
    for comp in companies:
        print(f"\n========== PROCESSING COMPANY: {comp} ==========")
        # Filter for the current company.
        comp_data = combined_long[combined_long["Company"] == comp]
        # Pivot using both Company and Year as indices.
        df_wide = pivot_combined_data(comp_data, index_cols=["Company", "Year"])
        print(f"Wide-format data for {comp} has {df_wide.shape[0]} rows and {df_wide.shape[1]} columns.")

        # Optional: count metrics by aspect (ignoring "Company" and "Year").
        aspect_counts = Counter([col.split('_')[0] for col in df_wide.columns if col not in ["Company", "Year"]])
        print("Metric counts by aspect:")
        for aspect, count in aspect_counts.items():
            print(f"  {aspect}: {count} metrics")
        print()
        
        # Impute missing values via linear interpolation.
        df_wide.interpolate(method='linear', limit_direction='both', inplace=True)

        # Dynamically construct the response variable name.
        # For instance, if you want to model "GHG Emissions" from SASB Metrics,
        # the expected column would be "<company>_SASB_Metrics_GHG Emissions"
        response_variable = f"{comp}_SASB_Metrics_GHG Emissions"  # Adjust if needed.
        if response_variable not in df_wide.columns:
            print(f"❌ Response variable '{response_variable}' not found for {comp}. Skipping modeling.")
            continue

        predictor_vars = [col for col in df_wide.columns if col not in ["Company", "Year", response_variable]]
        missing_cols = [col for col in [response_variable] + predictor_vars if col not in df_wide.columns]
        if missing_cols:
            print(f"❌ Missing columns for modeling for {comp}: {missing_cols}")
            continue

        print("\n========== MODELING (IMPORTANT METRICS ANALYZER) ==========")
        results, selected_predictors, scaler = fit_mle_model(df_wide, response_variable, predictor_vars)
        
        # Calculate VIF on the selected predictors.
        vif_df = calculate_vif(df_wide, selected_predictors)
        print("\nVIF for Selected Predictors:")
        print(vif_df)
        
        # Extract and display variable importance weights.
        weight_dict = extract_importance_weights(results, selected_predictors)
        
        # Create folders for figures and results based on company.
        fig_folder = os.path.join("fig", comp)
        os.makedirs(fig_folder, exist_ok=True)
        resid_fig_path = os.path.join(fig_folder, f"{comp}_residual_plot.png")
        
        results_folder = os.path.join("results", comp)
        os.makedirs(results_folder, exist_ok=True)
        results_file = os.path.join(results_folder, f"{comp}_model_results.txt")
        
        # Save residual diagnostics plot.
        check_residuals(results, save_path=resid_fig_path)
        
        # Save model results (summary and variable importance) to a text file.
        with open(results_file, "w") as f:
            f.write(results.summary().as_text())
            f.write("\n\nSelected Predictors:\n")
            f.write(", ".join(selected_predictors))
            f.write("\n\nVariable Importance Weights:\n")
            for var, weight in weight_dict.items():
                f.write(f"{var}: {weight:.2f}\n")
        print(f"Modeling complete for {comp}.")
        print(f"Residual plot saved to: {resid_fig_path}")
        print(f"Model results saved to: {results_file}")
   
    print("\n========== DONE ==========")

if __name__ == "__main__":
    main()
