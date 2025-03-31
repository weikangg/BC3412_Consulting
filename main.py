import os
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
from utils.results_saver import save_model_results, setup_company_logger  # Updated import

def main():
    # Top-level logging for overall progress can go to the console.
    print("\n========== LOADING CSV FILES ==========")
    data_frames = load_all_csvs()

    print("\n========== LOADED DATAFRAMES ==========")
    for key in data_frames:
        print(f"✔ Loaded: {key} ({len(data_frames[key])} rows)")

    print("\n========== CLEANING & PIVOTING DATAFRAMES ==========")
    cleaned_data_long = {}
    for key, df in data_frames.items():
        print(f"➡️  Cleaning & Pivoting: {key} ...", end=" ")
        original_rows = len(df)
        df_long = clean_and_pivot_dataframe(df)
        cleaned_rows = len(df_long)
        print(f"✔ Converted: {original_rows} rows -> {cleaned_rows} rows (long format)")
        cleaned_data_long[key] = df_long

    print("\n========== COMBINING CLEANED DATA ==========")
    combined_long = combine_cleaned_data(cleaned_data_long)
    print(f"Combined long-format data has {combined_long.shape[0]} rows.")

    companies = combined_long["Company"].unique()
    print("\nFound companies:", companies)

    for comp in companies:
        # Set up a logger for this company.
        logger = setup_company_logger(comp)
        logger.info("========== PROCESSING COMPANY: %s ==========", comp)

        comp_data = combined_long[combined_long["Company"] == comp]
        df_wide = pivot_combined_data(comp_data, index_cols=["Company", "Year"])
        logger.info("Wide-format data for %s has %d rows and %d columns.", comp, df_wide.shape[0], df_wide.shape[1])

        # List and log available columns (excluding 'Company' and 'Year').
        columns = [col for col in df_wide.columns if col not in ["Company", "Year"]]
        logger.info("Columns for company %s: %s", comp, columns)

        df_wide.interpolate(method='linear', limit_direction='both', inplace=True)

        response_variable = f"{comp}_SASB_Metrics_GHG Emissions"  # Adjust if needed.
        if response_variable not in df_wide.columns:
            logger.error("❌ Response variable '%s' not found for %s. Skipping modeling.", response_variable, comp)
            continue

        predictor_vars = [col for col in df_wide.columns if col not in ["Company", "Year", response_variable]]
        missing_cols = [col for col in [response_variable] + predictor_vars if col not in df_wide.columns]
        if missing_cols:
            logger.error("❌ Missing columns for modeling for %s: %s", comp, missing_cols)
            continue

        logger.info("========== MODELING (IMPORTANT METRICS ANALYZER) ==========")
        results, selected_predictors, scaler = fit_mle_model(df_wide, response_variable, predictor_vars, logger=logger)

        vif_df = calculate_vif(df_wide, selected_predictors)
        logger.info("VIF for Selected Predictors:\n%s", vif_df)

        weight_dict = extract_importance_weights(results, selected_predictors, logger=logger)

        fig_folder = os.path.join("fig", comp)
        os.makedirs(fig_folder, exist_ok=True)
        resid_fig_path = os.path.join(fig_folder, f"{comp}_residual_plot.png")

        results_folder = os.path.join("results", comp)
        os.makedirs(results_folder, exist_ok=True)
        results_file = os.path.join(results_folder, f"{comp}_model_results.json")

        check_residuals(results, save_path=resid_fig_path, logger=logger)

        # Save the model results using our structured JSON format.
        save_model_results(results, selected_predictors, weight_dict, vif_df, results_file, logger=logger)
        logger.info("Model results saved to: %s", results_file)
        logger.info("Residual plot saved to: %s", resid_fig_path)

    print("\n========== DONE ==========")


if __name__ == "__main__":
    main()
