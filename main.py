import os
import pandas as pd

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
from ans_burning_qn1_and_2.duration_analyzer import analyze_duration
from utils.results_saver import save_model_results, setup_company_logger,save_duration_results
from pandasgui import show

def main():
    print("\n========== LOADING CSV FILES ==========")
    all_data_frames = load_all_csvs()

    metrics_data_frames = {key: df for key, df in all_data_frames.items() if "Targets" not in key}
    targets_data_frames = {key: df for key, df in all_data_frames.items() if "Targets" in key}

    print("\n========== LOADED DATAFRAMES ==========")
    for key in metrics_data_frames:
        print(f"✔ Metrics Loaded: {key} ({len(metrics_data_frames[key])} rows)")
    for key in targets_data_frames:
        print(f"✔ Targets Loaded: {key} ({len(targets_data_frames[key])} rows)")

    print(targets_data_frames["nextera_energy_Targets"])

    print("\n========== CLEANING & PIVOTING DATAFRAMES ==========")
    cleaned_data_long = {}
    for key, df in metrics_data_frames.items():
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

        # List available columns.
        columns = [col for col in df_wide.columns if col not in ["Company", "Year"]]
        logger.info("Columns for company %s: %s", comp, columns)

        df_wide.infer_objects(copy=False)
        df_wide.interpolate(method='linear', limit_direction='both', inplace=True)

        # --- Calculate Total Emissions ---
        scope1 = f"{comp}_SASB_Metrics_Scope 1 Emissions"
        scope2 = f"{comp}_SASB_Metrics_Scope 2 Emissions"
        scope3 = f"{comp}_SASB_Metrics_Scope 3 Emissions"
        if scope1 in df_wide.columns and scope2 in df_wide.columns and scope3 in df_wide.columns:
            df_wide[f"{comp}_SASB_Metrics_Total Emissions"] = (
                df_wide[scope1].fillna(0) + df_wide[scope2].fillna(0) + df_wide[scope3].fillna(0)
            )
            total_emissions = f"{comp}_SASB_Metrics_Total Emissions"
        else:
            logger.error("One or more of Scope 1, 2, 3 emissions columns not found for %s. Skipping.", comp)
            continue

        # Extract the unit from the comp_data DataFrame
        mask = comp_data["Metric"] == "Scope 1 Emissions"

        if mask.any():
            unit = comp_data.loc[mask, "Units"].iloc[0]
        else:
            unit = "error"

        logger.info("Using unit: %s", unit)

        # --- Use Targets Data ---
        target_key = f"{comp}_Targets"
        if target_key not in targets_data_frames:
            logger.error("Targets data not found for company %s. Skipping duration analysis.", comp)
            continue
        df_targets = targets_data_frames[target_key]


        logger.info("========== ANALYZING DURATION TO NET ZERO (DURATION ANALYZER) ==========")
        initial_forecast_tag = "initial"
        _, initial_net_zero_year = analyze_duration(comp, df_wide, df_targets, unit=unit, forecast_tag=initial_forecast_tag, logger=logger)
        duration_results = {
            "net_zero_year": initial_net_zero_year,
            "is_hit_targets": initial_net_zero_year is not None,
        }
        duration_results_file = os.path.join("results", comp, f"{comp}_duration_results.json")
        save_duration_results(duration_results, duration_results_file, forecast_tag=initial_forecast_tag, logger=logger)

        logger.info("========== MODELING (IMPORTANT METRICS ANALYZER) ==========")
        results, selected_predictors, scaler = fit_mle_model(df_wide, total_emissions, [col for col in df_wide.columns if col not in ["Company", "Year", total_emissions]], logger=logger)

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

        # Save the model results.
        save_model_results(results, selected_predictors, weight_dict, vif_df, results_file, logger=logger)
        logger.info("Model results saved to: %s", results_file)
        logger.info("Residual plot saved to: %s", resid_fig_path)

    print("\n========== DONE ==========")

if __name__ == "__main__":
    main()
