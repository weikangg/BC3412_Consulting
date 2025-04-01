import os
import pandas as pd

from groundwork.data_cleaning import (
    pivot_combined_data
)
from groundwork.important_metrics_analyzer import (
    fit_mle_model,
    calculate_vif,
    extract_importance_weights,
    check_residuals
)
from groundwork.score_timeseries import compute_score_timeseries, plot_company_scores
from ans_burning_qn1_and_2.duration_analyzer import analyze_duration
from utils.results_saver import (
    save_model_results,
    setup_company_logger,
    save_duration_results,
    save_company_score_details
)
from utils.utils import extract_metric_unit

def process_company(
    comp,
    combined_long,
    targets_data_frames,
    industry_yearly_scores
):
    """
    Encapsulates the logic for processing a single company:
      - pivot & clean data
      - run duration analysis
      - model & get importance weights
      - compute per-year scores
      - update industry_yearly_scores
      - save results

    Parameters
    ----------
    comp : str
        Company name
    combined_long : pd.DataFrame
        The combined long-format data
    targets_data_frames : dict
        Dictionary of target DataFrames keyed by {company}_Targets
    industry_yearly_scores : dict
        Global dictionary mapping Year -> list of overall scores across companies

    Returns
    -------
    None
        (All results are saved to disk and global structures updated in place.)
    """
    logger = setup_company_logger(comp)
    logger.info("========== PROCESSING COMPANY: %s ==========", comp)

    comp_data = combined_long[combined_long["Company"] == comp]
    df_wide = pivot_combined_data(comp_data, index_cols=["Company", "Year"])

    logger.info(
        "Wide-format data for %s has %d rows and %d columns.",
        comp, df_wide.shape[0], df_wide.shape[1]
    )

    columns = [col for col in df_wide.columns if col not in ["Company", "Year"]]
    logger.info("Columns for company %s: %s", comp, columns)

    # Clean up
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
        return

    # Extract the unit from the comp_data DataFrame
    unit = extract_metric_unit("Scope 1 Emissions", comp_data, logger)

    # --- Use Targets Data ---
    target_key = f"{comp}_Targets"
    if target_key not in targets_data_frames:
        logger.error("Targets data not found for company %s. Skipping duration analysis.", comp)
        return
    df_targets = targets_data_frames[target_key]

    logger.info("========== ANALYZING DURATION TO NET ZERO (DURATION ANALYZER) ==========")
    initial_forecast_tag = "initial"
    _, initial_net_zero_year = analyze_duration(
        comp, df_wide, df_targets,
        unit=unit,
        forecast_tag=initial_forecast_tag,
        logger=logger
    )
    duration_results = {
        "net_zero_year": initial_net_zero_year,
        "is_hit_targets": initial_net_zero_year is not None,
    }
    duration_results_file = os.path.join("results", comp, f"{comp}_duration_results.json")
    save_duration_results(duration_results, duration_results_file, forecast_tag=initial_forecast_tag, logger=logger)

    logger.info("========== MODELING (IMPORTANT METRICS ANALYZER) ==========")
    predictors = [col for col in df_wide.columns if col not in ["Company", "Year", total_emissions]]
    results, selected_predictors, scaler = fit_mle_model(df_wide, total_emissions, predictors, logger=logger)

    vif_df = calculate_vif(df_wide, selected_predictors)
    logger.info("VIF for Selected Predictors:\n%s", vif_df)

    weight_dict = extract_importance_weights(results, selected_predictors, logger=logger)

    # Plot residuals and save model results.
    fig_folder = os.path.join("fig", comp)
    os.makedirs(fig_folder, exist_ok=True)
    resid_fig_path = os.path.join(fig_folder, f"{comp}_residual_plot.png")

    results_folder = os.path.join("results", comp)
    os.makedirs(results_folder, exist_ok=True)
    results_file = os.path.join(results_folder, f"{comp}_model_weights_results.json")

    check_residuals(results, save_path=resid_fig_path, logger=logger)
    save_model_results(results, selected_predictors, weight_dict, vif_df, results_file, logger=logger)
    logger.info("Model results saved to: %s", results_file)
    logger.info("Residual plot saved to: %s", resid_fig_path)

    # --- Calculate Scores for Each Year ---
    logger.info("========== CALCULATING (SCORE OF COMPANY EACH YEAR) ==========")
    company_scores_df = compute_score_timeseries(df_wide, weight_dict, logger=logger)
    comp_scores_fig_path = os.path.join(fig_folder, f"{comp}_comp_scores_plot.png")
    plot_company_scores(company_scores_df, comp, comp_scores_fig_path, logger=logger)

    # Build a detailed score dictionary and update global industry scores
    detailed_scores = {}  # key: year -> {"metrics": {...}, "overall_score": ...}
    for _, row in company_scores_df.iterrows():
        year = row["Year"]
        year_str = str(year)

        # Build dictionary of metric scores
        metrics_detail = {}
        for metric, w in weight_dict.items():
            score_col = f"{metric}_score"
            metrics_detail[metric] = {
                "score": row.get(score_col, 50),
                "weight": w
            }

        detailed_scores[year_str] = {
            "metrics": metrics_detail,
            "overall_score": row["overall_score"]
        }

        # Update global industry scores
        if year not in industry_yearly_scores:
            industry_yearly_scores[year] = []
        industry_yearly_scores[year].append(row["overall_score"])

    # Save the per-year scores for the company:
    save_company_score_details(comp, detailed_scores, logger=logger)

