import os
import pandas as pd

from ans_burning_qn1_and_2.scenario_analyzer import run_dynamic_scenario
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
    save_company_score_details, save_scenario_rules
)
from utils.utils import extract_metric_unit, get_max_year_from_df


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
    _, initial_net_zero_year, final_year_emission = analyze_duration(
        comp, df_wide, df_targets,
        unit=unit,
        forecast_tag=initial_forecast_tag,
        logger=logger
    )
    duration_results = {
        "net_zero_year": initial_net_zero_year,
        "is_hit_targets": initial_net_zero_year is not None,
        "final_year_emission": final_year_emission,
        "emission_unit": unit
    }
    duration_results_file = os.path.join("results", comp, f"{comp}_duration_results.json")
    save_duration_results(duration_results, duration_results_file, forecast_tag=initial_forecast_tag, logger=logger)

    logger.info("========== MODELING (IMPORTANT METRICS ANALYZER) ==========")
    predictors = [col for col in df_wide.columns if col not in ["Company", "Year", total_emissions]]
    results, selected_predictors, scaler = fit_mle_model(df_wide, total_emissions, predictors, logger=logger)

    vif_df = calculate_vif(df_wide, selected_predictors)
    logger.info("VIF for Selected Predictors:\n%s", vif_df)

    weight_dict = extract_importance_weights(results, selected_predictors, logger=logger)

    # --- Run Target-Seeking Scenario ---
    scenario_rules = None  # Initialize
    scenario_net_zero = None  # Initialize
    scenario_final_emission = None  # Initialize
    scenario_detailed_scores = None

    if df_targets is None:
        logger.warning(f"[{comp}] Skipping target-seeking scenario analysis because targets data is missing.")
        scenario_net_zero = None
    else:
        logger.info("\n========== RUNNING TARGET-SEEKING SCENARIO ==========")
        start_year = df_wide["Year"].max() + 1
        # Determine end_year based on target years
        target_years = sorted([int(col.strip()) for col in df_targets.columns if col.strip().isdigit()])
        if not target_years:
            logger.error(f"[{comp}] No valid target years found in Targets data. Cannot run target-seeking scenario.")
            scenario_net_zero = None
        else:
            end_year = target_years[-1]  # Simulate up to the final target year
            logger.info(f"Target-seeking scenario range: {start_year} to {end_year}")

            scenario_rules, scenario_net_zero, scenario_final_emission,scenario_detailed_scores  = run_dynamic_scenario(
                glm_results=results,
                glm_features=selected_predictors,
                glm_scaler=scaler,
                comp=comp,
                base_df_wide=df_wide,
                df_targets=df_targets,  # Pass the targets dataframe
                weight_dict=weight_dict,
                start_year=start_year,
                end_year=end_year,
                unit=unit,
                logger=logger
            )

    # --- Save Scenario Results ---
    if scenario_rules  is not None:
        logger.info(
            f"[{comp}] Scenario analysis completed. Net Zero: {scenario_net_zero}, Final Emission: {scenario_final_emission}")

        # Save the scenario rules
        rules_file = os.path.join("results", comp, f"{comp}_scenario_rules.json")
        save_scenario_rules(scenario_rules, rules_file, logger=logger)  # Call new save function

        # Save scenario duration results including final emission
        scenario_duration_results = {
            "net_zero_year": scenario_net_zero,
            "is_hit_targets": scenario_net_zero is not None,
            "final_year_emission": scenario_final_emission,
            "emission_unit": unit  # Good to store the unit too
        }
        save_duration_results(
            scenario_duration_results,
            duration_results_file,  # Append to the same duration file
            forecast_tag="target_seeking_scenario",  # Use specific tag
            logger=logger
        )

        if scenario_detailed_scores is not None:
            save_company_score_details(comp, scenario_detailed_scores, tag="target_seeking_scenario", logger=logger)
        else:
            logger.warning(f"[{comp}] No scenario scores were calculated or returned. Skipping save.")
    else:
        logger.warning(f"[{comp}] Scenario analysis was skipped or failed. No scenario results saved.")

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

    return df_wide, company_scores_df, weight_dict


