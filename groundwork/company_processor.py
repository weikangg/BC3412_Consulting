import os
import pandas as pd
from pandasgui import show

from ans_burning_qn1_and_2.scenario_analyzer import run_phased_scenario
from ans_burning_qn3 import risk_analyzer
from groundwork.data_cleaning import (
    pivot_combined_data
)
from groundwork.important_metrics_analyzer import (
    fit_mle_model,
    calculate_vif,
    extract_importance_weights,
    check_residuals
)
from groundwork.recommendations_formatter import compile_company_results, build_recommendation_for_company
from groundwork.score_timeseries import compute_score_timeseries, plot_company_scores, combine_scope_weights

from ans_burning_qn1_and_2.duration_analyzer import analyze_duration

from utils.results_saver import (
    save_model_results,
    setup_company_logger,
    save_duration_results,
    save_company_score_details,
    save_phased_scenario_rules, save_individual_model_outputs, save_individual_recommendation
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

    logger.info("========== MODELING (TOTAL & SCOPES 1, 2 and 3) ==========")
    predictors = [col for col in df_wide.columns if col not in ["Company", "Year", total_emissions, scope1, scope2, scope3]]
    try:
        logger.info("--- Fitting model for Total Emissions ---")
        results_total, predictors_total, scaler_total = fit_mle_model(df_wide, total_emissions, predictors,
                                                                      logger=logger)
        vif_df = calculate_vif(df_wide, predictors_total)
        logger.info("VIF for Selected Predictors:\n%s", vif_df)
        weights_total = extract_importance_weights(results_total, predictors_total, logger=logger)
        save_individual_model_outputs(comp, "total", results_total, predictors_total, weights_total, vif_df, logger)

    except Exception as e:
        logger.error(f"Failed to model Total Emissions: {e}", exc_info=True)
        results_total, predictors_total, scaler_total, weights_total = None, [], None, {}  # Handle failure

    # --- Model Scope 1 ---
    try:
        logger.info("--- Fitting model for Scope 1 Emissions ---")
        results_s1, predictors_s1, scaler_s1 = fit_mle_model(df_wide, scope1, predictors, logger=logger)
        weights_s1 = extract_importance_weights(results_s1, predictors_s1, logger=logger)
        vif_s1 = calculate_vif(df_wide, predictors_s1)
        save_individual_model_outputs(comp, "scope1", results_s1, predictors_s1, weights_s1, vif_s1, logger)
    except Exception as e:
        logger.error(f"Failed to model Scope 1 Emissions: {e}", exc_info=True)
        results_s1, predictors_s1, scaler_s1, weights_s1 = None, [], None, {}

    # --- Model Scope 2 ---
    try:
        logger.info("--- Fitting model for Scope 2 Emissions ---")
        results_s2, predictors_s2, scaler_s2 = fit_mle_model(df_wide, scope2, predictors, logger=logger)
        weights_s2 = extract_importance_weights(results_s2, predictors_s2, logger=logger)
        vif_s2 = calculate_vif(df_wide, predictors_s2)
        save_individual_model_outputs(comp, "scope2", results_s2, predictors_s2, weights_s2, vif_s2, logger)
    except Exception as e:
        logger.error(f"Failed to model Scope 2 Emissions: {e}", exc_info=True)
        results_s2, predictors_s2, scaler_s2, weights_s2 = None, [], None, {}

    # --- Model Scope 3 ---
    try:
        logger.info("--- Fitting model for Scope 3 Emissions ---")
        results_s3, predictors_s3, scaler_s3 = fit_mle_model(df_wide, scope3, predictors, logger=logger)
        weights_s3 = extract_importance_weights(results_s3, predictors_s3, logger=logger)
        vif_s3 = calculate_vif(df_wide, predictors_s3)
        save_individual_model_outputs(comp, "scope3", results_s3, predictors_s3, weights_s3, vif_s3, logger)
    except Exception as e:
        logger.error(f"Failed to model Scope 3 Emissions: {e}", exc_info=True)
        results_s3, predictors_s3, scaler_s3, weights_s3 = None, [], None, {}

    # --- Run PHASED Target-Seeking Scenario ---
    phased_scenario_rules = None
    scenario_net_zero = None
    scenario_final_emission = None
    scenario_scores_df = None
    scenario_detailed_scores = None
    scenario_weight_dict = combine_scope_weights(weights_total, weights_s1, weights_s2, weights_s3)

    if df_targets is None:
        logger.warning(f"[{comp}] Skipping phased scenario: targets data missing.")
    elif results_total is None:  # Need the total model for prediction
        logger.warning(f"[{comp}] Skipping phased scenario: Total Emissions model failed.")
    else:
        logger.info("\n========== RUNNING PHASED TARGET-SEEKING SCENARIO ==========")
        start_year = int(df_wide["Year"].max()) + 1
        target_years = sorted([int(col.strip()) for col in df_targets.columns if col.strip().isdigit()])
        if not target_years:
            logger.error(f"[{comp}] No valid target years. Cannot run phased scenario.")
        else:
            end_year = target_years[-1]
            # --- Define Phases (Example: Equal Split) ---
            total_duration = end_year - start_year + 1
            phase_len = total_duration // 3
            short_end_year = start_year + phase_len - 1
            medium_end_year = short_end_year + phase_len
            # Ensure phases cover the whole duration
            if total_duration % 3 != 0: medium_end_year += (total_duration % 3) - 1

            phase_boundaries = {
                'short': (start_year, short_end_year),
                'medium': (short_end_year + 1, medium_end_year),
                'long': (medium_end_year + 1, end_year)
            }
            logger.info(f"Phased scenario range: {start_year}-{end_year}")
            logger.info(
                f"Phase Boundaries: Short={phase_boundaries['short']}, Medium={phase_boundaries['medium']}, Long={phase_boundaries['long']}")

            # --- Collate model results for passing ---
            scope_model_data = {
                'scope1': {'results': results_s1, 'predictors': predictors_s1, 'scaler': scaler_s1},
                'scope2': {'results': results_s2, 'predictors': predictors_s2, 'scaler': scaler_s2},
                'scope3': {'results': results_s3, 'predictors': predictors_s3, 'scaler': scaler_s3},
                'total': {'results': results_total, 'predictors': predictors_total, 'scaler': scaler_total},
            }

            # --- Call the new phased scenario function ---
            phased_scenario_rules, scenario_net_zero, scenario_final_emission, scenario_scores_df, scenario_detailed_scores = run_phased_scenario(
                comp=comp,
                # Pass Total Emission model for final prediction
                total_glm_results=results_total,
                total_glm_features=predictors_total,  # All predictors considered by total model
                total_glm_scaler=scaler_total,
                # Pass individual scope models for rule building
                scope_model_data=scope_model_data,
                base_df_wide=df_wide.copy(),
                df_targets=df_targets,
                phase_boundaries=phase_boundaries,  # Pass phase years
                weight_dict=weights_total,  # Pass total weights for scoring
                unit=unit,
                logger=logger
            )

    # --- Save Phased Scenario Results ---
    if phased_scenario_rules is not None:
        logger.info(
            f"[{comp}] Phased Scenario analysis completed. Net Zero: {scenario_net_zero}, Final Emission: {scenario_final_emission}")
        # Save phased rules (using a new or modified save function)
        phased_rules_file = os.path.join("results", comp, f"{comp}_phased_scenario_rules.json")
        save_phased_scenario_rules(phased_scenario_rules, phased_rules_file, logger=logger)  # New save function

        # Save final duration results
        scenario_duration_results = {
            "net_zero_year": scenario_net_zero,
            "is_hit_targets": scenario_net_zero is not None,
            "final_year_emission": scenario_final_emission,
            "emission_unit": unit
        }
        save_duration_results(scenario_duration_results, duration_results_file, forecast_tag="phased_scenario",
                              logger=logger)  # Use new tag

        # Save final scenario scores
        if scenario_detailed_scores is not None:
            fig_folder = os.path.join("fig", comp)
            comp_scores_fig_path = os.path.join(fig_folder, f"{comp}_scenario_rules_comp_scores_plot.png")
            plot_company_scores(scenario_scores_df, comp, comp_scores_fig_path, logger=logger)
            save_company_score_details(str(comp), scenario_detailed_scores, tag="phased_scenario",
                                       logger=logger)  # Use new tag
        else:
            logger.warning(f"[{comp}] No phased scenario scores calculated.")
    else:
        logger.warning(f"[{comp}] Phased scenario analysis skipped or failed.")

    fig_folder = os.path.join("fig", comp)
    os.makedirs(fig_folder, exist_ok=True)

    # --- Calculate Scores for Each Year ---
    logger.info("========== CALCULATING HISTORICAL SCORES (Based on S1, S2, S3 Emission Weights) ==========")

    comp_scores_fig_path = os.path.join(fig_folder, f"{comp}_historical_comp_scores_plot.png")

    scenario_scores_df = compute_score_timeseries(df_wide, scenario_weight_dict, logger=logger)
    plot_company_scores(scenario_scores_df, comp, comp_scores_fig_path, logger=logger)
    # --- Extract detailed scores dictionary from the scenario_scores_df ---
    scenario_detailed_scores = {}
    if scenario_scores_df is not None and not scenario_scores_df.empty:
        logger.info(f"[{comp}] Extracting detailed scores from scenario results...")
        for _, row in scenario_scores_df.iterrows():
            year = int(row["Year"])  # Ensure year is int
            year_str = str(year)
            metrics_detail = {}
            for metric, w in scenario_weight_dict.items():
                score_col = f"{metric}_score"
                # Use .get with default 50 for safety if score column missing
                metrics_detail[metric] = {
                    "score": row.get(score_col, 50.0),  # Ensure float
                    "weight": w
                }
            scenario_detailed_scores[year_str] = {
                "metrics": metrics_detail,
                "overall_score": row.get("overall_score", 50.0)  # Ensure float and default
            }
        logger.info(f"[{comp}] Extracted scenario scores for {len(scenario_detailed_scores)} years.")
    else:
        logger.warning(
            f"[{comp}] compute_score_timeseries did not return valid results for scenario. Cannot extract scores.")
        scenario_detailed_scores = None  # Indicate failure
    save_company_score_details(comp, scenario_detailed_scores, tag="historical",
                               logger=logger)

    # Build a detailed score dictionary and update global industry scores
    detailed_scores = {}  # key: year -> {"metrics": {...}, "overall_score": ...}
    for _, row in scenario_scores_df.iterrows():
        year = row["Year"]
        year_str = str(year)

        # Build dictionary of metric scores
        metrics_detail = {}
        for metric, w in weights_total.items():
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

    logger.info(f"========== BUILDING RECOMMENDATION SUMMARY for {comp} ==========")
    company_results_folder = os.path.join("results", comp)
    try:
        # 1. Load all necessary JSON results just saved for this company
        comp_results_data = compile_company_results(company_results_folder)

        # 2. Build the recommendation summary
        recommendation_summary = build_recommendation_for_company(comp_results_data)

        # 3. Save the individual summary
        save_individual_recommendation(comp, recommendation_summary, logger)

    except Exception as e:
        logger.error(f"[{comp}] Failed to build or save individual recommendation summary: {e}", exc_info=True)

    return df_wide, scenario_scores_df, weights_total


