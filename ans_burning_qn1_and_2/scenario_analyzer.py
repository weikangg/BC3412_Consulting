import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging
from pandasgui import show
# Keep imports
from ans_burning_qn1_and_2.duration_analyzer import analyze_duration, parse_targets
from consts.consts import get_invert_flag
from groundwork.score_timeseries import compute_score_timeseries, plot_company_scores
from utils.results_saver import save_company_score_details
from utils.utils import extract_metric_name

NET_ZERO_THRESHOLD = 0.0 # Use consistent threshold

# --- Main Scenario Orchestrator ---
def run_phased_scenario(
    comp,
    total_glm_results, # Model for predicting final Total Emissions
    total_glm_features,# All predictors in Total Emissions model
    total_glm_scaler,
    scope_model_data,  # Dict: {'scope1': {'results':...}, 'scope2':...}
    base_df_wide,
    df_targets,
    phase_boundaries, # Dict: {'short': (start, end), 'medium': ..., 'long': ...}
    weight_dict,      # Weights for scoring (e.g., from Total Emissions model)
    unit,
    logger
):
    """
    Runs a "Target-Seeking" scenario simulation.

    1. Parses emission targets from df_targets.
    2. Builds scenario rules aiming to meet these targets by adjusting key metrics.
    3. Runs the simulation using these rules.
    4. Combines data and re-runs duration analysis.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    start_year = phase_boundaries['short'][0]
    end_year = phase_boundaries['long'][1]
    logger.info(f"[{comp}] Starting PHASED scenario analysis ({start_year}-{end_year})")

    # --- 1. Build PHASED Target-Seeking Rules (Get separate dicts) ---
    try:
        rules_short, rules_medium, rules_long, baseline_emissions_forecast_df = build_phased_target_seeking_rules(comp, scope_model_data, base_df_wide, df_targets, phase_boundaries, logger)
        if rules_short is None:  # Check if rule building failed overall
            logger.error(f"[{comp}] Failed to build phased rules.")
            return None, None, None, None, None
    except Exception as e:
            logger.error(f"[{comp}] Exception during phased rule building: {e}", exc_info=True)
            return None, None, None, None, None
    # --- 1b. Combine Rules Cumulatively ---
    logger.info(f"[{comp}] Combining phase rules cumulatively...")
    cumulative_scenario_rules = {}
    all_metrics_with_rules = set(rules_short.keys()) | set(rules_medium.keys()) | set(rules_long.keys())
    short_start, short_end = phase_boundaries['short']
    medium_start, medium_end = phase_boundaries['medium']
    long_start, long_end = phase_boundaries['long']

    last_rule_value = {}  # Track last applied rule % for each metric

    for year in range(start_year, end_year + 1):
        for metric in all_metrics_with_rules:
            rule_value_this_year = None
            # Check which phase the current year falls into
            if short_start <= year <= short_end:
                # Apply short-term rule directly if it exists
                rule_value_this_year = rules_short.get(metric, {}).get(year)
            elif medium_start <= year <= medium_end:
                # Apply medium-term rule if it exists, otherwise use last known rule
                rule_value_this_year = rules_medium.get(metric, {}).get(year, last_rule_value.get(metric))
            elif long_start <= year <= long_end:
                # Apply long-term rule if it exists, otherwise use last known rule
                rule_value_this_year = rules_long.get(metric, {}).get(year, last_rule_value.get(metric))

            # If a rule was found or propagated, store it and update last known value
            if rule_value_this_year is not None:
                cumulative_scenario_rules.setdefault(metric, {})[year] = rule_value_this_year
                last_rule_value[metric] = rule_value_this_year  # Update last value for propagation
            # If no rule found/propagated (should only happen if metric never had rule), implicitly 0

    # --- Fill Gaps for metrics that NEVER had rules ---
    all_scope_predictors = set()
    for scope in ['scope1', 'scope2', 'scope3']:
        model_info = scope_model_data.get(scope, {})
        if model_info.get('predictors'):
            all_scope_predictors.update(model_info['predictors'])
    # Also include predictors needed just by the TOTAL model if different,
    # as generate_scenario_predictors baseline logic might need them.
    if total_glm_features:
        all_scope_predictors.update(total_glm_features)

    all_predictors_for_gen = list(all_scope_predictors)
    if not all_predictors_for_gen:
        logger.error(f"[{comp}] No predictors found across any relevant models. Cannot generate scenario.")
        return None, None, None, None,None
    all_predictors_for_gen = list(all_scope_predictors)  # Get list as before
    all_years_list = list(range(start_year, end_year + 1))
    for metric in all_predictors_for_gen:
        cumulative_scenario_rules.setdefault(metric, {})  # Ensure metric dict exists
        for year in all_years_list:
            # Fill with 0 only if no rule exists for this year from calculation/propagation
            if year not in cumulative_scenario_rules[metric]:
                cumulative_scenario_rules[metric][year] = 0.0

    logger.info(f"[{comp}] Cumulative rules created for {len(cumulative_scenario_rules)} metrics.")

    try:
        # Pass the COMBINED cumulative_scenario_rules
        scenario_predictors_df = generate_scenario_predictors(
            comp, start_year, end_year, base_df_wide, cumulative_scenario_rules, all_predictors_for_gen, logger
        )
    except Exception as e:
        logger.error(f"[{comp}] Error generating phased scenario predictors: {e}", exc_info=True)
        return None, None, None, None,None

    # --- 3. Predict S1, S2, S3 Emissions Separately ---
    predicted_s1 = pd.Series(0.0, index=scenario_predictors_df.index)  # Default to 0 if model fails
    predicted_s2 = pd.Series(0.0, index=scenario_predictors_df.index)
    predicted_s3 = pd.Series(0.0, index=scenario_predictors_df.index)
    prediction_successful = True

    # Predict Scope 1
    s1_model_info = scope_model_data.get('scope1', {})
    if s1_model_info.get('results'):
        try:
            logger.info(f"[{comp}] Predicting Scope 1 emissions...")
            predicted_s1 = predict_emissions_from_glm(
                glm_results=s1_model_info['results'],
                scaler=s1_model_info['scaler'],
                original_predictors=s1_model_info['predictors'],  # Use S1 predictors
                df_predict=scenario_predictors_df  # Pass df with all predictors
            ).fillna(0)  # Fill potential NaNs from prediction failure with 0
        except Exception as e:
            logger.error(f"[{comp}] Error predicting SCOPE 1 emissions: {e}", exc_info=True)
            prediction_successful = False  # Mark failure
    else:
        logger.warning(f"[{comp}] Scope 1 model not available. Assuming 0 contribution.")

    # Predict Scope 2
    s2_model_info = scope_model_data.get('scope2', {})
    if s2_model_info.get('results') and prediction_successful:  # Proceed only if previous steps okay
        try:
            logger.info(f"[{comp}] Predicting Scope 2 emissions...")
            predicted_s2 = predict_emissions_from_glm(
                glm_results=s2_model_info['results'],
                scaler=s2_model_info['scaler'],
                original_predictors=s2_model_info['predictors'],  # Use S2 predictors
                df_predict=scenario_predictors_df
            ).fillna(0)

        except Exception as e:
            logger.error(f"[{comp}] Error predicting SCOPE 2 emissions: {e}", exc_info=True)
            prediction_successful = False
    else:
        logger.warning(f"[{comp}] Scope 2 model not available or prior prediction failed. Assuming 0 contribution.")

    # Predict Scope 3
    s3_model_info = scope_model_data.get('scope3', {})
    if s3_model_info.get('results') and prediction_successful:
        try:
            logger.info(f"[{comp}] Predicting Scope 3 emissions...")
            predicted_s3 = predict_emissions_from_glm(
                glm_results=s3_model_info['results'],
                scaler=s3_model_info['scaler'],
                original_predictors=s3_model_info['predictors'],  # Use S3 predictors
                df_predict=scenario_predictors_df
            ).fillna(0)

        except Exception as e:
            logger.error(f"[{comp}] Error predicting SCOPE 3 emissions: {e}", exc_info=True)
            prediction_successful = False
    else:
        logger.warning(f"[{comp}] Scope 3 model not available or prior prediction failed. Assuming 0 contribution.")

    # Check if any prediction failed critically
    if not prediction_successful:
        logger.error(f"[{comp}] Scenario failed due to error during scope emission prediction.")
        return None, None, None, None,None

    # --- 3b. Apply Phased Locking Logic ---
    logger.info(f"[{comp}] Applying phased 'locking' to prevent emission increases...")
    adjusted_s1 = predicted_s1.copy()
    adjusted_s2 = predicted_s2.copy()
    adjusted_s3 = predicted_s3.copy()  # Start with predicted values

    short_end_year = phase_boundaries['short'][1]
    medium_end_year = phase_boundaries['medium'][1]

    # Determine Lock Values (handle potential missing years or NaN predictions)
    s1_lock_value = np.nan
    if short_end_year in predicted_s1.index:
        s1_lock_value = predicted_s1.loc[short_end_year]
        if pd.isna(s1_lock_value):
            logger.warning(f"S1 lock value at {short_end_year} is NaN.")
        else:
            logger.info(f"S1 lock value (end of short phase {short_end_year}): {s1_lock_value:.3g}")
    else:
        logger.warning(f"S1 lock year {short_end_year} not in prediction index.")

    s2_lock_value = np.nan
    if medium_end_year in predicted_s2.index:
        s2_lock_value = predicted_s2.loc[medium_end_year]
        if pd.isna(s2_lock_value):
            logger.warning(f"S2 lock value at {medium_end_year} is NaN.")
        else:
            logger.info(f"S2 lock value (end of medium phase {medium_end_year}): {s2_lock_value:.3g}")
    else:
        logger.warning(f"S2 lock year {medium_end_year} not in prediction index.")

    # Apply Locks
    if pd.notna(s1_lock_value):
        # For years in medium and long phase
        for year in adjusted_s1.index[adjusted_s1.index > short_end_year]:
            if pd.notna(adjusted_s1.loc[year]) and adjusted_s1.loc[year] > s1_lock_value:
                logger.debug(f"Year {year}: Locking S1 {adjusted_s1.loc[year]:.3g} -> {s1_lock_value:.3g}")
                adjusted_s1.loc[year] = s1_lock_value

    if pd.notna(s2_lock_value):
        # For years in long phase only
        for year in adjusted_s2.index[adjusted_s2.index > medium_end_year]:
            if pd.notna(adjusted_s2.loc[year]) and adjusted_s2.loc[year] > s2_lock_value:
                logger.debug(f"Year {year}: Locking S2 {adjusted_s2.loc[year]:.3g} -> {s2_lock_value:.3g}")
                adjusted_s2.loc[year] = s2_lock_value

    # --- Calculate Scenario Total Emissions using ADJUSTED & fillna(0) scope values ---
    scenario_total_emissions = adjusted_s1.fillna(0) + adjusted_s2.fillna(0) + adjusted_s3.fillna(0)
    scenario_total_emissions = np.maximum(scenario_total_emissions, 0)
    scenario_predictors_df[f"{comp}_SASB_Metrics_Total Emissions"] = scenario_total_emissions
    # Optionally add predicted S1, S2, S3 as separate columns if needed later
    # scenario_predictors_df[f"{comp}_SASB_Metrics_Scope 1 Emissions_Predicted"] = predicted_s1
    # scenario_predictors_df[f"{comp}_SASB_Metrics_Scope 2 Emissions_Predicted"] = predicted_s2
    # scenario_predictors_df[f"{comp}_SASB_Metrics_Scope 3 Emissions_Predicted"] = predicted_s3
    logger.info(f"[{comp}] Calculated scenario Total Emissions by summing scope predictions.")

    # --- Determine ACTUAL last historical year BEFORE combining ---
    # Ensure 'Year' column exists in the original historical df
    if 'Year' not in base_df_wide.columns:
        if base_df_wide.index.name == 'Year':
            base_df_wide = base_df_wide.reset_index()  # Make Year a column if it's the index
        else:
            logger.error(f"[{comp}] 'Year' column/index not found in original base_df_wide.")
            return base_df_wide.copy(), None, None  # Return original if error
    actual_last_hist_year = int(base_df_wide["Year"].max())  # Convert to int just in case
    logger.info(f"Actual last historical year determined as: {actual_last_hist_year}")

    # --- 4. Combine Historical and Scenario Data ---
    scenario_predictors_df = scenario_predictors_df.reset_index()
    scenario_predictors_df["Company"] = comp

    historical_df = base_df_wide.copy()
    future_df = scenario_predictors_df[scenario_predictors_df["Year"] >= start_year].copy()

    common_cols = list(set(historical_df.columns) | set(future_df.columns)) # Use union to keep all columns
    historical_aligned = historical_df.reindex(columns=common_cols)
    future_aligned = future_df.reindex(columns=common_cols)

    combined_df = pd.concat([historical_aligned, future_aligned], ignore_index=True).sort_values("Year")
    combined_df.reset_index(drop=True, inplace=True)
    combined_df.interpolate(method='linear', limit_direction='both', inplace=True) # Interpolate potentially missing values

    logger.info(f"[{comp}] Combined historical and target-seeking scenario data created. Shape: {combined_df.shape}")

    # --- 5. Calculate Scores for the Combined Scenario Timeline ---
    logger.info(f"[{comp}] Calculating scores for the combined historical + scenario timeline...")
    scenario_scores_df = compute_score_timeseries(combined_df, weight_dict, logger=logger)

    # --- Extract detailed scores dictionary from the scenario_scores_df ---
    scenario_detailed_scores = {}
    if scenario_scores_df is not None and not scenario_scores_df.empty:
        logger.info(f"[{comp}] Extracting detailed scores from scenario results...")
        for _, row in scenario_scores_df.iterrows():
            year = int(row["Year"])  # Ensure year is int
            year_str = str(year)
            metrics_detail = {}
            for metric, w in weight_dict.items():
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

    # --- 6. Re-run Duration Analysis ---
    baseline_forecast_df_to_merge = baseline_emissions_forecast_df # Rename for clarity
    if baseline_forecast_df_to_merge is not None:
        emission_col = f"{comp}_SASB_Metrics_Total Emissions"
        baseline_col = f"{emission_col}_baseline_forecast"
        baseline_forecast_df = baseline_forecast_df_to_merge.rename(columns={emission_col: baseline_col})
        baseline_forecast_df_for_merge = baseline_forecast_df.reset_index(drop=True)
        if 'Year' not in baseline_forecast_df_for_merge.columns:
            logger.error(
                f"[{comp}] 'Year' column not found in baseline forecast after reset_index. Columns: {baseline_forecast_df_for_merge.columns}")
        elif baseline_col not in baseline_forecast_df_for_merge.columns:
            logger.error(
                f"[{comp}] Baseline forecast column '{baseline_col}' not found after rename. Columns: {baseline_forecast_df_for_merge.columns}")
        else:
            # Select only the needed columns for the merge
            cols_to_merge = ['Year', baseline_col]
            logger.info(f"Attempting to merge baseline forecast using columns: {cols_to_merge}")

            try:
                # Perform the merge
                combined_df = pd.merge(
                    combined_df,
                    baseline_forecast_df_for_merge[cols_to_merge],  # Use the reset index dataframe
                    on='Year',
                    how='left'
                )

                logger.info(f"Successfully merged baseline forecast column '{baseline_col}'.")
            except Exception as e:
                logger.error(f"[{comp}] Failed to merge baseline forecast data: {e}", exc_info=True)
                # Proceed without the baseline column if merge fails
                baseline_col = None  # Ensure it's None if merge failed
    else:
        logger.warning(f"[{comp}] No baseline forecast DataFrame available to merge.")

    logger.info(f"[{comp}] Running duration analysis on target-seeking scenario data...")
    scenario_ensemble_df, scenario_net_zero, scenario_final_emission = analyze_duration(
        comp, combined_df, df_targets, unit, "phased_scenario", logger,  # Use new tag
        NET_ZERO_THRESHOLD, baseline_col if baseline_col and baseline_col in combined_df.columns else None,
        actual_last_hist_year
    )

    # --- Return the rules, results, AND the calculated scenario scores ---
    logger.info(
        f"[{comp}] Phased Scenario analysis complete. Net Zero: {scenario_net_zero}, Final Emission: {scenario_final_emission}")
    return cumulative_scenario_rules, scenario_net_zero, scenario_final_emission, scenario_scores_df, scenario_detailed_scores

# In scenario_analyzer.py

# --- Core Rule Building Logic (CORRECTED VERSION) ---
def build_phased_target_seeking_rules(
    comp, scope_model_data, base_df_wide, df_targets, phase_boundaries, logger
):
    """
    Builds scenario rules (% change relative to no-change baseline)
    aiming to hit emission targets specified in df_targets.
    Returns the rules dict and the baseline TOTAL forecast emissions DataFrame.
    """
    rules_short = {}
    rules_medium = {}
    rules_long = {}
    baseline_emissions_df = None # Initialize emission baseline df

    start_year_overall = phase_boundaries['short'][0]
    end_year_overall = phase_boundaries['long'][1]
    emission_col_total = f"{comp}_SASB_Metrics_Total Emissions"
    last_hist_year = int(base_df_wide["Year"].max())

    # --- A. Parse & Interpolate TOTAL Targets ---
    baseline_emission_ref = base_df_wide.loc[base_df_wide['Year'] == last_hist_year, emission_col_total].iloc[0]
    if pd.isna(baseline_emission_ref) or baseline_emission_ref <= 0:
        logger.error(f"Invalid baseline emission ref in year {last_hist_year}.")
        return None, None
    logger.info(f"Using baseline emission ref from {last_hist_year}: {baseline_emission_ref:.2e}")
    target_values_abs, target_years = parse_targets(df_targets, baseline_emission_ref, logger)
    if target_values_abs is None: return None, None

    try: # Wrap interpolation in try-except
        target_series = pd.Series(target_values_abs)
        target_series.index = pd.to_datetime(target_series.index.astype(str) + '-12-31')
        all_years_dt = pd.to_datetime([f"{y}-12-31" for y in range(start_year_overall, end_year_overall + 1)])
        last_hist_dt = pd.to_datetime(f"{last_hist_year}-12-31")
        emission_series_for_interp = pd.Series({last_hist_dt: baseline_emission_ref})
        target_series = pd.concat([emission_series_for_interp, target_series])
        interpolated_targets = target_series.reindex(target_series.index.union(all_years_dt)).interpolate(method='time')
        targets_all_years = interpolated_targets.loc[all_years_dt] # Select only future years
        targets_all_years.index = targets_all_years.index.year
    except Exception as e:
        logger.error(f"Error interpolating targets: {e}", exc_info=True)
        return None, None

    # --- B. Baseline Forecast ---
    all_potential_predictors = set()
    for scope in ['scope1', 'scope2', 'scope3', 'total']:
        model_info = scope_model_data.get(scope, {})
        if model_info and model_info.get('predictors'):
            all_potential_predictors.update(model_info['predictors'])
    all_potential_predictors = list(all_potential_predictors)
    if not all_potential_predictors: logger.error(f"[{comp}] No predictors found."); return None, None

    logger.info("Generating baseline 'no-change' forecast...")
    baseline_rules_empty = {}
    try:
        # Generate the baseline PREDICTOR values (constant based on last history)
        baseline_predictors_df = generate_scenario_predictors(
            comp, start_year_overall, end_year_overall, base_df_wide,
            baseline_rules_empty, all_potential_predictors, logger
        )
        # Predict baseline TOTAL EMISSIONS
        temp_total_model = scope_model_data.get('total', {})
        if temp_total_model.get('results'):
            baseline_emissions_df = predict_emissions_from_glm(
                temp_total_model['results'], temp_total_model['scaler'],
                temp_total_model['predictors'], baseline_predictors_df
            ).to_frame(name=emission_col_total)
            baseline_emissions_df['Year'] = baseline_emissions_df.index # Add Year column
            logger.info("Baseline total emissions forecast generated.")
        else:
            logger.error("Total emissions model results missing for baseline forecast.")
            return None, None
    except Exception as e:
        logger.error(f"Failed to generate baseline forecast: {e}", exc_info=True)
        return None, None

    # --- C. Calculate Required Changes & Distribute ---
    for phase, (start_p, end_p) in phase_boundaries.items():
        logger.info(f"--- Building rules for Phase: {phase} ({start_p}-{end_p}) ---")
        # Determine which rule dictionary to populate
        if phase == 'short':
            current_phase_rules = rules_short
        elif phase == 'medium':
            current_phase_rules = rules_medium
        elif phase == 'long':
            current_phase_rules = rules_long
        else:
            logger.error(f"Unknown phase: {phase}"); continue
        phase_keys_list = list(phase_boundaries.keys())
        try: scope_index = phase_keys_list.index(phase); scope_key = f"scope{scope_index + 1}"
        except ValueError: logger.error(f"Invalid phase key '{phase}'. Skipping."); continue
        model_info = scope_model_data.get(scope_key)
        if model_info is None or model_info.get('results') is None: logger.warning(f"No valid model for {scope_key}. Skipping phase '{phase}'."); continue
        results_phase, predictors_phase, scaler_phase = model_info['results'], model_info['predictors'], model_info['scaler']
        coefs_phase = results_phase.params.drop("const", errors="ignore")
        # Handle scaler potentially being None if model fit failed partially
        scaler_scales_phase = pd.Series(scaler_phase.scale_, index=scaler_phase.feature_names_in_) if scaler_phase and hasattr(scaler_phase, 'scale_') else pd.Series()

        # ... (identify targetable_coefs_series_phase using consts.py logic - same) ...
        targetable_coefs_phase = {}
        for metric in predictors_phase:
            if metric not in coefs_phase.index: continue
            coef_value = coefs_phase[metric]

            if coef_value > 0:
                 targetable_coefs_phase[metric] = coef_value
        targetable_coefs_series_phase = pd.Series(targetable_coefs_phase)
        sum_targetable_abs_coefs_phase = abs(targetable_coefs_series_phase).sum()


        if sum_targetable_abs_coefs_phase <= 0: logger.info(f"No targetable metrics found for phase '{phase}'."); continue
        logger.info(f"Found {len(targetable_coefs_series_phase)} targetable metrics for phase '{phase}'.")

        # Iterate through years WITHIN this phase
        for year in range(start_p, end_p + 1):
            if year not in baseline_emissions_df.index or year not in targets_all_years.index: logger.warning(f"[{comp}-{year}] Skip rule gen: Missing baseline/target."); continue

            baseline_emission_y = baseline_emissions_df.loc[year, emission_col_total]
            target_emission_y = targets_all_years.loc[year]

            if baseline_emission_y <= target_emission_y:
                for metric in targetable_coefs_series_phase.index:
                    current_phase_rules.setdefault(metric, {})[year] = 0.0
                continue

            # Calculate required change in TOTAL linear predictor
            delta_linear_predictor_total = np.log(max(target_emission_y, 1e-9)) - np.log(max(baseline_emission_y, 1e-9))
            if delta_linear_predictor_total >= 0:
                for metric in targetable_coefs_series_phase.index:
                    current_phase_rules.setdefault(metric, {})[year] = 0.0
                continue

            # Distribute TOTAL delta across THIS PHASE'S targetable metrics
            for metric in targetable_coefs_series_phase.index:
                coef_m_phase = targetable_coefs_series_phase[metric]
                proportion = abs(coef_m_phase) / sum_targetable_abs_coefs_phase
                delta_X_scaled_m = delta_linear_predictor_total * proportion

                # Inverse Transform using THIS PHASE'S scaler scales
                scale_m_phase = scaler_scales_phase.get(metric) # Get scale factor
                if scale_m_phase is None or scale_m_phase == 0:
                    percentage_change_m = 0.0
                else:
                    delta_X_unscaled_m = delta_X_scaled_m * scale_m_phase
                    # Calculate % change relative to baseline predictor value for THIS year
                    baseline_predictor_y = baseline_predictors_df.loc[year, metric]
                    if pd.isna(baseline_predictor_y) or baseline_predictor_y == 0:
                        percentage_change_m = 0.0
                    else:
                        # We calculate the % change needed relative to baseline
                        # The cumulative function will apply this % change to the previous year's value
                        target_abs_value_m = max(baseline_predictor_y + delta_X_unscaled_m, 0.0)
                        percentage_change_m = (target_abs_value_m - baseline_predictor_y) / baseline_predictor_y

                # Store the calculated PERCENTAGE CHANGE rule
                current_phase_rules.setdefault(metric, {})[year] = percentage_change_m
                # logger.debug(f"[{comp} - {year}, Ph '{phase}'] Rule {metric}: {percentage_change_m*100:.2f}%")

    return rules_short, rules_medium, rules_long, baseline_emissions_df


# --- Predictor Generation (REVERTED to CUMULATIVE version) ---
def generate_scenario_predictors(comp, start_year, end_year, base_df, # base_df for initial values
                                 scenario_rules, final_predictors, logger=None):
                                 # baseline_predictors_df argument REMOVED
    """
    Generates predictor values year by year, applying rules cumulatively.
    Applies the year's change factor rule to the PREVIOUS year's simulated value.
    """
    if logger is None: logger = logging.getLogger(__name__)

    # Ensure 'Year' column exists and get last historical year/row
    if 'Year' not in base_df.columns:
        if base_df.index.name == 'Year': base_df = base_df.reset_index()
        else:
            logger.error(f"Base DataFrame must have a 'Year' column or index for {comp}.")
            raise ValueError("Missing 'Year' in base_df")
    last_hist_year = int(base_df["Year"].max())

    # Validate start/end years
    if start_year <= last_hist_year:
        logger.warning(f"[{comp}] Start year ({start_year}) <= last hist year ({last_hist_year}). Adjusting to {last_hist_year + 1}")
        start_year = last_hist_year + 1
    if start_year > end_year:
         logger.error(f"[{comp}] Start year ({start_year}) > end year ({end_year}). Cannot generate.")
         years_index = pd.RangeIndex(start_year, end_year + 1, name="Year")
         return pd.DataFrame(columns=final_predictors, index=years_index) # Return empty df

    last_row = base_df[base_df["Year"] == last_hist_year].iloc[0]

    # Initialize with the last known historical values
    # Use only predictors expected in the output list 'final_predictors'
    previous_year_values = {}
    for col in final_predictors:
         previous_year_values[col] = last_row.get(col, np.nan)
         if pd.isna(previous_year_values[col]):
              logger.warning(f"[{comp}] Predictor '{col}' has NaN value in last historical year ({last_hist_year}). Will propagate.")

    scenario_data = []
    years = range(start_year, end_year + 1)
    logger.info(f"[{comp}] Generating CUMULATIVE predictor values for years {start_year}-{end_year}...")

    for y in years:
        current_year_values = {}
        # logger.debug(f"--- Year {y} ---") # Optional detailed log
        for col in final_predictors: # Iterate through all predictors needed
            prev_val = previous_year_values.get(col) # Get value from previous year's calculation

            # Get % change rule for THIS year y and THIS column col
            # If no rule specified for this metric OR this year, change_factor is 0
            change_factor = scenario_rules.get(col, {}).get(y, 0.0)

            if pd.isna(prev_val):
                # Propagate NaN if previous value was NaN
                current_val = np.nan
            else:
                # Apply the change factor cumulatively to the *previous* year's value
                current_val = prev_val * (1.0 + change_factor)
                # Apply floor
                current_val = max(current_val, 0.0)
                # Log only if change applied
                # if change_factor != 0.0:
                #     logger.debug(f"Metric '{extract_metric_name(col)}': Prev={prev_val:.3g}, Change={change_factor*100:.2f}%, Current={current_val:.3g}")

            current_year_values[col] = current_val

        # Store results for this year
        row_dict_this_year = current_year_values.copy()
        row_dict_this_year["Year"] = y
        scenario_data.append(row_dict_this_year)

        # Update previous_year_values dictionary for the next iteration
        previous_year_values = current_year_values # CRITICAL step for cumulative logic

    # Create DataFrame
    scenario_df = pd.DataFrame(scenario_data).set_index("Year")

    # Final checks (same as before)
    missing_cols = [col for col in final_predictors if col not in scenario_df.columns]
    if missing_cols: logger.warning(f"[{comp}] Columns missing after generation: {missing_cols}")
    nan_cols = scenario_df.columns[scenario_df.isnull().any()].tolist()
    if nan_cols: logger.warning(f"[{comp}] Predictor columns have NaN values in CUMULATIVE output: {nan_cols}")

    logger.info(f"[{comp}] Generated CUMULATIVE scenario predictor DataFrame shape: {scenario_df.shape}")
    return scenario_df

# --- Emission Prediction (Add more checks) ---
def predict_emissions_from_glm(glm_results, scaler, original_predictors, df_predict):
    logger = logging.getLogger(__name__)

    # --- Add Input Checks ---
    if glm_results is None:
        logger.error("GLM results object is None. Cannot predict.")
        raise ValueError("GLM results object is None.")
    if scaler is None:
        logger.error("Scaler object is None. Cannot predict.")
        raise ValueError("Scaler object is None.")
    if not original_predictors: # Check if list is empty
        logger.warning("List of original predictors is empty. Prediction might be trivial (const only?).")
        # Allow continuing if only const exists? Or raise error? Let's raise error for safety.
        raise ValueError("original_predictors list is empty.")
    # -----------------------

    logger.info(f"Predicting emissions using model for response: {glm_results.model.endog_names}") # Log which model is used
    logger.info(f"Predicting for {len(df_predict)} future years...")

    # Ensure required columns exist in df_predict based on the specific model's predictors
    required_cols = [p for p in original_predictors if p in glm_results.params.index and p != 'const']
    logger.debug(f"Required predictors for this model: {required_cols}")

    missing_in_df = [col for col in required_cols if col not in df_predict.columns]
    if missing_in_df:
        logger.error(f"df_predict is missing required predictor columns for this model: {missing_in_df}")
        raise ValueError(f"df_predict is missing required predictor columns: {missing_in_df}")

    # Select only required columns + handle NaNs
    X_raw = df_predict[required_cols].copy()
    # ... (NaN handling: interpolation, mean imputation - same as before) ...
    if X_raw.isnull().any().any():
        logger.warning(f"NaNs found in predictors for years: {X_raw[X_raw.isnull().any(axis=1)].index.tolist()}. Interpolating.")
        X_raw.interpolate(method='linear', limit_direction='both', inplace=True, axis=0)
        if X_raw.isnull().any().any():
             nan_cols_remaining = X_raw.columns[X_raw.isnull().any()].tolist()
             logger.warning(f"NaNs remain after interpolation in: {nan_cols_remaining}. Imputing with mean.")
             # Impute NaNs - consider if mean is appropriate for all metrics
             for col in nan_cols_remaining:
                  col_mean = X_raw[col].mean()
                  if pd.isna(col_mean): col_mean = 0; logger.error(f"Mean for {col} is NaN, imputing with 0.")
                  X_raw[col].fillna(col_mean, inplace=True)


    # Scaling
    try:
        # Ensure scaler has necessary features
        if not hasattr(scaler, 'feature_names_in_'):
             logger.warning("Scaler does not have 'feature_names_in_'. Assuming columns match order.")
             cols_to_scale = required_cols # Use all required cols if no feature names
        else:
             cols_to_scale = [col for col in required_cols if col in scaler.feature_names_in_]

        if not cols_to_scale:
             # If no predictors need scaling (e.g., model only used const or non-numeric?)
             logger.warning(f"No columns required by model need scaling according to scaler features for {glm_results.model.endog_names}. Proceeding without scaling applied.")
             X_scaled = X_raw[cols_to_scale].copy() # Should be empty or just pass raw? Need X_scaled structure.
             # This case needs careful handling. If no scaling needed, structure for X_design needs care.
             # For now, assume scaling is usually needed. If required_cols exist but none are in scaler, it's an error.
             if required_cols and not cols_to_scale:
                  raise ValueError("Model requires predictors, but none are recognized by the scaler.")
             # If required_cols is empty, X_scaled will be empty df.
             if not required_cols:
                   X_scaled = pd.DataFrame(index=df_predict.index)


        # Check if X_raw is empty before transform
        elif X_raw.empty or X_raw[cols_to_scale].empty:
             logger.warning(f"Input data for scaling (X_raw or subset) is empty for {glm_results.model.endog_names}.")
             X_scaled = pd.DataFrame(columns=cols_to_scale, index=df_predict.index) # Create empty df with correct structure
        else:
             X_scaled_values = scaler.transform(X_raw[cols_to_scale])
             X_scaled = pd.DataFrame(X_scaled_values, columns=cols_to_scale, index=df_predict.index)

    except Exception as e:
        logger.error(f"Error applying scaler transform for {glm_results.model.endog_names}: {e}", exc_info=True)
        raise

    # Add constant and align columns
    X_design = sm.add_constant(X_scaled, has_constant='add')
    # Get expected columns from model params (excluding const)
    model_params_cols = [p for p in glm_results.params.index if p != 'const']
    # Ensure all required model parameter columns are present in X_design, add if missing (fill with 0?)
    # This handles cases where a predictor was constant historically but needed by model.
    cols_for_pred = ['const'] + model_params_cols
    X_design_aligned = X_design.reindex(columns=cols_for_pred, fill_value=0) # Fill missing predictors with 0 after scaling
    logger.debug(f"Columns for prediction matrix: {X_design_aligned.columns.tolist()}")


    # Predict
    try:
        linear_predictor = X_design_aligned @ glm_results.params
    # ... (rest of prediction: exp, floor, logging, return - same) ...
    except Exception as e:
        logger.error(f"Error calculating linear predictor for {glm_results.model.endog_names}: {e}", exc_info=True)
        raise
    pred_emissions = np.exp(linear_predictor)
    pred_emissions = np.maximum(pred_emissions, 1e-9)
    logger.info(f"Prediction complete for {glm_results.model.endog_names}. Range: {pred_emissions.min():.2e} to {pred_emissions.max():.2e}")
    return pd.Series(pred_emissions, index=df_predict.index, name="Predicted_Emissions")