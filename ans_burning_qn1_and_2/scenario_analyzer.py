import numpy as np
import pandas as pd
import statsmodels.api as sm
import logging
from pandasgui import show
# Keep imports
from ans_burning_qn1_and_2.duration_analyzer import analyze_duration, parse_targets
from consts.consts import get_invert_flag
from groundwork.score_timeseries import compute_score_timeseries
from utils.utils import extract_metric_name

NET_ZERO_THRESHOLD = 0.01 # Use consistent threshold

# --- Main Scenario Orchestrator ---
def run_dynamic_scenario(
    glm_results,
    glm_features,
    glm_scaler,
    comp,
    base_df_wide,
    df_targets, # Required for this approach
    weight_dict,
    start_year,
    end_year,
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

    logger.info(f"[{comp}] Starting Target-Seeking scenario analysis ({start_year}-{end_year})")

    # --- 1. Build Target-Seeking Rules ---
    try:
        scenario_rules, baseline_forecast_df = build_target_seeking_rules(
            comp=comp,
            glm_results=glm_results,
            glm_scaler=glm_scaler,
            final_predictors=glm_features,
            base_df_wide=base_df_wide,
            df_targets=df_targets,
            start_year=start_year,
            end_year=end_year,
            logger=logger
        )
        if scenario_rules is None:
             logger.error(f"[{comp}] Failed to build target-seeking rules. Aborting scenario.")
             return None, None, None
    except Exception as e:
        logger.error(f"[{comp}] Exception during rule building: {e}", exc_info=True)
        return None, None, None


    # --- 2. Generate Future Predictor Values using Rules ---
    try:
        scenario_predictors_df = generate_scenario_predictors(
            comp=comp,
            start_year=start_year,
            end_year=end_year,
            base_df=base_df_wide,
            scenario_rules=scenario_rules,
            final_predictors=glm_features,
            logger=logger
        )
    except Exception as e:
         logger.error(f"[{comp}] Error generating scenario predictors: {e}", exc_info=True)
         return None, None, None

    # --- 3. Predict Emissions under Scenario ---
    try:
        scenario_emissions = predict_emissions_from_glm(
            glm_results=glm_results,
            scaler=glm_scaler,
            original_predictors=glm_features,
            df_predict=scenario_predictors_df
        )
        scenario_predictors_df[f"{comp}_SASB_Metrics_Total Emissions"] = scenario_emissions
    except Exception as e:
        logger.error(f"[{comp}] Error predicting scenario emissions: {e}", exc_info=True)
        return None, None, None

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

    # --- 5. Re-run Duration Analysis ---
    # Add baseline forecast to combined_df for plotting comparison
    if baseline_forecast_df is not None:
        emission_col = f"{comp}_SASB_Metrics_Total Emissions"
        baseline_col = f"{emission_col}_baseline_forecast"
        baseline_forecast_df = baseline_forecast_df.rename(columns={emission_col: baseline_col})
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
    _, scenario_net_zero,scenario_final_emission  = analyze_duration(
        comp,
        combined_df,
        df_targets, # Pass targets for plotting comparison
        unit=unit,
        forecast_tag="target_seeking_scenario",
        logger=logger,
        threshold_ratio=NET_ZERO_THRESHOLD,
        baseline_forecast_col = baseline_col if baseline_col and baseline_col in combined_df.columns else None,
        actual_last_hist_year = actual_last_hist_year
    )

    logger.info(f"[{comp}] Target-Seeking Scenario analysis complete. Projected Net Zero Year: {scenario_net_zero}")

    # --- Return the rules and the key results ---
    return scenario_rules, scenario_net_zero, scenario_final_emission


# --- Core Rule Building Logic ---
def build_target_seeking_rules(
    comp, glm_results, glm_scaler, final_predictors, base_df_wide, df_targets, start_year, end_year, logger
):
    """
    Builds scenario rules aiming to hit emission targets specified in df_targets.
    Returns the rules dict and the baseline forecast emissions DataFrame.
    """
    emission_col = f"{comp}_SASB_Metrics_Total Emissions"
    last_hist_year = base_df_wide["Year"].max()

    # --- A. Parse Targets ---
    # Use last historical emission as baseline reference for % targets
    # Could also use first year: base_df_wide.iloc[0][emission_col]
    baseline_emission_ref = base_df_wide[base_df_wide['Year'] == last_hist_year][emission_col].iloc[0]
    if pd.isna(baseline_emission_ref) or baseline_emission_ref <= 0:
        logger.error(f"Invalid baseline emission ({baseline_emission_ref}) in year {last_hist_year}. Cannot calculate targets.")
        return None, None
    logger.info(f"Using baseline emission reference from {last_hist_year}: {baseline_emission_ref:.2e}")

    target_values_abs, target_years = parse_targets(df_targets, baseline_emission_ref, logger)
    if target_values_abs is None:
        return None, None # Error parsing targets

    # Interpolate targets for all years in simulation range
    target_series = pd.Series(target_values_abs)
    target_series.index = pd.to_datetime(target_series.index.astype(str) + '-12-31') # Use datetime index
    all_years_dt = pd.to_datetime([f"{y}-12-31" for y in range(start_year, end_year + 1)])
    # Need a starting point for interpolation - use last historical emission
    last_hist_dt = pd.to_datetime(f"{last_hist_year}-12-31")
    emission_series_for_interp = pd.Series({last_hist_dt: baseline_emission_ref})
    target_series = pd.concat([emission_series_for_interp, target_series])
    # Reindex, interpolate, then get values for simulation years
    interpolated_targets = target_series.reindex(target_series.index.union(all_years_dt)).interpolate(method='time')
    targets_all_years = interpolated_targets[all_years_dt]
    targets_all_years.index = targets_all_years.index.year # Back to year index

    # --- B. Baseline Forecast ---
    # Assume predictors stay constant at last known value ("no-change" baseline)
    logger.info("Generating baseline 'no-change' forecast...")
    baseline_rules = {} # Empty rules = no change
    baseline_predictors_df = generate_scenario_predictors(
        comp, start_year, end_year, base_df_wide, baseline_rules, final_predictors, logger
    )
    baseline_emissions_df = predict_emissions_from_glm(
        glm_results, glm_scaler, final_predictors, baseline_predictors_df
    ).to_frame(name=emission_col) # Ensure it's a DataFrame with correct column name
    baseline_emissions_df['Year'] = baseline_emissions_df.index # Add Year column
    logger.info("Baseline forecast generated.")

    # --- C. Calculate Required Changes & Distribute ---
    scenario_rules = {}  # {metric: {year: percentage_change}}
    coefs = glm_results.params.drop("const", errors="ignore")

    # --- MODIFIED: Identify targetable metrics based on consts.py and coefficient sign ---
    targetable_coefs = {}
    for metric in final_predictors:
        if metric not in coefs.index:
            continue  # Skip if metric wasn't in the final model coefficients

        coef_value = coefs[metric]
        metric_name_only = extract_metric_name(metric)  # Get base name like "Nitrogen Oxides..."
        invert_flag = get_invert_flag(metric_name_only)  # Check consts.py

        # Target if:
        # 1) Reducing metric is good (invert=True) AND coefficient is positive (metric increases emissions)
        # OR
        # 2) Increasing metric is good (invert=False) AND coefficient is negative (metric decreases emissions)
        if (invert_flag and coef_value > 0) or (not invert_flag and coef_value < 0):
            targetable_coefs[metric] = coef_value
            logger.info(f"Metric '{metric}' is targetable (Invert={invert_flag}, Coef={coef_value:.3f})")
        else:
            logger.info(f"Metric '{metric}' NOT targetable (Invert={invert_flag}, Coef={coef_value:.3f})")

    # Convert dict back to Series for easier calculations
    targetable_coefs_series = pd.Series(targetable_coefs)

    # Calculate sum of ABSOLUTE values of targetable coefficients for proportion calculation
    sum_targetable_abs_coefs = abs(targetable_coefs_series).sum()
    # ------------------------------------------------------------------------------------

    if sum_targetable_abs_coefs <= 0:
        logger.warning(
            f"[{comp}] No suitable coefficients found to target for reduction based on consts.py mapping. Scenario will follow baseline.")
        return {}, baseline_emissions_df  # Return empty rules

    logger.info(f"Found {len(targetable_coefs_series)} metrics suitable for targeting emissions reduction.")

    # Get scaler scales for inverse transform
    scaler_scales = pd.Series(glm_scaler.scale_, index=glm_scaler.feature_names_in_)

    for year in range(start_year, end_year + 1):
        baseline_emission_y = baseline_emissions_df.loc[year, emission_col]
        target_emission_y = targets_all_years.loc[year]

        # If baseline is already meeting target, no change needed for this year
        if baseline_emission_y <= target_emission_y:
             # We still need rules for generate_scenario_predictors, set change to 0
             for metric in targetable_coefs_series.index:
                 if metric not in scenario_rules: scenario_rules[metric] = {}
                 scenario_rules[metric][year] = 0.0
             continue

        # Calculate required change in the linear predictor (log scale)
        # Avoid log(0) or log(<0)
        target_emission_y = max(target_emission_y, 1e-9) # Floor target
        baseline_emission_y = max(baseline_emission_y, 1e-9) # Floor baseline
        delta_linear_predictor = np.log(target_emission_y) - np.log(baseline_emission_y)

        if delta_linear_predictor >= 0: # Should be negative if reduction needed
            logger.warning(f"[{comp} - {year}] Delta linear predictor is non-negative ({delta_linear_predictor:.2f}). No reduction applied.")
            continue

        # Distribute this change across positive metrics based on coefficient size
        for metric in targetable_coefs_series.index:
            coef_m = targetable_coefs_series[metric]
            # Allocate reduction proportionally
            if sum_targetable_abs_coefs == 0:  # Avoid division by zero
                delta_X_scaled_m = 0.0
            else:
                # Proportion = |coef for this metric| / sum(|coefs for all targetable metrics|)
                proportion = abs(coef_m) / sum_targetable_abs_coefs
                delta_X_scaled_m = delta_linear_predictor * proportion  # SCALAR * SCALAR => SCALAR

            # --- Inverse Transform: Scaled change -> Unscaled change ---
            # delta_X_unscaled = delta_X_scaled * scale
            scale_m = scaler_scales.get(metric, 1.0) # Get scale for this metric
            delta_X_unscaled_m = delta_X_scaled_m * scale_m

            # --- Calculate Target Absolute Value and Percentage Change ---
            # Baseline value for the predictor in this year (from no-change forecast)
            baseline_predictor_y = baseline_predictors_df.loc[year, metric]
            if pd.isna(baseline_predictor_y) or baseline_predictor_y == 0:
                 logger.warning(f"Cannot calculate % change for '{metric}' in {year}: baseline value is {baseline_predictor_y}")
                 percentage_change_m = 0.0 # Assign no change if baseline is invalid
            else:
                 # Target absolute value = baseline + change
                 target_abs_value_m = baseline_predictor_y + delta_X_unscaled_m
                 # Ensure target doesn't go below zero if inappropriate
                 target_abs_value_m = max(target_abs_value_m, 0) # Basic floor
                 # Calculate percentage change relative to baseline predictor value
                 percentage_change_m = (target_abs_value_m - baseline_predictor_y) / baseline_predictor_y

            # Store the percentage change rule
            if metric not in scenario_rules:
                scenario_rules[metric] = {}
            scenario_rules[metric][year] = percentage_change_m

            # Optional: Store absolute target for debugging/analysis
            # if metric not in required_metric_changes_abs: required_metric_changes_abs[metric] = {}
            # required_metric_changes_abs[metric][year] = target_abs_value_m


    # Ensure rules exist for all positive metrics for all years (fill gaps with 0 change)
    all_rule_years = set(range(start_year, end_year + 1))
    for metric in targetable_coefs_series.index:
        if metric not in scenario_rules: scenario_rules[metric] = {}
        current_years = set(scenario_rules[metric].keys())
        missing_years = all_rule_years - current_years
        for year in missing_years:
            scenario_rules[metric][year] = 0.0 # Assume 0 change if no reduction was needed


    logger.info(f"[{comp}] Target-seeking rules generated for {len(scenario_rules)} metrics.")
    return scenario_rules, baseline_emissions_df


# --- Predictor Generation (Keep previous version) ---
def generate_scenario_predictors(comp, start_year, end_year, base_df, scenario_rules, final_predictors, logger=None):
    """Generates predictor values based on rules. Assumes rules provide % change factor per year."""
    # ... (Implementation from previous Option B response is suitable here) ...
    # ... (It iterates years, applies baseline * (1 + change_factor)) ...
    if logger is None: logger = logging.getLogger(__name__)
    last_hist_year = base_df["Year"].max()
    if start_year <= last_hist_year: raise ValueError(f"Start year ({start_year}) must be after last historical year ({last_hist_year})")
    last_row = base_df[base_df["Year"] == last_hist_year].iloc[0]
    scenario_data = []
    years = range(start_year, end_year + 1)
    logger.info(f"[{comp}] Generating predictor values for years {start_year}-{end_year} using target-seeking rules...")
    for y in years:
        row_dict = {}
        for col in final_predictors:
            baseline_value = last_row.get(col)
            if pd.isna(baseline_value):
                logger.warning(f"[{comp}] Baseline value for predictor '{col}' in year {last_hist_year} is missing. Using NaN for year {y}.")
                row_dict[col] = np.nan
                continue
            # Get the percentage change factor (e.g., -0.1 for 10% reduction)
            change_factor = scenario_rules.get(col, {}).get(y, 0.0)
            projected_value = baseline_value * (1.0 + change_factor)
            # Add floor just in case
            projected_value = max(projected_value, 0)
            row_dict[col] = projected_value
        row_dict["Year"] = y
        scenario_data.append(row_dict)
    scenario_df = pd.DataFrame(scenario_data).set_index("Year")
    missing_cols = [col for col in final_predictors if col not in scenario_df.columns]
    if missing_cols: logger.warning(f"[{comp}] Predictor columns missing baseline data, NaN in scenario: {missing_cols}")
    logger.info(f"[{comp}] Generated scenario predictor DataFrame shape: {scenario_df.shape}")
    return scenario_df

# --- Emission Prediction (Keep previous version) ---
def predict_emissions_from_glm(glm_results, scaler, original_predictors, df_predict):
    """Predicts emissions using GLM, handles scaling and NaNs."""
    # ... (Implementation from previous Option B response is suitable here) ...
    # ... (Handles scaling, constant, prediction, NaN checks) ...
    logger = logging.getLogger(__name__)
    logger.info(f"Predicting emissions for {len(df_predict)} future years...")
    if not all(col in df_predict.columns for col in original_predictors):
        missing = [col for col in original_predictors if col not in df_predict.columns]
        raise ValueError(f"df_predict is missing required predictor columns: {missing}")
    X_raw = df_predict[original_predictors].copy()
    if X_raw.isnull().any().any():
        logger.warning(f"NaN values found in predictors for prediction years {X_raw[X_raw.isnull().any(axis=1)].index.tolist()}. Attempting interpolation.")
        X_raw.interpolate(method='linear', limit_direction='both', inplace=True)
        if X_raw.isnull().any().any():
             logger.error("NaN values remain in predictors after interpolation. Cannot predict.")
             return pd.Series(np.nan, index=df_predict.index, name="Predicted_Emissions")
    try: X_scaled = scaler.transform(X_raw)
    except Exception as e: logger.error(f"Error applying scaler transform: {e}"); raise
    X_design = pd.DataFrame(X_scaled, columns=original_predictors, index=df_predict.index)
    X_design = sm.add_constant(X_design, has_constant='add')
    try: X_design = X_design[glm_results.params.index] # Ensure column order matches model
    except Exception as e: logger.error(f"Column mismatch: Model={glm_results.params.index}, Data={X_design.columns}"); raise
    try: linear_predictor = X_design @ glm_results.params
    except Exception as e: logger.error(f"Error calculating linear predictor: {e}"); raise
    pred_emissions = np.exp(linear_predictor)
    # Add floor for predicted emissions
    pred_emissions = np.maximum(pred_emissions, 1e-9)
    logger.info(f"Prediction complete. Range: {pred_emissions.min():.2e} to {pred_emissions.max():.2e}")
    return pd.Series(pred_emissions, index=df_predict.index, name="Predicted_Emissions")