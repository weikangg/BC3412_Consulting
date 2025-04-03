import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


# --- Helper Function to parse targets ---
def parse_targets(df_targets, base_emission, logger):
    """Parses the targets dataframe to get absolute emission targets per year."""
    target_values = {}
    target_years = sorted([int(col.strip()) for col in df_targets.columns if col.strip().isdigit()])
    if not target_years:
        logger.error("No numeric years found in target columns.")
        return None, None

    try:
        # Assuming target metric is in the first row, first column
        target_metric_name = df_targets.columns[0]
        target_row = df_targets[df_targets[target_metric_name].str.contains("Emission Target", case=False, na=False)]
        if target_row.empty:
             logger.error("Could not find row containing 'Emission Target' in target metric column.")
             # Fallback: try using the first row
             target_row = df_targets.iloc[[0]]
             logger.warning("Using first row of targets file as fallback.")

        target_row = target_row.iloc[0] # Get the first matching row as a Series

        for year in target_years:
            year_str = str(year)
            if year_str not in target_row.index:
                logger.warning(f"Target year {year_str} not found in target row columns.")
                continue
            # Assume value is percentage reduction (e.g., 0.7 means 70% reduction)
            reduction_fraction = pd.to_numeric(target_row[year_str], errors='coerce')
            if pd.isna(reduction_fraction):
                 logger.warning(f"Could not parse target value for year {year_str}. Skipping.")
                 continue

            # Calculate absolute target: Baseline * (1 - Reduction Fraction)
            target_val = base_emission * (1.0 - reduction_fraction)
            target_values[year] = target_val
            logger.info(f"Target parsed for {year}: {reduction_fraction*100:.1f}% reduction => {target_val:.2e}")

    except Exception as e:
        logger.error(f"Error parsing targets data: {e}")
        return None, None

    if not target_values:
        logger.error("Failed to parse any valid target values.")
        return None, None

    return target_values, target_years

def ensemble_forecast_emissions(historical_df, forecast_years, logger=None, threshold_ratio=0.01):
    if logger is None:
        logger = logging.getLogger(__name__)
    # Prophet forecast
    prophet_model = Prophet(yearly_seasonality=False)
    prophet_model.fit(historical_df)
    future_dates = pd.DataFrame({'ds': pd.to_datetime([f"{year}-12-31" for year in forecast_years])})
    prophet_forecast = prophet_model.predict(future_dates)
    prophet_pred = prophet_forecast['yhat'].values
    logger.info("Prophet forecast completed.")
    # Exponential Smoothing Forecast
    es_model = ExponentialSmoothing(historical_df['y'], trend='add', seasonal=None,
                                    initialization_method='estimated').fit()
    es_forecast = es_model.forecast(len(forecast_years))
    es_pred = es_forecast.values
    logger.info("Exponential Smoothing forecast completed.")
    # ARIMA Forecast
    try:
        arima_model = ARIMA(historical_df['y'], order=(1, 1, 1)).fit()
        arima_forecast = arima_model.forecast(len(forecast_years))
        arima_pred = arima_forecast.values
        logger.info("ARIMA forecast completed.")
    except Exception as e:
        logger.error("ARIMA model failed: %s. Using Exponential Smoothing forecast as fallback.", e)
        arima_pred = es_pred
    # Ensemble average
    ensemble_pred = (prophet_pred + es_pred + arima_pred) / 3.0
    ensemble_df = future_dates.copy()
    ensemble_df['ensemble'] = ensemble_pred
    # Determine net-zero year based on baseline (first historical value)
    baseline = historical_df.iloc[0]['y']
    threshold = threshold_ratio * baseline
    net_zero_year = None
    for idx, row in ensemble_df.iterrows():
        if row['ensemble'] <= threshold:
            net_zero_year = row['ds'].year
            break
    if net_zero_year is None:
        logger.info("Net-zero threshold not reached within the forecast horizon.")
    else:
        logger.info("Net-zero is forecasted to be reached in year: %d", net_zero_year)
    return ensemble_df, net_zero_year


def analyze_duration(company, df_wide, targets_df, unit="tons", forecast_tag="initial", logger=None,
                     threshold_ratio=0,baseline_forecast_col=None,actual_last_hist_year=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    # Use the total emissions column
    emission_col = f"{company}_SASB_Metrics_Total Emissions"
    if emission_col not in df_wide.columns:
        logger.error("Emission column '%s' not found in df_wide for company %s.", emission_col, company)
        return None, None

    # --- Determine the TRUE historical boundary year ---
    if actual_last_hist_year is None:
        # Initial call: Determine from df_wide (which should only contain historical data)
        logger.debug("Initial duration call: Determining historical boundary from input df_wide.")
        # Ensure 'Year' column exists
        if 'Year' not in df_wide.columns:
            logger.error("Initial call failed: 'Year' column missing from df_wide.")
            return None, None
        # Find max year among rows with non-NA emission values
        temp_hist_df = df_wide[['Year', emission_col]].dropna()
        if temp_hist_df.empty:
            logger.error("Initial call failed: No non-NA historical emission data found.")
            return None, None
        current_max_hist_year = int(temp_hist_df['Year'].max())
        is_scenario_run_check = False
        logger.info(f"Initial call: Max historical year set to {current_max_hist_year}.")
    else:
        # Scenario check call: Use the explicitly passed year
        logger.debug(f"Scenario duration check: Using provided actual_last_hist_year: {actual_last_hist_year}")
        current_max_hist_year = int(actual_last_hist_year)  # Ensure int
        is_scenario_run_check = True

    # --- Extract ONLY the true historical data for Prophet ---
    # Filter the input df_wide up to the determined historical boundary
    historical_data_for_prophet = df_wide[df_wide['Year'] <= current_max_hist_year][
        ['Year', emission_col]].dropna().copy()

    if historical_data_for_prophet.empty:
        logger.error(f"Failed to extract any historical data up to year {current_max_hist_year} from df_wide.")
        return None, None

    historical_data_for_prophet['Year'] = historical_data_for_prophet['Year'].astype(int)
    historical_data_for_prophet = historical_data_for_prophet.sort_values('Year')
    logger.info("Extracted %d TRUE historical records (up to year %d) for base analysis.",
                len(historical_data_for_prophet), current_max_hist_year)

    # Prepare prophet_df using ONLY the true historical data
    historical_data_for_prophet['ds'] = pd.to_datetime(historical_data_for_prophet['Year'].astype(str) + '-12-31')
    historical_data_for_prophet = historical_data_for_prophet.rename(columns={emission_col: 'y'})
    prophet_df = historical_data_for_prophet[['ds', 'y']]  # This now correctly contains ONLY historical points

    # --- Proceed with logic based on whether it's the scenario check run ---
    ensemble_df = None  # Initialize
    net_zero_year = None  # Initialize
    final_year_emission = None # Initialize new return value

    if is_scenario_run_check:
        # If it's the scenario check run, analyze the future part of df_wide
        logger.info("Analyzing duration based on provided scenario data in df_wide (post-history).")
        # Extract the future forecast part from the input df_wide
        forecast_df = df_wide[df_wide['Year'] > current_max_hist_year][['Year', emission_col]].dropna().copy()

        if forecast_df.empty:
            logger.warning(f"No future data (Year > {current_max_hist_year}) found in df_wide for scenario check.")
            # ensemble_df remains None, net_zero_year remains None
        else:
            forecast_df['ds'] = pd.to_datetime(forecast_df['Year'].astype(str) + '-12-31')
            forecast_df = forecast_df.rename(columns={emission_col: 'yhat'})  # Use 'yhat'

            # Calculate net-zero based on these provided 'yhat' values
            if not prophet_df.empty:  # Need historical baseline
                baseline = prophet_df.iloc[0]['y']
                threshold = threshold_ratio * baseline
                for idx, row in forecast_df.iterrows():
                    if 'yhat' in row and pd.notna(row['yhat']) and row['yhat'] <= threshold:
                        net_zero_year = int(row['ds'].year)  # Ensure int
                        break
                logger.info(
                    f"Net-zero threshold ({threshold:.2e}) comparison done on scenario data. Result: {net_zero_year}")
            else:
                logger.warning(
                    "Cannot determine net-zero threshold for scenario: historical data (prophet_df) is empty.")

            # Prepare ensemble_df for consistent return format
            final_year_emission = forecast_df.iloc[-1]['yhat'] # Get last value from yhat column
            logger.info(f"Scenario final year ({forecast_df.iloc[-1]['ds'].year}) emission: {final_year_emission:.2e}")
            ensemble_df = forecast_df[['ds', 'yhat']].rename(columns={'yhat': 'ensemble'})
            if 'ensemble' not in ensemble_df.columns: ensemble_df['ensemble'] = np.nan  # Handle potential failure

    else:
        # If it's the initial run, perform ensemble forecast using the (correctly isolated) historical prophet_df
        logger.info("Performing ensemble forecast on historical data (up to %d).", current_max_hist_year)
        # ... (logic for determining forecast_years using current_max_hist_year - remains the same as your last corrected version) ...
        target_years = []
        # ... (rest of forecast_years calculation) ...
        if targets_df is not None:
            if not targets_df.empty:
                target_years = sorted([int(col.strip()) for col in targets_df.columns if col.strip().isdigit()])
                if not target_years: logger.warning("targets_df provided but no numeric year columns found.")
            else:
                logger.warning("targets_df is provided but empty.")

        forecast_years = []
        start_forecast_year = current_max_hist_year + 1
        default_forecast_horizon = 15

        if not target_years:
            forecast_years = list(range(start_forecast_year, start_forecast_year + default_forecast_horizon))
            logger.warning(
                f"No valid target years. Forecasting default {default_forecast_horizon}-year range: {forecast_years}")
        else:
            last_target_year = target_years[-1]
            if start_forecast_year > last_target_year:
                logger.warning(
                    f"History ({current_max_hist_year}) >= last target year ({last_target_year}). Forecasting default {default_forecast_horizon} years.")
                forecast_years = list(range(start_forecast_year, start_forecast_year + default_forecast_horizon))
            else:
                forecast_years = list(range(start_forecast_year, last_target_year + 1))
                logger.info(f"Forecasting from {start_forecast_year} to last target year {last_target_year}.")

        if not forecast_years:
            logger.error("Forecast years list is unexpectedly empty! Cannot run ensemble forecast.")
            # ... (plotting code for failure as before) ...
            return None, None  # Indicate failure

        logger.info(f"Final forecast years for ensemble: {forecast_years}")
        # Ensure prophet_df is not empty before forecasting
        if prophet_df.empty:
            logger.error("Cannot perform ensemble forecast: Historical data (prophet_df) is empty.")
            # ... (plotting code for failure) ...
            return None, None
        # Call ensemble forecast
        ensemble_df, net_zero_year = ensemble_forecast_emissions(prophet_df, forecast_years, logger=logger,
                                                                 threshold_ratio=threshold_ratio)
        # Get emission from the last row/year of the ensemble forecast
        if ensemble_df is not None and not ensemble_df.empty:
            final_year_emission = ensemble_df.iloc[-1]['ensemble']  # Get last value
            logger.info(
                f"Initial forecast final year ({ensemble_df.iloc[-1]['ds'].year}) emission: {final_year_emission:.2e}")
        else:
            logger.warning("Initial forecast did not produce results. Final emission is None.")


    # --- Plotting --- (remains the same, uses the correctly prepared prophet_df and potentially None ensemble_df)
    # ... (plotting code as in previous version - check for None ensemble_df) ...
    # (Ensure baseline calculation for plotting uses prophet_df.iloc[0]['y'])
    plt.figure(figsize=(12, 8))
    plt.plot(prophet_df['ds'], prophet_df['y'], 'bo-', label='Historical Emissions')  # Always plot true history

    # Plot Forecast (Scenario or Ensemble) if available
    if ensemble_df is not None and not ensemble_df.empty and 'ensemble' in ensemble_df.columns:
        plt.plot(ensemble_df['ds'], ensemble_df['ensemble'], 'r*--', label=f'{forecast_tag} Forecast')

        # Plot Baseline Forecast Comparison (using df_wide passed in, which is combined_df on scenario runs)
        if baseline_forecast_col and baseline_forecast_col in df_wide.columns:
            ensemble_years = ensemble_df['ds'].dt.year
            baseline_plot_df = df_wide[df_wide['Year'].isin(ensemble_years)][
                ['Year', baseline_forecast_col]].dropna().copy()
            if not baseline_plot_df.empty:
                baseline_plot_df['ds'] = pd.to_datetime(baseline_plot_df['Year'].astype(str) + '-12-31')
                plt.plot(baseline_plot_df['ds'], baseline_plot_df[baseline_forecast_col], 'k.:',
                         label='Baseline (No Change) Forecast')

        # Plot Net-Zero Threshold
        if not prophet_df.empty:
            baseline_for_threshold = prophet_df.iloc[0]['y']  # Use first historical point
            plt.axhline(y=baseline_for_threshold * threshold_ratio, color='gray', linestyle='--',
                        label=f'Net Zero Threshold ({threshold_ratio * 100:.0f}%)')
    else:
        logger.warning("Plotting forecast skipped: ensemble_df is None, empty, or missing 'ensemble' column.")

    # Plot Emission Targets
    if targets_df is not None and not prophet_df.empty:
        try:
            baseline_for_targets = prophet_df.iloc[0]['y']  # Use first historical point
            target_values_abs, target_years_parsed = parse_targets(targets_df, baseline_for_targets, logger)
            if target_values_abs:
                target_dates = pd.to_datetime([f"{year}-12-31" for year in target_values_abs.keys()])
                plt.plot(target_dates, list(target_values_abs.values()), 'g^-', label='Emissions Targets')
        except Exception as e:
            logger.warning(f"Could not plot targets: {e}")

    # Finalize and save plot
    plt.title(f"{company} - Emissions Forecast ({forecast_tag.replace('_', ' ').title()})")
    plt.xlabel("Year")
    plt.ylabel(f"Emissions ({unit})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    fig_folder = os.path.join("fig", company)
    os.makedirs(fig_folder, exist_ok=True)
    fig_path = os.path.join(fig_folder, f"{company}_ensemble_forecast_{forecast_tag}.png")
    plt.savefig(fig_path);
    plt.close()
    logger.info("Ensemble forecast figure saved to: %s", fig_path)

    return ensemble_df, net_zero_year,final_year_emission