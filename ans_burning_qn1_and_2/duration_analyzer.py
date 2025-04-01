import os
import pandas as pd
import matplotlib.pyplot as plt
import logging
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA


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
                     threshold_ratio=0):
    if logger is None:
        logger = logging.getLogger(__name__)
    # Use the total emissions column
    emission_col = f"{company}_SASB_Metrics_Total Emissions"
    if emission_col not in df_wide.columns:
        logger.error("Emission column '%s' not found in df_wide for company %s.", emission_col, company)
        return None, None
    historical_df = df_wide[['Year', emission_col]].dropna().copy()
    historical_df['Year'] = historical_df['Year'].astype(int)
    historical_df = historical_df.sort_values('Year')
    logger.info("Extracted %d historical records for company %s.", len(historical_df), company)
    historical_df['ds'] = pd.to_datetime(historical_df['Year'].astype(str) + '-12-31')
    historical_df = historical_df.rename(columns={emission_col: 'y'})
    prophet_df = historical_df[['ds', 'y']]

    # Get forecast years from targets_df (columns that are numeric)
    target_years = sorted([int(col.strip()) for col in targets_df.columns if col.strip().isdigit()])
    # Suppose row 0 has the fraction for each year
    target_row = targets_df.iloc[0]
    # If 0.2 means a 20% cut from baseline => baseline*(1 - 0.2).
    # If 0.2 means 20% of baseline => baseline*0.2.
    # Adjust logic as needed:
    baseline = prophet_df.iloc[0]['y']
    target_values = []
    for y in target_years:
        fraction = target_row[str(y)]  # e.g. 0.2 means 20% cut
        # interpret fraction as "we want to reduce by fraction"
        # => actual target = baseline * (1 - fraction)
        target_val = baseline * (1 - fraction)
        target_values.append(target_val)
    logger.info("Forecast years (from targets): %s", target_years)

    ensemble_df, net_zero_year = ensemble_forecast_emissions(prophet_df, target_years, logger=logger,
                                                             threshold_ratio=threshold_ratio)

    if not net_zero_year:
        msg = f'{company} is unable to hit net zero targets by target year {target_years[-1]}.'
        logger.info(msg)
    else:
        msg = f'{company} will hit net zero targets by  year {net_zero_year}'
        logger.info(msg)

    plt.figure(figsize=(12, 6))
    plt.plot(prophet_df['ds'], prophet_df['y'], 'bo-', label='Historical Emissions')
    plt.plot(ensemble_df['ds'], ensemble_df['ensemble'], 'ro--', label='Ensemble Forecast')

    # Plot net-zero threshold (baseline * threshold_ratio)
    baseline = prophet_df.iloc[0]['y']
    plt.axhline(y=baseline * threshold_ratio, color='gray', linestyle='--', label='Net Zero Threshold')

    # Plot target line
    target_dates = pd.to_datetime([f"{year}-12-31" for year in target_years])
    plt.plot(target_dates, target_values, 'g^-', label='Emissions Targets')

    plt.title(f"{company} - Ensemble Forecast of Total Emissions")
    plt.xlabel("Year")
    plt.ylabel(f"Emissions ({unit})")
    plt.legend()
    plt.grid(True)

    # Save figure using forecast_tag in the file name.
    fig_folder = os.path.join("fig", company)
    os.makedirs(fig_folder, exist_ok=True)
    fig_path = os.path.join(fig_folder, f"{company}_ensemble_forecast_{forecast_tag}.png")
    plt.savefig(fig_path)
    plt.close()
    logger.info("Ensemble forecast figure saved to: %s", fig_path)

    return ensemble_df, net_zero_year