import logging
from sklearn.linear_model import LassoCV
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt

def select_features(X: pd.DataFrame, y: pd.Series, cv: int = 5, random_state: int = 0, logger:logging.Logger = None) -> list:
    """
    Use LassoCV to select features. Returns a list of predictors with non-zero coefficients.
    
    Parameters:
      X : DataFrame of predictors.
      y : Series of the response variable.
      cv: Number of folds for cross-validation.
      random_state: Seed for reproducibility.
      
    Returns:
      List of selected feature names.
    """
    lasso = LassoCV(cv=cv, random_state=random_state).fit(X, y)
    coef = lasso.coef_
    selected_features = X.columns[coef != 0].tolist()
    logger.info(f"LASSO selected features: {selected_features}")
    return selected_features


def fit_mle_model(df: pd.DataFrame, response: str, predictors: list, logger=None):
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # 1. Filter data (same as before)
    # Ensure response column exists before subsetting
    if response not in df.columns:
        logger.error(f"Response variable '{response}' not found in DataFrame columns.")
        return None, [], None
    valid_predictors = [p for p in predictors if p in df.columns]
    if not valid_predictors:
        logger.error(f"No valid predictors found in DataFrame columns for response '{response}'.")
        return None, [], None

    model_df = df[[response] + valid_predictors].copy() # Use copy to avoid SettingWithCopyWarning
    model_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop rows where response OR ANY predictor is NaN before filtering response > 0
    rows_before_na_drop = len(model_df)
    model_df.dropna(subset=[response] + valid_predictors, inplace=True)
    rows_after_na_drop = len(model_df)
    if rows_before_na_drop > rows_after_na_drop:
         logger.info(f"Dropped {rows_before_na_drop - rows_after_na_drop} rows with NaNs for response '{response}'.")

    # Ensure response is positive for Gamma GLM
    model_df = model_df[model_df[response] > 0]
    if model_df.empty:
        logger.error(f"No valid data left for '{response}' after filtering NaNs and non-positive response.")
        return None, [], None

    logger.info(f"Modeling '{response}' on {len(model_df)} rows after filtering.")
    # Refresh valid_predictors based on remaining columns after potential drops/filtering
    valid_predictors = [p for p in valid_predictors if p in model_df.columns and model_df[p].nunique() > 1]
    if not valid_predictors:
        logger.error(f"No valid predictors remaining after data filtering for '{response}'.")
        return None, [], None

    # 2. Scale predictors for LASSO/Correlation
    try:
        prelim_scaler = StandardScaler()
        X_scaled_temp_vals = prelim_scaler.fit_transform(model_df[valid_predictors])
        X_temp = pd.DataFrame(X_scaled_temp_vals, columns=valid_predictors, index=model_df.index)
        X_temp_with_const = sm.add_constant(X_temp) # Keep original X_temp for correlation
        y = model_df[response]
    except Exception as e:
        logger.error(f"Error during initial scaling for '{response}': {e}", exc_info=True)
        return None, [], None

    # 3. LASSO Feature Selection
    selected = [] # Initialize
    try:
        # Pass X_temp (no const) and y to select_features
        selected = select_features(X_temp, y, logger=logger)
    except Exception as e:
        logger.error(f"Error during LASSO selection for '{response}': {e}", exc_info=True)
        # Decide whether to proceed to backup or fail
        logger.warning("Proceeding to correlation backup despite LASSO error.")
        selected = [] # Ensure selected is empty to trigger backup


    # --- Backup Feature Selection: Correlation + VIF ---
    correlation_backup_used = False
    if not selected:
        correlation_backup_used = True
        logger.warning(f"LASSO selected 0 features for '{response}'. Attempting Correlation backup.")
        correlation_threshold = 0.15 # Example threshold (adjust as needed)
        try:
            # Calculate correlation with response 'y' using X_temp (scaled, no constant)
            correlations = X_temp.corrwith(y)
            # Select features with absolute correlation above threshold
            selected_by_corr = correlations[abs(correlations) >= correlation_threshold].index.tolist()

            if not selected_by_corr:
                 logger.error(f"Correlation backup also found 0 features for '{response}' (threshold {correlation_threshold}). Cannot build model.")
                 return None, [], None # No features found by backup either
            else:
                 logger.info(f"Correlation selected features (threshold > {correlation_threshold}): {selected_by_corr}")
                 # Use these features for the subsequent VIF check
                 selected = selected_by_corr # Overwrite empty 'selected' list
        except Exception as e:
             logger.error(f"Error during correlation backup selection for '{response}': {e}", exc_info=True)
             return None, [], None # Error during backup selection
    # --------------------------------------------------

    # 4. VIF calculation loop (Starts with features from LASSO or Correlation backup)
    final_features = selected.copy()
    vif_threshold = 10

    while True:
        if not final_features:
             logger.warning(f"No features remaining after VIF elimination for '{response}'.")
             break

        # Use X_temp_with_const here as VIF needs the constant
        # Check if features still exist in the scaled data before selecting
        current_valid_features = ['const'] + [f for f in final_features if f in X_temp_with_const.columns]
        if len(current_valid_features) <= 1: # Only 'const' left or empty
            logger.info(f"No predictors left to check VIF for {response}.")
            break

        X_current = X_temp_with_const[current_valid_features].copy()

        # Check for sufficient columns (at least const + 1 predictor)
        if X_current.shape[1] < 2:
             logger.info(f"Only {len(final_features)} feature(s) remaining. Stopping VIF checks.")
             break

        try: # Add try-except around VIF calculation
             vif_data = pd.DataFrame({
                 'Variable': X_current.columns,
                 'VIF': [ variance_inflation_factor(X_current.values, i) for i in range(X_current.shape[1]) ]
             })
             # Check for NaNs/Infs in VIF results, often indicates perfect collinearity removed by previous step
             if vif_data['VIF'].isnull().any() or np.isinf(vif_data['VIF']).any():
                  infinite_vifs = vif_data[np.isinf(vif_data['VIF'])]
                  if not infinite_vifs.empty:
                       worst_predictor = infinite_vifs.iloc[0]['Variable']
                       logger.warning(f"Removing predictor '{worst_predictor}' for {response} due to infinite VIF (perfect collinearity).")
                       if worst_predictor in final_features: final_features.remove(worst_predictor)
                       continue # Re-run VIF loop
                  else:
                       logger.warning(f"NaNs or unexpected issue in VIF calculation for {response}. Stopping VIF check.")
                       break # Stop VIF if calculation returns NaNs unexpectedly

        except Exception as e:
             logger.error(f"Error during VIF calculation for {response}: {e}. Predictors: {final_features}", exc_info=True)
             break # Stop VIF loop on error

        vif_data = vif_data[vif_data["Variable"] != "const"]
        if vif_data.empty: break # No predictors left to check

        max_vif = vif_data["VIF"].max() # Will be NaN if all are infinite/NaN - handled above

        if max_vif < vif_threshold:
            break # VIFs are acceptable
        else:
            # Find predictor with highest VIF (among finite values if infinite was already handled)
            worst_predictor = vif_data.loc[vif_data["VIF"].idxmax(), "Variable"]
            logger.info(f"Removing predictor '{worst_predictor}' for {response} due to high VIF: {max_vif:.2f}")
            if worst_predictor in final_features: final_features.remove(worst_predictor)

        if not final_features:
             logger.warning(f"Removed last predictor due to high VIF for {response}.")
             break

    logger.info(f"Final predictors after LASSO{' (Correlation Backup)' if correlation_backup_used else ''} + VIF for {response}: {final_features}")

    # 5. Check if features remain & Final Scaling
    if not final_features:
        logger.warning(f"No final features selected for '{response}'. Cannot fit GLM.")
        return None, [], None
    try:
        scaler = StandardScaler()
        # Use the original model_df but only with final_features for fitting the scaler
        X_to_scale_final = model_df[final_features]
        if X_to_scale_final.isnull().any().any(): logger.warning(f"NaNs present in final features for scaling '{response}'.")
        # Fit scaler only on non-NaN data if necessary, though dropna earlier should prevent this
        X_scaled_final = scaler.fit_transform(X_to_scale_final)
        X_final = pd.DataFrame(X_scaled_final, columns=final_features, index=X_to_scale_final.index) # Use index from scaled data
        X_final = sm.add_constant(X_final, has_constant='add') # Add constant for GLM
    except Exception as e:
        logger.error(f"Error during final scaling for {response}: {e}", exc_info=True)
        return None, [], None

    # 6. Fit final GLM
    try:
        # Align response 'y' index with 'X_final' index before fitting
        y_aligned = y.loc[X_final.index]
        if y_aligned.isnull().any(): logger.warning(f"NaNs found in aligned response 'y' for {response}")

        # Check again for NaNs/Infs right before fitting
        if X_final.isnull().values.any() or y_aligned.isnull().values.any() or \
           not np.all(np.isfinite(X_final.values)) or not np.all(np.isfinite(y_aligned.values)):
             logger.error(f"Invalid values remain in X_final or y_aligned before GLM fit for {response}. Cannot fit.")
             return None, [], None

        model = sm.GLM(y_aligned, X_final, family=sm.families.Gamma(link=sm.families.links.Log()))
        results = model.fit()
        logger.info(f"Model fitted for {response}.") # Removed summary logging for brevity
        # logger.debug(f"Model summary for {response}:\n%s", results.summary()) # Use debug if needed
    except Exception as e:
        logger.error(f"Error fitting final GLM for {response}: {e}", exc_info=True)
        return None, [], None # Return None if GLM fails

    return results, final_features, scaler

def calculate_vif(df: pd.DataFrame, predictors: list):
    """
    Calculate the Variance Inflation Factor (VIF) for each predictor.
    Returns a DataFrame with VIF values.
    """
    X = df[predictors].dropna()
    X = sm.add_constant(X)
    vif_data = pd.DataFrame({
        'Variable': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    })
    return vif_data

def extract_importance_weights(results, predictors: list, logger:logging.Logger = None ):
    """
    Convert standardized coefficients (excluding the intercept) to weights out of 100.
    
    Returns:
      weight_dict: A dictionary mapping each predictor to its weight (percentage).
    """
    # Exclude the intercept (first coefficient)
    coefs = results.params[1:]
    # Use absolute values for relative importance
    importance = np.abs(coefs)
    total = importance.sum()
    weights = (importance / total) * 100
    weight_dict = dict(zip(predictors, weights))
    
    logger.info("\nVariable Importance Weights (out of 100):")
    for var, weight in weight_dict.items():
        logger.info(f"  {var}: {weight:.2f}")
    return weight_dict

def check_residuals(results, save_path=None, logger:logging.Logger = None):
    """
    Plot residual diagnostics to check model assumptions.
    Generates:
      - Histogram of residuals.
      - QQ plot for normality of residuals.
      
    Parameters:
      logger:
      results: The fitted GLM results.
      save_path: Optional; if provided, the figure will be saved to this path.
    """
    resid = results.resid_response
    plt.figure(figsize=(12, 5))
    
    # Residual histogram
    plt.subplot(1, 2, 1)
    plt.hist(resid, bins=20, edgecolor='k')
    plt.title("Residuals Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    
    # QQ plot
    plt.subplot(1, 2, 2)
    sm.qqplot(resid, line='45', fit=True, ax=plt.gca())
    plt.title("QQ Plot of Residuals")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Figure saved to {save_path}")
