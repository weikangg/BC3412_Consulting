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

    # 1. Filter out inf, NaN, and ensure response > 0
    model_df = df[[response] + predictors].replace([np.inf, -np.inf], np.nan).dropna()
    model_df = model_df[model_df[response] > 0]
    if model_df.empty:
        raise ValueError("No valid data left after filtering out non-positive or missing values.")

    logger.info("Modeling on %d rows after filtering missing, infinite, and non-positive responses.", len(model_df))

    # 2. Temporarily scale *all* given predictors
    #    We'll call this "prelim_scaler", so we can do the LASSO step
    prelim_scaler = StandardScaler()
    X_scaled_temp = prelim_scaler.fit_transform(model_df[predictors])
    X_temp = pd.DataFrame(X_scaled_temp, columns=predictors, index=model_df.index)
    X_temp = sm.add_constant(X_temp)
    y = model_df[response]

    # 3. LASSO to select features
    selected = select_features(X_temp.drop("const", axis=1), y, logger=logger)
    # => selected is a subset of your 'predictors'

    # 4. Next, remove high-VIF features from 'selected'
    #    We'll do the same approach as you had, building X_current
    final_features = selected.copy()
    vif_threshold = 10

    while True:
        X_current = X_temp[["const"] + final_features].copy()
        vif_data = pd.DataFrame({
            'Variable': X_current.columns,
            'VIF': [
                variance_inflation_factor(X_current.values, i)
                for i in range(X_current.shape[1])
            ]
        })
        vif_data = vif_data[vif_data["Variable"] != "const"]
        max_vif = vif_data["VIF"].max()
        if max_vif < vif_threshold:
            break
        else:
            worst_predictor = vif_data.loc[vif_data["VIF"].idxmax(), "Variable"]
            logger.info("Removing predictor '%s' due to high VIF: %.2f", worst_predictor, max_vif)
            final_features.remove(worst_predictor)

    logger.info("Final predictors after LASSO + VIF: %s", final_features)

    # 5. Now that we have final_features, let's re-fit the scaler on only those columns
    scaler = StandardScaler()
    X_scaled_final = scaler.fit_transform(model_df[final_features])
    X_final = pd.DataFrame(X_scaled_final, columns=final_features, index=model_df.index)
    X_final = sm.add_constant(X_final)

    # 6. Fit the GLM on final features
    model = sm.GLM(y, X_final, family=sm.families.Gamma(link=sm.families.links.Log()))
    results = model.fit()
    logger.info("Model fitted. Summary:\n%s", results.summary())

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
