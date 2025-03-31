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
    """
    Fit a GLM model using Maximum Likelihood Estimation.

    Parameters:
      df         : Cleaned DataFrame containing both the response and predictor variables.
      response   : The dependent variable (e.g., 'Emissions'). Must be strictly positive.
      predictors : List of predictor column names.
      logger     : Optional logger object for logging messages.

    Returns:
      results         : The fitted GLM model results.
      final_features  : The final list of predictor names used.
      scaler          : The StandardScaler fitted to the predictors.
    """

    # If no logger is provided, use a module-level logger.
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    # Replace infinite values with NaN and drop rows with missing data.
    model_df = df[[response] + predictors].replace([np.inf, -np.inf], np.nan).dropna()

    # Ensure the response is strictly positive.
    model_df = model_df[model_df[response] > 0]
    if model_df.empty:
        raise ValueError("No valid data left for modeling after filtering out non-positive or missing values.")

    logger.info("Modeling on %d rows after filtering missing, infinite, and non-positive responses.", len(model_df))

    # Standardize predictors.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(model_df[predictors])
    # Create a DataFrame with standardized predictors.
    X = pd.DataFrame(X_scaled, columns=predictors, index=model_df.index)
    # Add constant for GLM.
    X = sm.add_constant(X)
    y = model_df[response]

    # Optionally perform automated feature selection with LASSO.
    selected = select_features(X.drop("const", axis=1), y, logger=logger)

    # Now, remove predictors with high VIF.
    # Start with the predictors selected by LASSO.
    final_features = selected.copy()
    vif_threshold = 10
    while True:
        # Construct design matrix with constant for the current features.
        X_current = sm.add_constant(X[final_features])
        # Calculate VIF for each predictor (excluding the constant).
        vif_data = pd.DataFrame({
            'Variable': X_current.columns,
            'VIF': [variance_inflation_factor(X_current.values, i) for i in range(X_current.shape[1])]
        })
        vif_data = vif_data[vif_data["Variable"] != "const"]
        max_vif = vif_data["VIF"].max()
        if max_vif < vif_threshold:
            break
        else:
            # Identify the predictor with the highest VIF.
            worst_predictor = vif_data.loc[vif_data["VIF"].idxmax(), "Variable"]
            logger.info("Removing predictor '%s' due to high VIF: %.2f", worst_predictor, max_vif)
            final_features.remove(worst_predictor)

    logger.info("Final predictors after VIF filtering: %s", final_features)

    # Construct final design matrix.
    X_selected = sm.add_constant(X[final_features])
    # Fit a GLM with Gamma family and Log link.
    model = sm.GLM(y, X_selected, family=sm.families.Gamma(link=sm.families.links.Log()))
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
