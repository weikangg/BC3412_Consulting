import json
import logging
import os
import numpy as np
import pandas as pd

from groundwork.important_metrics_analyzer import check_residuals
from utils.utils import extract_metric_name


def save_json(data, file_path):
    """Helper to save a dictionary as a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def save_model_results(results, selected_predictors, weight_dict, vif_df, save_path, logger:logging.Logger = None):
    """
    Save initial important model variables outputs to a JSON file.

    Parameters:
      logger:
      results            : The fitted model results (from statsmodels).
      selected_predictors: List of predictors used in the final model.
      weight_dict        : Dictionary mapping predictors to importance weights.
      vif_df             : DataFrame containing VIF values.
      save_path          : Full path (including filename) where the JSON file will be saved.
    """
    # Prepare the structured output
    output = {
        "model_summary": results.summary().as_text(),
        "selected_predictors": selected_predictors,
        "variable_importance_weights": weight_dict,
        "vif": vif_df.to_dict(orient="records"),
        "residuals": results.resid_response.tolist(),
        "coefficients": results.params.to_dict(),
    }

    # Write the JSON output to file with pretty-printing (indentation)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    logger.info(f"Model results saved in structured format to: {save_path}")


def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # Handle potential NaN/Inf before converting
        if np.isnan(obj):
            return None # Represent NaN as null in JSON
        elif np.isinf(obj):
            # Decide representation for Inf, e.g., a large number string or null
            return str(obj) # Or None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    # Add other types if necessary
    return obj

def save_phased_scenario_rules(phased_rules, save_path, logger: logging.Logger = None):
    """
    Saves the combined phased scenario rules to a JSON file.
    Optionally reformat to {year: {metric: pct_change%}}.
    """
    # Optional: Reformat the dictionary: Year -> Metric -> Change%
    rules_by_year = {}
    if phased_rules:
        all_years = set()
        for metric, year_map in phased_rules.items():
             all_years.update(year_map.keys())

        for year in sorted(list(all_years)):
            year_str = str(year)
            rules_by_year[year_str] = {}
            for metric, year_map in phased_rules.items():
                 if year in year_map:
                     try:
                         readable_metric = extract_metric_name(metric)
                     except ImportError: readable_metric = metric
                     except Exception: readable_metric = metric
                     pct_change = year_map[year]
                     # Save only if change is non-zero, or save all? Let's save non-zero for clarity.
                     if pct_change != 0.0:
                          rules_by_year[year_str][readable_metric] = f"{pct_change * 100:.2f}%"
    else:
        rules_by_year = {}

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    rules_serializable = convert_numpy_types(rules_by_year) # Use helper
    try:
        with open(save_path, "w") as f:
            json.dump(rules_serializable, f, indent=4)
        if logger: logger.info("Phased scenario rules saved to: %s", save_path)
    except Exception as e:
        if logger: logger.error(f"Error saving phased scenario rules to {save_path}: {e}", exc_info=True)

# --- Modified Function to Save Duration Results ---
def save_duration_results(duration_results, save_path, forecast_tag, logger: logging.Logger = None):
    """
    Save duration analysis results (including final emission) to a JSON file.
    If the file already exists, update it with a new key given by forecast_tag.

    Parameters:
      duration_results: dict with duration analysis results (e.g., net_zero_year, final_year_emission)
      save_path       : Full path to the JSON file.
      forecast_tag    : A tag (e.g., "initial" or "target_seeking_scenario") to use as the key.
      logger          : Optional logger for logging messages.
    """
    # If the file exists, load existing data; otherwise, start with an empty dict.
    if os.path.exists(save_path):
        try:
            with open(save_path, "r") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode existing JSON file at {save_path}. Starting fresh.")
            existing_data = {}
        except Exception as e:
             logger.error(f"Error loading existing duration results from {save_path}: {e}. Starting fresh.")
             existing_data = {}
    else:
        existing_data = {}

    # Update the data using the forecast tag as the key.
    # Ensure data being added is serializable
    existing_data[forecast_tag] = convert_numpy_types(duration_results)

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the updated dictionary.
    try:
        with open(save_path, "w") as f:
            json.dump(existing_data, f, indent=4)
        if logger:
            logger.info("Duration results saved/updated in: %s (Tag: %s)", save_path, forecast_tag)
    except TypeError as e:
         if logger:
              logger.error(f"JSON serialization error saving duration results: {e}. Data: {existing_data[forecast_tag]}")
    except Exception as e:
         if logger:
              logger.error(f"Error saving duration results to {save_path}: {e}", exc_info=True)

def save_company_score_details(company_name, detailed_scores, tag="historical", logger: logging.Logger = None):
    """
    Saves detailed score information (per year) to a JSON file.
    Organizes scores by tags (e.g., "historical", "target_seeking_scenario").

    Parameters:
      company_name   : Name of the company.
      detailed_scores: Dict of scores {year_str: {"metrics": ..., "overall_score": ...}}
      tag            : String identifier for this set of scores (e.g., "historical").
      logger         : Optional logger.
    """
    # Build path
    results_folder = os.path.join("results", company_name)
    os.makedirs(results_folder, exist_ok=True)
    detailed_scores_file = os.path.join(results_folder, f"{company_name}_{tag}_score_details.json")

    # Load existing data if file exists
    if os.path.exists(detailed_scores_file):
        try:
            with open(detailed_scores_file, "r") as f:
                existing_data = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode existing score details file: {detailed_scores_file}. Overwriting.")
            existing_data = {}
        except Exception as e:
             logger.error(f"Error loading existing score details from {detailed_scores_file}: {e}. Overwriting.")
             existing_data = {}
    else:
        existing_data = {}

    # Ensure scores are serializable
    scores_serializable = convert_numpy_types(detailed_scores)

    # Add/Update the scores under the specified tag
    existing_data[tag] = scores_serializable
    if logger:
        logger.info(f"Updating score details file with tag: '{tag}'")

    # Save the updated dictionary
    try:
        with open(detailed_scores_file, "w") as f:
            json.dump(existing_data, f, indent=4)
        if logger:
            logger.info(f"Detailed scores saved/updated in: {detailed_scores_file}")
    except TypeError as e:
         if logger:
              logger.error(f"JSON serialization error saving score details: {e}. Tag: {tag}")
    except Exception as e:
         if logger:
              logger.error(f"Error saving score details to {detailed_scores_file}: {e}", exc_info=True)

def save_individual_model_outputs(
    comp: str,
    model_tag: str, # e.g., "total", "scope1", "scope2", "scope3"
    results, # The statsmodels results object
    selected_predictors: list,
    weight_dict: dict,
    vif_df: pd.DataFrame,
    logger: logging.Logger
):
    """
    Saves the VIF, weights, coefficients, and residual plot for a single fitted model.

    Parameters:
      comp: Company name.
      model_tag: Identifier for the model (e.g., "scope1").
      results: Fitted statsmodels GLM results object.
      selected_predictors: List of predictors used in the final model.
      weight_dict: Dictionary of importance weights for predictors.
      vif_df: DataFrame with VIF values for selected predictors.
      logger: Logger instance.
    """
    # Check if essential inputs are valid
    if results is None or not selected_predictors or vif_df is None:
        logger.warning(f"[{comp}] Skipping saving results for model '{model_tag}': Missing model results, predictors, or VIF data.")
        return

    logger.info(f"--- Saving results for {model_tag} model ---")

    # --- Define File Paths ---
    fig_folder = os.path.join("fig", comp)
    results_folder = os.path.join("results", comp)
    os.makedirs(fig_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # Use model_tag in filenames for uniqueness
    resid_fig_path = os.path.join(fig_folder, f"{comp}_{model_tag}_residual_plot.png")
    results_file = os.path.join(results_folder, f"{comp}_{model_tag}_model_results.json")

    # --- Check Residuals and Save Plot ---
    try:
        # Ensure check_residuals can handle potential missing attributes if model failed partially
        if hasattr(results, 'resid_response'):
             check_residuals(results, save_path=resid_fig_path, logger=logger)
             logger.info(f"[{comp}] {model_tag} residual plot saved to: {resid_fig_path}")
        else:
             logger.warning(f"[{comp}] Cannot generate residual plot for {model_tag}: `resid_response` not found.")
    except Exception as e:
        logger.error(f"[{comp}] Error generating/saving residual plot for {model_tag}: {e}", exc_info=True)
    output = {
             "model_summary": results.summary().as_text() if hasattr(results, 'summary') else "Summary unavailable",
             "selected_predictors": selected_predictors if selected_predictors is not None else [],
             "variable_importance_weights": weight_dict if weight_dict is not None else {},
             "vif": vif_df.to_dict(orient="records") if vif_df is not None else [],
             "residuals": results.resid_response.tolist() if hasattr(results, 'resid_response') else [],
             "coefficients": results.params.to_dict() if hasattr(results, 'params') else {},
             "log_likelihood": results.llf if hasattr(results, 'llf') else None, # Example: Add other stats
             "aic": results.aic if hasattr(results, 'aic') else None,
             "bic": results.bic if hasattr(results, 'bic') else None,
         }
    # Use the helper to handle numpy types
    output_serializable = convert_numpy_types(output)

    # --- Save JSON Results ---
    try:
        with open(results_file, "w") as f:
            json.dump(output_serializable, f, indent=4)
        logger.info(f"[{comp}] {model_tag} model results saved to: {results_file}")
    except Exception as e:
        logger.error(f"[{comp}] Error saving model results JSON for {model_tag}: {e}", exc_info=True)

def setup_company_logger(company, results_dir="results"):
    """
    Set up a logger for a specific company.

    This logger writes log messages to both the console and a file named
    '{company}_logs.txt' in the company's results directory.

    Parameters:
      company      : The company identifier (string).
      results_dir  : The base directory where results are stored.

    Returns:
      A configured logger instance.
    """
    # Create the company results directory if it does not exist.
    company_dir = os.path.join(results_dir, company)
    os.makedirs(company_dir, exist_ok=True)
    log_file = os.path.join(company_dir, f"{company}_logs.txt")

    # Create a logger object with the company name.
    logger = logging.getLogger(company)
    logger.setLevel(logging.INFO)

    # Remove any existing handlers to avoid duplication.
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create a file handler to write to the log file.
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    # Create a console handler.
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Define a common formatter.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger.
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
