import json
import logging
import os
import numpy as np


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

# --- New Function to Save Scenario Rules ---
def save_scenario_rules(scenario_rules, save_path, logger: logging.Logger = None):
    """
    Saves the calculated scenario rules to a JSON file in a readable format.
    Transforms {metric: {year: pct_change}} to {year: {metric: pct_change}}.

    Parameters:
      scenario_rules: dict in the format {metric: {year: pct_change}}
      save_path     : Full path to the JSON file.
      logger        : Optional logger.
    """
    # Reformat the dictionary: Year -> Metric -> Change
    rules_by_year = {}
    if scenario_rules: # Check if not None or empty
        for metric, year_map in scenario_rules.items():
            # Extract base metric name for readability
            try:
                 # Assumes utils.utils is accessible or import it here
                 from utils.utils import extract_metric_name
                 readable_metric = extract_metric_name(metric)
            except ImportError:
                 readable_metric = metric # Fallback if import fails
            except Exception:
                 readable_metric = metric # General fallback

            for year, pct_change in year_map.items():
                year_str = str(year) # Use string for year key in JSON
                if year_str not in rules_by_year:
                    rules_by_year[year_str] = {}
                # Store as percentage for readability
                rules_by_year[year_str][readable_metric] = f"{pct_change * 100:.2f}%" # Store change as formatted percentage string

        # Sort outer dictionary by year
        rules_by_year_sorted = dict(sorted(rules_by_year.items(), key=lambda item: int(item[0])))
    else:
        rules_by_year_sorted = {} # Save empty dict if rules are None/empty


    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Convert numpy types before saving
    rules_serializable = convert_numpy_types(rules_by_year_sorted)

    try:
        with open(save_path, "w") as f:
            json.dump(rules_serializable, f, indent=4)
        if logger:
            logger.info("Formatted scenario rules saved to: %s", save_path)
    except TypeError as e:
        if logger:
             logger.error(f"JSON serialization error saving rules: {e}. Data: {rules_serializable}")
        # Optionally save raw data if serialization fails
    except Exception as e:
         if logger:
              logger.error(f"Error saving scenario rules to {save_path}: {e}", exc_info=True)


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
    detailed_scores_file = os.path.join(results_folder, f"{company_name}_score_details.json")

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
