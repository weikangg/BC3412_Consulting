import json
import logging
import os

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
        "coefficients": results.params.to_dict()
    }
    
    # Write the JSON output to file with pretty-printing (indentation)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)
    
    logger.info(f"Model results saved in structured format to: {save_path}")

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
