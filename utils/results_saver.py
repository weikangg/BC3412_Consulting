import json

def save_model_results(results, selected_predictors, weight_dict, vif_df, save_path):
    """
    Save model outputs to a JSON file.

    Parameters:
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
    
    print(f"Model results saved in structured format to: {save_path}")