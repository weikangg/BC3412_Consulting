import os
import json

from utils.results_saver import setup_company_logger


def load_json(filepath):
    """Helper function to load JSON data from a file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compile_company_results(company_folder):
    """
    Given a company folder (e.g., results/nextera_energy),
    load the following JSON files:
      - <company>_duration_results.json
      - <company>_historical_score_details.json
      - <company>_phased_scenario_rules.json
      - <company>_phased_scenario_score_details.json
      - <company>_scope1_model_results.json
      - <company>_scope2_model_results.json
      - <company>_scope3_model_results.json
      - <company>_total_model_results.json
    Returns a dictionary with these results.
    """
    results = {}
    company_name = os.path.basename(company_folder)
    file_names = {
        "duration_results": f"{company_name}_duration_results.json",
        "historical_scores": f"{company_name}_historical_score_details.json",
        "phased_scenario_rules": f"{company_name}_phased_scenario_rules.json",
        "phased_scenario_scores": f"{company_name}_phased_scenario_score_details.json",
        "scope1_model": f"{company_name}_scope1_model_results.json",
        "scope2_model": f"{company_name}_scope2_model_results.json",
        "scope3_model": f"{company_name}_scope3_model_results.json",
        "total_model": f"{company_name}_total_model_results.json"
    }
    for key, fname in file_names.items():
        fpath = os.path.join(company_folder, fname)
        if os.path.exists(fpath):
            try:
                results[key] = load_json(fpath)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
        else:
            print(f"File {fpath} not found for company {company_name}.")
    return results


def build_recommendation_for_company(comp_results):
    """
    Build a recommendation summary from the company results.

    The summary includes:
      - Most important metric (from total_model variable_importance_weights)
      - Recommended action for that metric (from phased_scenario_rules)
      - Expected emissions in 2045 (from phased scenario duration results)
      - Historical score trend (from historical_scores)
      - Phased scenario score trend (from phased_scenario_scores)
    """
    recommendation = {}

    # 1. Most important metric from the total model's variable_importance_weights
    total_model = comp_results.get("total_model", {})
    var_weights = total_model.get("variable_importance_weights", {})
    if var_weights:
        # Choose the metric with the highest weight.
        most_imp_metric = max(var_weights.items(), key=lambda x: x[1])[0]
        recommendation["most_important_metric"] = most_imp_metric
        recommendation["importance_weight"] = var_weights[most_imp_metric]
    else:
        recommendation["most_important_metric"] = None
        recommendation["importance_weight"] = None

    # 2. Recommended action: from phased scenario rules, get the incremental changes for the most important metric.
    rules = comp_results.get("phased_scenario_rules", {})
    if rules and recommendation["most_important_metric"] in rules:
        # The rules are stored as a dictionary of {year: "change%"}.
        recommendation["recommended_action"] = rules[recommendation["most_important_metric"]]
    else:
        recommendation["recommended_action"] = {}

    # 3. Expected emissions in 2045: from duration results (phased scenario)
    duration = comp_results.get("duration_results", {})
    phased_duration = duration.get("phased_scenario", {})
    recommendation["expected_emissions_2045"] = phased_duration.get("final_year_emission", None)
    recommendation["emission_unit"] = phased_duration.get("emission_unit", None)

    # 4. Historical score trend: from historical score details
    historical_scores = comp_results.get("historical_scores", {})
    if historical_scores:
        # Assuming the JSON structure contains a key "historical" with years as keys.
        hist = historical_scores.get("historical", {})
        hist_trend = {year: details.get("overall_score") for year, details in hist.items()}
        recommendation["historical_score_trend"] = hist_trend
    else:
        recommendation["historical_score_trend"] = {}

    # 5. Phased scenario score trend: from phased scenario score details
    phased_scores = comp_results.get("phased_scenario_scores", {})
    if phased_scores:
        ph_trend = phased_scores.get("phased_scenario", {})
        ph_trend_scores = {year: details.get("overall_score") for year, details in ph_trend.items()}
        recommendation["phased_score_trend"] = ph_trend_scores
    else:
        recommendation["phased_score_trend"] = {}
    recommendation["phased_scenario_rules"] = comp_results.get("phased_scenario_rules", {})

    return recommendation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")  # adjust as needed

def compile_recommendations(results_dir=RESULTS_DIR, output_file=None):
    """
    Walks through company subfolders in results_dir, loads the pre-generated
    '{comp}_recommendation_summary.json' for each, and aggregates them
    into a single output file.
    """
    if output_file is None:
         output_file = os.path.join(results_dir, "compiled_recommendations.json")

    print(f"\n========== COMPILING RECOMMENDATIONS from {results_dir} ==========")
    company_folders = [f for f in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, f)) and f.lower() != "industry_average"]

    if not company_folders:
        print("No company result folders found.")
        return

    compiled = {"companies": {}}
    for company_name in company_folders:
        folder_path = os.path.join(results_dir, company_name)
        recommendation_file = os.path.join(folder_path, f"{company_name}_recommendation_summary.json")

        if os.path.exists(recommendation_file):
            print(f"Loading recommendation for: {company_name}")
            recommendation_data = load_json(recommendation_file)
            if recommendation_data is not None:
                 # Store the loaded recommendation summary
                 compiled["companies"][company_name] = recommendation_data
                 # Optional: If you still want the raw results included:
                 # comp_results = compile_company_results(folder_path) # Reload raw results
                 # compiled["companies"][company_name] = {
                 #     "results": comp_results, # uncomment if needed
                 #     "recommendation": recommendation_data
                 # }
            else:
                 print(f"Failed to load recommendation summary for {company_name}")
        else:
            print(f"Recommendation summary file not found for {company_name} at {recommendation_file}")

    # Save the aggregated recommendations
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(compiled, f, indent=4)
        print(f"Compiled recommendations written to {output_file}")
    except Exception as e:
         print(f"Error writing compiled recommendations to {output_file}: {e}")

if __name__ == "__main__":
    # Pass results dir if it's different from default
    compile_recommendations()