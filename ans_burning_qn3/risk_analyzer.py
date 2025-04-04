import os
import json

import openai
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Get variable
api_key = os.getenv("OPENAI_API_KEY")
# --------------------------
# Configuration and Schema
# --------------------------

# JSON Schema for the structured risk response.
risk_schema = {
    "type": "object",
    "properties": {
        "potential_transition_risks": {"type": "string"},
        "mitigation_strategies": {"type": "string"},
        "risk_score": {"type": "number"}

    },
    "required": ["potential_transition_risks", "mitigation_strategies", "risk_score"],
    "additionalProperties": False
}

# Model configuration: adjust the model name as needed.
MODEL_NAME = "gpt-4o-2024-08-06"
TEMPERATURE = 0.7
MAX_TOKENS = 2000


# --------------------------
# Helper Functions
# --------------------------

def load_compiled_recommendations(filepath):
    """Load compiled recommendations JSON from the given filepath."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_structured_risk_assessment_response(prompt, schema, model=MODEL_NAME, temperature=TEMPERATURE,
                                            max_tokens=MAX_TOKENS):
    """
    Calls the OpenAI ChatCompletion API with a prompt that instructs the model
    to produce a JSON response that adheres to the provided JSON schema.

    The prompt should instruct the model to strictly follow the schema.
    Returns the parsed JSON response.
    """
    # Append instructions to the prompt for structured output.
    instructions = (
        "Your response MUST be a JSON that strictly follows this JSON Schema:\n"
        f"{json.dumps(schema, indent=2)}\n"
        "Do not include any extra text or explanation."
    )
    full_prompt = prompt + "\n\n" + instructions

    client = OpenAI(api_key=api_key )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are an expert in risk analysis and transition planning."},
            {"role": "user", "content": full_prompt}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "calendar_event",
                "schema": risk_schema,
                "strict": True
            }
        }
    )
    print(response)
    output = response.output_text
    if isinstance(output, str):
        try:
            structured_output = json.loads(output)
        except Exception as e:
            print("Error parsing structured response:", e)
            structured_output = {}
    elif isinstance(output, dict):
        structured_output = output
    else:
        structured_output = {}
    return structured_output

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "fig")  # adjust as needed

def plot_risk_trend(company, risk_yearly_scores, output_folder="fig"):
    """
    Plot the aggregated risk trend (average risk score) for a company over the years.
    risk_yearly_scores should be a dict mapping year (str or int) -> aggregated risk score.
    Saves the plot to a PNG file in the output_folder.
    """
    years = sorted([int(y) for y in risk_yearly_scores.keys()])
    scores = [risk_yearly_scores[str(year)] for year in years]

    plt.figure(figsize=(10, 6))
    plt.plot(years, scores, marker="o", linestyle="-")
    plt.xlabel("Year")
    plt.ylabel("Average Risk Score")
    plt.title(f"Risk Trend for {company}")
    plt.grid(True)

    output_path = os.path.join(f"{FIG_DIR}/{company}/risk_trend.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Risk trend plot saved for {company} at {output_path}")

# --------------------------
# Core Functions
# --------------------------

def build_risk_analysis_sequential(compiled_data):
    """
    For each company, process the phased recommendations year-by-year.
    For each (company, year, metric), if there is a previous risk score for that metric,
    include it in the prompt so that the model can account for a decreasing trend.

    Returns a dictionary structured as:
    {
      "Company1": {
          2024: {
              "Metric1": { "potential_transition_risks": "...", "mitigation_strategies": "...", "risk_score": 45 },
              "Metric2": { ... }
          },
          2025: { ... },
          ...
      },
      "Company2": { ... }
    }
    """
    risk_analysis = {}
    companies = compiled_data.get("companies", {})

    for company, data in companies.items():
        if company == "chubu" or company == "nextera_energy":
            continue
        risk_analysis[company] = {}
        # We assume the phased recommendations are stored under "phased_scenario_rules" in the "results" field.
        phased_scenario_rules = data.get("phased_scenario_rules", {})
        # Sort years numerically
        sorted_years = sorted(phased_scenario_rules.keys(), key=lambda y: int(y))
        # This dict will track the previous risk scores for each metric.
        previous_risk = {}

        for year in sorted_years:
            year_int = int(year)
            risk_analysis[company][year_int] = {}
            for metric, change_str in phased_scenario_rules[year].items():
                # Build prompt
                prompt = f"Company '{company}' is recommended to adjust the metric '{metric}' by {change_str} in year {year_int}."
                if metric in previous_risk:
                    prompt += f" The previous risk score for this metric was {previous_risk[metric]}. "
                prompt += (
                        "Based on this recommendation, provide a JSON response (strictly following the schema below) that includes:\n"
                        "- potential_transition_risks: a brief description of the main risks (e.g., technological, regulatory, market, or financial risks)\n"
                        "- mitigation_strategies: a brief description of strategies to mitigate these risks\n"
                        "- risk_score: a numeric risk score between 0 (no risk) and 100 (extreme risk).\n"
                        "Ensure that if effective mitigation is applied, the risk score decreases over time.\n"
                        f"For context, all the years that we will have to plan risks for this company is in this list {sorted_years}. We can start with a higher risk number and slowly decline to zero when getting the risk scores assuming mitigation strategies are followed."
                        "JSON Schema:\n" +
                        json.dumps({
                            "type": "object",
                            "properties": {
                                "potential_transition_risks": {"type": "string"},
                                "mitigation_strategies": {"type": "string"},
                                "risk_score": {"type": "number"}
                            },
                            "required": ["potential_transition_risks", "mitigation_strategies", "risk_score"],
                            "additionalProperties": False
                        }, indent=2)
                )
                # Call your API helper to get the structured response
                assessment = get_structured_risk_assessment_response(prompt, risk_schema)
                risk_analysis[company][year_int][metric] = assessment
                # If a risk score is returned, store it for future chaining.
                if "risk_score" in assessment:
                    print(f'Risk score for year {year}: {assessment["risk_score"]}')
                    previous_risk[metric] = assessment["risk_score"]
                print(f"Processed {company} - {metric} - {year_int}")
    return risk_analysis


def aggregate_risk_scores(risk_analysis):
    """
    For each company and each year, aggregate the risk scores (e.g. average across metrics).
    Returns a dictionary: { company: { year: aggregated_risk_score } }.
    """
    aggregated = {}
    for company, years_data in risk_analysis.items():
        aggregated[company] = {}
        for year, metrics_data in years_data.items():
            scores = []
            for metric, risk_info in metrics_data.items():
                score = risk_info.get("risk_score")
                if isinstance(score, (int, float)):
                    scores.append(score)
            if scores:
                aggregated_score = np.mean(scores)
            else:
                aggregated_score = None
            aggregated[company][year] = aggregated_score
    return aggregated


def save_json(data, filepath):
    """Save dictionary data as a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def append_json_data(new_data, filepath):
    # Load existing data if the file exists
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # If both new_data and existing_data are dictionaries, update existing_data.
    if isinstance(existing_data, dict) and isinstance(new_data, dict):
        # You might want to merge recursively or simply update at the top level.
        existing_data.update(new_data)
    else:
        # If they're not dictionaries, just use new_data.
        existing_data = new_data

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)


# --------------------------
# Main Execution
# --------------------------

def main():
    # Ensure your OpenAI API key is set
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Please set OPENAI_API_KEY in your environment.")
        return

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
    compiled_file = f"{RESULTS_DIR}/compiled_recommendations.json"
    if not os.path.exists(compiled_file):
        print(f"Compiled recommendations file not found at: {compiled_file}")
        return
    compiled_data = load_compiled_recommendations(compiled_file)

    print("Building risk analysis...")
    risk_analysis = build_risk_analysis_sequential(compiled_data)
    risk_analysis_file = os.path.join(RESULTS_DIR, "risk_analysis.json")
    append_json_data(risk_analysis, risk_analysis_file)
    print("Risk analysis JSON appended to", risk_analysis_file)

    aggregated_scores = aggregate_risk_scores(risk_analysis)
    agg_scores_file = os.path.join(RESULTS_DIR, "aggregated_risk_scores.json")
    append_json_data(aggregated_scores, agg_scores_file)
    print("Aggregated risk scores appended to", agg_scores_file)

    # Plot risk trend for each company
    for company, yearly_scores in aggregated_scores.items():
        filtered = {str(year): score for year, score in yearly_scores.items() if score is not None}
        if filtered:
            plot_risk_trend(company, filtered, output_folder="fig")
        else:
            print(f"No risk scores to plot for {company}.")

if __name__ == "__main__":
    main()

