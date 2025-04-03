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
    print(model)
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

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"{company}_risk_trend.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Risk trend plot saved for {company} at {output_path}")


# --------------------------
# Core Functions
# --------------------------

def build_risk_analysis(compiled_data):
    """
    For each company, and for each year in the phased recommendations,
    use the recommended changes (phased scenario rules) to build a risk analysis.
    For each (company, year, metric) use the OpenAI API with structured output
    to get potential risks, mitigation strategies, and a risk score.

    Returns a dictionary in the following structure:
    {
      "Company1": {
          "2024": {
              "Metric1": {
                  "potential_transition_risks": "...",
                  "mitigation_strategies": "...",
                  "risk_score": 45
              },
              "Metric2": { ... }
          },
          "2025": { ... },
          ...
      },
      "Company2": { ... }
    }
    """
    risk_analysis = {}
    companies = compiled_data.get("companies", {})

    # Iterate over companies
    for company, data in companies.items():
        risk_analysis[company] = {}
        recommendation = data.get("results", {})
        # The phased recommendations are assumed to be stored under "recommended_action"
        # This structure is expected to be: { metric: { year: "change%" } }
        phased_scenario_rules = recommendation.get("phased_scenario_rules", {})
        for year, metric_actions in phased_scenario_rules.items():
            year = int(year)
            if year == 2024 or year == 2035 or year == 2045:
                # For each metric, iterate through years
                for metric, change_str in metric_actions.items():
                    # Ensure we have a sub-dictionary for this year.
                    if year not in risk_analysis[company]:
                        risk_analysis[company][year] = {}
                    # Build a prompt for the current (company, metric, year)
                    prompt = (
                        f"Company '{company}' is recommended to adjust the metric '{metric}' by {change_str} in year {year}. "
                        "Based on this recommendation, provide a JSON response (strictly following the schema) that includes:\n"
                        "- potential_transition_risks: a brief description of the main risks (e.g., technological, regulatory, market, or financial risks)\n"
                        "- mitigation_strategies: a brief description of strategies to mitigate these risks\n"
                        "- risk_score: a numeric risk score between 0 (no risk) and 100 (extreme risk), "
                        "and indicate how with effective mitigation the risk would gradually decrease over time."
                    )
                    structured_response = get_structured_risk_assessment_response(prompt, risk_schema)
                    risk_analysis[company][year][metric] = structured_response
                    print(f"Processed {company} - {metric} - {year}")
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


# --------------------------
# Main Execution
# --------------------------

def main():
    # Ensure your OpenAI API key is set
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        print("Please set OPENAI_API_KEY in your environment.")
        return

    # Load compiled recommendations (adjust the path as needed)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")  # adjust as needed
    compiled_file = f"{RESULTS_DIR}/compiled_recommendations.json"
    if not os.path.exists(compiled_file):
        print(f"Compiled recommendations file not found at: {compiled_file}")
        return
    compiled_data = load_compiled_recommendations(compiled_file)

    # Build risk analysis using structured output via ChatGPT
    print("Building risk analysis...")
    risk_analysis = build_risk_analysis(compiled_data)
    save_json(risk_analysis, f"{RESULTS_DIR}/risk_analysis.json")
    print("Risk analysis JSON saved to risk_analysis.json")

    # Aggregate risk scores for plotting (average risk score per company per year)
    aggregated_scores = aggregate_risk_scores(risk_analysis)
    save_json(aggregated_scores, f"{RESULTS_DIR}/aggregated_risk_scores.json")
    print("Aggregated risk scores saved to aggregated_risk_scores.json")

    # Plot risk trend for each company and save figures
    for company, yearly_scores in aggregated_scores.items():
        # Filter out years where score is None
        filtered = {str(year): score for year, score in yearly_scores.items() if score is not None}
        if filtered:
            plot_risk_trend(company, filtered, output_folder="fig")
        else:
            print(f"No risk scores to plot for {company}.")


if __name__ == "__main__":
    main()
