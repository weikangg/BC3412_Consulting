import os
import json


def load_compiled_recommendations(filepath):
    """Load the compiled recommendations JSON file."""
    print("Loading compiled recommendations at {}".format(filepath))
    with open(filepath, "r") as f:
        return json.load(f)


def rank_companies_for_year(companies, year, trend_key):
    scores = []
    year_str = str(year)
    for company, rec in companies.items():
        # Instead of getting rec from data["recommendation"], use rec directly.
        trend = rec.get(trend_key, {})
        score = trend.get(year_str)
        if score is not None:
            scores.append((company, score))
    scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
    ranks = {}
    rank = 1
    for company, score in scores_sorted:
        ranks[company] = {"score": score, "rank": rank}
        rank += 1
    return ranks


def compute_rankings(compiled_data):
    """
    Computes, for each year in the union of years from historical and phased trends,
    the historical and phased scenario ranks for each company.
    Returns a dictionary with year keys.
    """
    companies = compiled_data.get("companies", {})
    all_years = set()
    for company, rec in companies.items():
        hist_trend = rec.get("historical_score_trend", {})
        phased_trend = rec.get("phased_score_trend", {})
        all_years.update(hist_trend.keys())
        all_years.update(phased_trend.keys())
    # Convert years to integers for sorting (if possible)
    try:
        all_years = sorted([int(y) for y in all_years])
    except Exception:
        all_years = sorted(all_years)

    ranking_by_year = {}
    for year in all_years:
        hist_ranks = rank_companies_for_year(companies, year, "historical_score_trend")
        phased_ranks = rank_companies_for_year(companies, year, "phased_score_trend")
        # Combine the rankings for each company for the year.
        year_ranks = {}
        for company in companies.keys():
            hist_data = hist_ranks.get(company, {"score": None, "rank": None})
            phased_data = phased_ranks.get(company, {"score": None, "rank": None})
            year_ranks[company] = {
                "historical_score": hist_data["score"],
                "historical_rank": hist_data["rank"],
                "phased_score": phased_data["score"],
                "phased_rank": phased_data["rank"]
            }
        ranking_by_year[str(year)] = year_ranks
    return ranking_by_year


def append_rankings(new_rankings, output_file):
    # Load existing data if it exists
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = {}
    else:
        existing_data = {}

    # new_rankings is structured as { year: { company: data } }
    for year, new_company_data in new_rankings.items():
        year_key = str(year)
        if year_key not in existing_data:
            existing_data[year_key] = {}
        # For each company in new_company_data:
        for company, company_data in new_company_data.items():
            # If company already exists, overwrite; if not, add it.
            existing_data[year_key][company] = company_data

    # Write the updated data back to the file.
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=4)


def main():
    # Set the path to your compiled recommendations JSON.
    # You can adjust this path as needed.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")  # adjust as needed
    filepath = f"{RESULTS_DIR}/compiled_recommendations.json"
    if not os.path.exists(filepath):
        print(f"Compiled recommendations file not found at: {filepath}")
        return
    output_file = f"{RESULTS_DIR}/rankings.json"
    compiled_data = load_compiled_recommendations(filepath)
    rankings = compute_rankings(compiled_data)
    print(rankings)
    append_rankings(rankings, output_file)
    print(f"Rankings written to {output_file}")

if __name__ == "__main__":
    main()
