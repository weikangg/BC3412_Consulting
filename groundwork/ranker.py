import os
import json


def load_compiled_recommendations(filepath):
    """Load the compiled recommendations JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)


def rank_companies_for_year(companies, year, trend_key):
    """
    Given all companies and a trend key ('historical_score_trend' or 'phased_score_trend'),
    return a dictionary mapping each company (that has a score for that year) to its score and rank.
    Higher overall score receives a higher rank (i.e. rank 1 is the best).
    """
    scores = []
    # Years in the JSON may be strings so convert year to string.
    year_str = str(year)
    for company, data in companies.items():
        rec = data.get("recommendation", {})
        trend = rec.get(trend_key, {})
        score = trend.get(year_str)
        if score is not None:
            scores.append((company, score))
    # Sort descending: highest score gets rank 1.
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
    for company, data in companies.items():
        rec = data.get("recommendation", {})
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


def main():
    # Set the path to your compiled recommendations JSON.
    # You can adjust this path as needed.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")  # adjust as needed
    filepath = f"{RESULTS_DIR}/compiled_recommendations.json"
    if not os.path.exists(filepath):
        print(f"Compiled recommendations file not found at: {filepath}")
        return
    compiled_data = load_compiled_recommendations(filepath)
    rankings = compute_rankings(compiled_data)
    output_file = f"{RESULTS_DIR}/rankings.json"
    with open(output_file, "w") as f:
        json.dump(rankings, f, indent=4)
    print(f"Rankings written to {output_file}")


if __name__ == "__main__":
    main()
