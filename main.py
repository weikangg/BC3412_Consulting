import os
import pandas as pd

from groundwork.data_import import load_all_csvs
from groundwork.data_cleaning import (
    clean_and_pivot_dataframe,
    combine_cleaned_data,
)
from groundwork.score_timeseries import plot_industry_average
from utils.company_processor import process_company
from utils.results_saver import save_json

def main():
    print("\n========== LOADING CSV FILES ==========")
    all_data_frames = load_all_csvs()

    metrics_data_frames = {k: df for k, df in all_data_frames.items() if "Targets" not in k}
    targets_data_frames = {k: df for k, df in all_data_frames.items() if "Targets" in k}

    print("\n========== LOADED DATAFRAMES ==========")
    for key in metrics_data_frames:
        print(f"✔ Metrics Loaded: {key} ({len(metrics_data_frames[key])} rows)")
    for key in targets_data_frames:
        print(f"✔ Targets Loaded: {key} ({len(targets_data_frames[key])} rows)")

    print("\n========== CLEANING & PIVOTING DATAFRAMES ==========")
    cleaned_data_long = {}
    for key, df in metrics_data_frames.items():
        print(f"➡️  Cleaning & Pivoting: {key} ...", end=" ")
        original_rows = len(df)
        df_long = clean_and_pivot_dataframe(df)
        cleaned_rows = len(df_long)
        print(f"✔ Converted: {original_rows} rows -> {cleaned_rows} rows (long format)")
        cleaned_data_long[key] = df_long

    print("\n========== COMBINING CLEANED DATA ==========")
    combined_long = combine_cleaned_data(cleaned_data_long)
    print(f"Combined long-format data has {combined_long.shape[0]} rows.")

    companies = combined_long["Company"].unique()
    print("\nFound companies:", companies)

    # Global mapping: year -> list of overall scores across companies.
    industry_yearly_scores = {}

    # --- Process each company ---
    for comp in companies:
        process_company(comp, combined_long, targets_data_frames, industry_yearly_scores)

    # --- Compute & Save Industry Average ---
    industry_avg_scores = {}
    for year, scores in industry_yearly_scores.items():
        industry_avg_scores[year] = sum(scores) / len(scores)

    industry_avg_df = pd.DataFrame({
        "Year": list(industry_avg_scores.keys()),
        "industry_avg_overall_score": list(industry_avg_scores.values())
    }).sort_values("Year")

    industry_avg_file = os.path.join("results", "industry_average", "industry_average.json")
    os.makedirs(os.path.dirname(industry_avg_file), exist_ok=True)
    save_json(industry_avg_df.to_dict(orient="records"), industry_avg_file)
    print(f"Industry average score saved to: {industry_avg_file}")

    plot_industry_average(industry_avg_df, save_path=os.path.join("fig", "industry_average.png"))
    print("\n========== DONE ==========")

if __name__ == "__main__":
    main()