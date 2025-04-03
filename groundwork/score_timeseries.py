import pandas as pd
import matplotlib.pyplot as plt
from consts.consts import get_invert_flag
from utils.utils import extract_metric_name

def zscore_to_100_scale(series, clip_range=(-3, 3), invert=False, logger=None):
    """
    Convert a Series to z-scores, then map that range linearly to [0,100].
    If std is zero, return 50 for all entries.
    """
    mean_ = series.mean()
    std_ = series.std()

    # Log some details about the input distribution.
    if logger:
        logger.debug(f"zscore_to_100_scale called with mean={mean_:.3f}, std={std_:.3f}, invert={invert}")
    else:
        print(f"[DEBUG] zscore_to_100_scale: mean={mean_:.3f}, std={std_:.3f}, invert={invert}")

    if std_ == 0:
        if logger:
            logger.debug("All values are identical; returning 50 for entire series.")
        else:
            print("[DEBUG] All values identical; returning 50.")
        return pd.Series(50, index=series.index)

    z = (series - mean_) / std_

    # Clip to avoid infinite tail
    z_clipped = z.clip(lower=clip_range[0], upper=clip_range[1])

    # Now map the clipped z-scores from [z_min, z_max] to [0, 100]
    z_min, z_max = clip_range
    if invert:
        scaled = (z_max - z_clipped) / (z_max - z_min) * 100
    else:
        scaled = (z_clipped - z_min) / (z_max - z_min) * 100

    if logger:
        logger.debug(f"zscore range pre-clip: min={z.min():.3f}, max={z.max():.3f}. "
                     f"Post-clip: min={z_clipped.min():.3f}, max={z_clipped.max():.3f}. "
                     f"Scaled to 0–100 with invert={invert}.")
    else:
        print(f"[DEBUG] zscore range pre-clip: min={z.min():.3f}, max={z.max():.3f}. "
              f"Post-clip: min={z_clipped.min():.3f}, max={z_clipped.max():.3f}. "
              f"Scaled to 0–100 with invert={invert}.")

    return scaled

def combine_scope_weights(*weight_dicts):
    """
    Combines multiple weight dictionaries. Prioritizes first dict (total),
    adds unique keys from others, then renormalizes to 100.
    """
    combined = {}
    metrics_seen = set()
    # Process dicts in order (total, s1, s2, s3 assumed)
    for w_dict in weight_dicts:
        if not w_dict: continue
        for metric, weight in w_dict.items():
            if metric not in metrics_seen:
                combined[metric] = weight
                metrics_seen.add(metric)
    total_weight = sum(combined.values())
    if total_weight == 0: return {}
    renormalized_weights = {m: (w / total_weight) * 100.0 for m, w in combined.items()}
    return renormalized_weights

def compute_score_timeseries(combined_wide_df, weight_dict, logger=None):
    """
    1. Normalize each metric across all years (within this DataFrame),
       using zscore_to_100_scale.
    2. Compute an overall weighted score from these normalized scores.

    If a logger is provided, we log relevant messages; otherwise we print.
    """
    df = combined_wide_df.copy()
    if logger:
        logger.info(f"Starting compute_score_timeseries for DataFrame of shape {df.shape}")
    else:
        print(f"[INFO] compute_score_timeseries: shape={df.shape}")

    def normalize_all_years(df_subset):
        for metric, weight in weight_dict.items():
            if metric in df_subset.columns:
                metric_name = extract_metric_name(metric)
                invert = get_invert_flag(metric_name)
                if logger:
                    logger.debug(f"Normalizing metric '{metric}' (invert={invert}) across entire DF.")
                else:
                    print(f"[DEBUG] Normalizing '{metric}' (invert={invert}).")

                df_subset[metric + "_score"] = zscore_to_100_scale(
                    df_subset[metric],
                    invert=invert,
                    logger=logger
                )
            else:
                # If a metric is missing from the DataFrame, assign 50 as neutral
                df_subset[metric + "_score"] = 50
                if logger:
                    logger.debug(f"Metric '{metric}' missing; using neutral score=50.")
                else:
                    print(f"[DEBUG] Metric '{metric}' missing; using 50.")
        return df_subset

    # We don't group by year; we apply normalization across the entire DF to
    # capture a bigger distribution than a single row.
    normalized_df = normalize_all_years(df)

    # Then compute overall score row-by-row.
    def compute_overall(row):
        total = 0
        for metric, weight in weight_dict.items():
            score = row.get(metric + "_score", 50)
            total += score * weight
        return total / 100.0

    normalized_df["overall_score"] = normalized_df.apply(
        compute_overall,
        axis=1
    ).round(0).astype(int)

    if logger:
        logger.info(f"Finished computing score_timeseries. Final shape={normalized_df.shape}")
    else:
        print(f"[INFO] Done. Output shape={normalized_df.shape}")

    return normalized_df

def plot_company_scores(scores_df, company, save_path=None, logger=None):
    """
    Plot the composite score time series for a specific company.
    Uses a date axis by converting each Year -> YYYY-12-31 to avoid
    floating-year labels like 2007.5.
    """
    comp_df = scores_df[scores_df["Company"] == company].copy()
    min_score = comp_df["overall_score"].min()
    max_score = comp_df["overall_score"].max()

    # Convert numeric year to actual datetime (YYYY-12-31).
    comp_df["plot_date"] = pd.to_datetime(comp_df["Year"].astype(str) + "-12-31")

    if logger:
        logger.info(f"Plotting {company}'s composite scores. Range: {min_score}–{max_score}")
    else:
        print(f"[INFO] {company} Score range: {min_score}–{max_score}")

    plt.figure(figsize=(10, 6))
    # Plot using the 'plot_date' column so x-axis is a proper datetime.
    plt.plot(comp_df["plot_date"], comp_df["overall_score"], marker="o", label=f"{company} Score")

    plt.title(f"Composite Score Over Time for {company}")
    plt.xlabel("Year")
    plt.ylabel("Composite Score (0–100)")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        if logger:
            logger.info(f"Saved {company} score plot to {save_path}")
        else:
            print(f"[INFO] Saved plot to {save_path}")

    plt.close()


def plot_industry_average(scores_df, save_path=None, logger=None):
    """
    Plot the industry average composite score over time using a date axis.
    """
    avg_scores = scores_df.groupby("Year")["industry_avg_overall_score"].mean().reset_index()
    # Create date column for plotting:
    avg_scores["plot_date"] = pd.to_datetime(avg_scores["Year"].astype(str) + "-12-31")

    if logger:
        logger.info(
            "Plotting industry average from a DataFrame of shape=%s. "
            "Min year=%s, Max year=%s",
            scores_df.shape,
            scores_df["Year"].min(),
            scores_df["Year"].max()
        )
    else:
        print(f"[INFO] Plotting industry average from DF shape={scores_df.shape}")

    plt.figure(figsize=(10, 6))
    plt.plot(avg_scores["plot_date"], avg_scores["industry_avg_overall_score"],
             marker="o", label="Industry Average Overall Score")
    plt.title("Industry Average Composite Score Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Composite Score (0–100)")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        if logger:
            logger.info(f"Saved industry average plot to {save_path}")
        else:
            print(f"[INFO] Saved industry average plot to {save_path}")

    plt.close()