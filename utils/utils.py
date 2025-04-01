def extract_metric_name(full_metric_key):
    """
    Given a full metric key in the format:
      {comp_name}_{metric_type}_{metric_name}
    where metric_type is one of: SASB_Metrics, Environment, Social, Governance, or Financial,
    return the metric_name portion.
    """
    parts = full_metric_key.split('_')
    metric_types = {"SASB", "Environment", "Social", "Governance", "Financial"}
    for i, part in enumerate(parts):
        if part in metric_types:
            # If using SASB_Metrics, skip the next token as well.
            if part == "SASB" and i+1 < len(parts) and parts[i+1] == "Metrics":
                return " ".join(parts[i+2:])
            return " ".join(parts[i+1:])
    return full_metric_key

def extract_metric_unit(full_metric_key, comp_data, logger=None):
    """
    Given a full metric key in the format:
      {comp_name}_{metric_type}_{metric_name}
    where metric_type is one of: SASB_Metrics, Environment, Social, Governance, or Financial,
    return the metric_name portion.
    """
    metric_name = extract_metric_name(full_metric_key)
    mask = comp_data["Metric"] == metric_name
    unit = comp_data.loc[mask, "Units"].iloc[0] if mask.any() else "error"
    logger.info("Using unit: %s", unit)
    return unit