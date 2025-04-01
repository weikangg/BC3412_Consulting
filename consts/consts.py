# ---------------------------------------------------- NOTE: ------------------------------------------------------------------#
# ------- This is hardcoded, may not be best practice, better practice may be using LLM to determine for us, ------------------#
# ------- but for best accuracy, it is hereby determined that human supervision is better and less time consuming  ------------#
# -----------------------------------------------------------------------------------------------------------------------------#

# Define an invert map for each metric.
# True means that higher raw values are worse (so they should be inverted when normalizing),
# False means that higher values are better.

def normalize_key(key: str) -> str:
    """Return the key in upper-case with surrounding whitespace removed."""
    return key.strip().upper()

def get_invert_flag(metric_name: str) -> bool:
    """
    Look up the invert flag for a given metric name in a case-insensitive way.
    Leading and trailing whitespace are ignored.
    """
    return SCORE_NORMALIZATION_INVERT_MAP.get(normalize_key(metric_name), False)

original_map = {
    "Scope 1 Emissions": True,
    "Scope 2 Emissions": True,
    "Scope 3 Emissions": True,
    "GHG Emissions": True,
    "Nitrogen Oxides, excluding nitrous oxide": True,
    "Sulfur Oxides (SOx)": True,
    "Particulate matter (PM 10)": True,
    "Lead (Pb)": True,
    "Mercury (Hg)": True,
    "Water withdrawn": True,
    "Water consumed": True,
    "Amount of Coal Combustion Residuals (CCR) generated": True,
    "Average retail electric rate for residential": True,
    "Average retail electric rate for commercial": True,
    "Average retail electric rate for industrial customers": True,
    "Typical monthly electric bill for residential customers for 500 kWh of electricity delivered per month": True,
    "Typical monthly electric bill for residential customers for 1000kWh of electricity delivered per month": True,
    "Total recordable Incident Rate": True,
    "Workforce Fatality Frequency": True,
    "Percentage of electric load served by smart grid technology": False,
    "System Average interruption duration Index": True,
    "System Average interruption frequency Index": True,
    "Customer Average Interruption Duration Index, inclusive of major event days": True,
    "Owned Net Generation Capacity (Renewable)": False,
    "Owned Net Generation Capacity (Non-Renewable)": True,
    "Owned Net Generation (Renewable)": False,
    "Owned Net Generation (Non-Renewable)": True,
    "Retail Electric Customers (Commercial)": False,
    "Retail Electric Customers (Industrial)": False,
    "Retail Electric Customers (Residential)": False,
    "CO2 Owned Generation Emissions": True,
    "CO2 Owned Generation Emissions Intensity": True,
    "Total Owned Generation CO2 Emissions": True,
    "Total Owned Generation CO2 Emissions Intensity": True,
    "NOx Emissions Intensity": True,
    "SO2 Emissions Intensity": True,
    "Hg Emissions Intensity": True,
    "Water withdrawals - consumptive rate": True,
    "Water withdrawals - non-consumptive rate": True,
    "Amount of hazardous waste manifested for disposal": True,
    "Percent of coal combustion products beneficially used": False,
    "Percentage of women in total workforce": False,
    "Percentage of minorities in total workforce": False,
    "Support of SDGs (Number of Goals)": False,
    "Total number of employees": False,
    "Percentage of women on board of directors": False,
    "Percentage of minorities on board of directors": False,
    "Environmental Fines": True,
    "Total number on board of directors": True,
    "Operating Revenue": False,
    "Operating Expenses": True,
    "Net Income": False,
    "Issuances of long-term debt, including premiums and discount": True,
    "Issuances of common stock": True,
    "Cash Flow from Operating Activities": False,
    "Capital Expenditure": True,
}

SCORE_NORMALIZATION_INVERT_MAP = {normalize_key(k): v for k, v in original_map.items()}

