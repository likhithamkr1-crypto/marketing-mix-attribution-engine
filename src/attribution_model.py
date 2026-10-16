import pandas as pd
import numpy as np

print("Loading customer journey data for attribution analysis...")
df = pd.read_csv("data/customer_journeys.csv")

# Compute First-Touch, Last-Touch, and Heuristic Attributions
print("\nCalculating heuristic attribution models...")

# Last Touch Attribution
last_touch = df[df['converted'] == 1]['touchpoints'].apply(lambda x: x.split(' > ')[-1])
lt_counts = last_touch.value_counts(normalize=True) * 100

# First Touch Attribution
first_touch = df[df['converted'] == 1]['touchpoints'].apply(lambda x: x.split(' > ')[0])
ft_counts = first_touch.value_counts(normalize=True) * 100

# Combine into a professional marketing attribution summary report
attribution_summary = pd.DataFrame({
    'First_Touch_%': ft_counts,
    'Last_Touch_%': lt_counts
}).fillna(0)

print("\n--- Marketing Channel Attribution Summary (%) ---")
print(attribution_summary.round(2))

# Save output report
attribution_summary.to_csv("outputs/attribution_comparison.csv")
print("\nAttribution report successfully saved to outputs/attribution_comparison.csv!")