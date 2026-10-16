import pandas as pd
import numpy as np

np.random.seed(42)

# Generate realistic multi-touch customer journeys
print("Generating enterprise multi-touch customer journeys...")
n_users = 10000
channels = ['Google_Search', 'Meta_Ads', 'TikTok', 'Email', 'Influencer']

journey_data = []
for user_id in range(1, n_users + 1):
    path_len = np.random.choice([1, 2, 3, 4], p=[0.4, 0.3, 0.2, 0.1])
    path = np.random.choice(channels, size=path_len, replace=True).tolist()
    
    converted = 1 if np.random.rand() < 0.25 else 0
    revenue = round(np.random.normal(180, 50), 2) if converted else 0.0
    
    journey_data.append({
        'user_id': user_id,
        'touchpoints': " > ".join(path),
        'path_length': path_len,
        'converted': converted,
        'revenue': max(0.0, revenue)
    })

df = pd.DataFrame(journey_data)
df.to_csv("data/customer_journeys.csv", index=False)
print("Success! Dataset saved to data/customer_journeys.csv with", len(df), "rows.")