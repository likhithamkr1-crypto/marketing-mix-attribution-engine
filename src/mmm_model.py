import pandas as pd
import numpy as np
import statsmodels.api as sm

print("Loading data for Marketing Mix Modeling (MMM)...")
df = pd.read_csv("data/customer_journeys.csv")

# Simulate weekly media spend and revenue dataset for MMM regression analysis
np.random.seed(42)
weeks = 52
channels = ['Google_Search', 'Meta_Ads', 'TikTok', 'Email', 'Influencer']

mmm_data = []
for w in range(1, weeks + 1):
    spend_google = np.random.uniform(5000, 15000)
    spend_meta = np.random.uniform(4000, 12000)
    spend_tiktok = np.random.uniform(3000, 10000)
    spend_email = np.random.uniform(1000, 4000)
    spend_influencer = np.random.uniform(2000, 8000)
    
    # Simulated baseline revenue + diminishing returns effect from ad spend
    base_rev = 20000
    rev = (
        base_rev 
        + 3.2 * np.sqrt(spend_google) 
        + 2.8 * np.sqrt(spend_meta) 
        + 2.1 * np.sqrt(spend_tiktok) 
        + 4.5 * np.sqrt(spend_email) 
        + 2.5 * np.sqrt(spend_influencer) 
        + np.random.normal(0, 1500)
    )
    
    mmm_data.append({
        'week': w,
        'spend_google': round(spend_google, 2),
        'spend_meta': round(spend_meta, 2),
        'spend_tiktok': round(spend_tiktok, 2),
        'spend_email': round(spend_email, 2),
        'spend_influencer': round(spend_influencer, 2),
        'total_revenue': round(rev, 2)
    })

df_mmm = pd.DataFrame(mmm_data)
df_mmm.to_csv("data/mmm_weekly_spend.csv", index=False)

# Fit OLS Regression Model for Media Effectiveness
X = df_mmm[['spend_google', 'spend_meta', 'spend_tiktok', 'spend_email', 'spend_influencer']]
X = sm.add_constant(X)
y = df_mmm['total_revenue']

model = sm.OLS(y, X).fit()
print("\n--- Marketing Mix Model Regression Results ---")
print(model.summary())

# Save coefficients report
coef_df = pd.DataFrame({'Coefficient': model.params, 'P-Value': model.pvalues})
coef_df.to_csv("outputs/mmm_coefficients.csv")
print("\nMensuration coefficients successfully saved to outputs/mmm_coefficients.csv!")