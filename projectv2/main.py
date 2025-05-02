from model import Model
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('death_results.csv')
age_map = {
        '0-4 years': 2,
        '5-11 years': 8,
        '12-17 years': 14.5,
        '18-29 years': 23.5,
        '30-39 years': 34.5,
        '40-49 years': 44.5,
        '50-64 years': 57,
        '65-74 years': 69.5,
        '75 years and over': 80  # Assumed average; you can adjust as needed
    }

# Map age group
df['Age Group'] = df['Age Group'].map(age_map)

df['Region'] = df['Region'].astype(str)
# One-hot encode 'Race'
encoded = pd.get_dummies(df[['Race', 'Region']], drop_first=True)

# Drop unused columns
df = df.drop(columns=['Race', 'Region'])

# Combine with one-hot encoded race columns
std_df = pd.concat([df, encoded], axis=1)

# Separate 'Year' before scaling
year_col = std_df['Year'] - 2020
features_to_scale = std_df.drop(columns=['Year'])

# Scale only the other columns
scaler = StandardScaler()
scaled_features = pd.DataFrame(scaler.fit_transform(features_to_scale), columns=features_to_scale.columns)

# Add 'Year' back
std_df_scaled = pd.concat([year_col.reset_index(drop=True), scaled_features], axis=1)

# Save to CSV
std_df_scaled.to_csv("std_df.csv", index=False)


model = Model(True)
model.test_models()

for feature in model.poly_feature_names:
    model.graph_feature(feature)