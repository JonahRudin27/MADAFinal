import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv('std_df.csv')

# Separate features and target
X = df.drop(columns='COVID Deaths')  # Replace 'y' with the actual name of your target column
y = df['COVID Deaths']

# Keep only numeric features for PCA
X = X.select_dtypes(include=['float64', 'int64'])

# Standardize the feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(X_scaled)

# Explained variance ratios
print("Explained variance ratios:", pca.explained_variance_ratio_)

# Get the loadings (principal component directions)
loadings = pd.DataFrame(pca.components_, columns=X.columns, index=[f'PC{i+1}' for i in range(3)])

# For each component, sort features by absolute value of loading
for pc in loadings.index:
    sorted_loadings = loadings.loc[pc].abs().sort_values(ascending=False)
    print(f"\nTop features contributing to {pc}:")
    print(sorted_loadings)