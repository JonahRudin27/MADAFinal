
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

# Load and prepare data
model_df = pd.read_csv('attached_assets/combined_dataset.csv')
model_data = model_df.dropna(subset=['Population', 'Deaths'])

# 1. Correlation Analysis
plt.figure(figsize=(12, 10))
correlation_matrix = model_data.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Features")
plt.tight_layout()
plt.savefig('Results/correlation_analysis.png')
plt.close()

# 2. Death Rate Trends
plt.figure(figsize=(15, 8))
for region in range(1, 11):
    region_data = model_data[model_data['Region'] == region]
    plt.plot(region_data['Year'], region_data['%Death_Rate'], label=f'Region {region}')
plt.title('Death Rate Trends by Region (2020-2025)')
plt.xlabel('Year')
plt.ylabel('Death Rate (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('Results/death_rate_trends.png')
plt.close()

# Regression Analysis
features = ['%Rep_Leg', '%Dem_Leg', '%Mix_Leg', '%Rep_Gov', '%Dem_Gov', 
           '%Rep_State', '%Dem_State', '%Mix_State']
X = model_data[features]
y = model_data['%Death_Rate']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

reg = LinearRegression()
reg.fit(X_scaled, y)

lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)

ridge = Ridge(alpha=0.1)
ridge.fit(X_scaled, y)

coefficients = pd.DataFrame({
    'Feature': features,
    'Linear_Coef': reg.coef_,
    'Lasso_Coef': lasso.coef_,
    'Ridge_Coef': ridge.coef_
}).sort_values('Linear_Coef', ascending=False)

plt.figure(figsize=(15, 8))
melted_coef = pd.melt(coefficients, id_vars=['Feature'], var_name='Regression_Type', value_name='Coefficient')
sns.barplot(data=melted_coef, x='Coefficient', y='Feature', hue='Regression_Type')
plt.title('Political Control Features Impact on Death Rate - Multiple Regression Methods')
plt.tight_layout()
plt.savefig('Results/political_impact.png')
plt.close()

# 4. Regional Analysis
regional_summary = model_data.groupby('Region').agg({
    'Deaths': 'sum',
    'Population': 'mean',
    '%Death_Rate': 'mean',
    '%Rep_State': 'mean',
    '%Dem_State': 'mean'
}).round(2)

# Plot regional summary
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

regional_summary['%Death_Rate'].plot(kind='bar', ax=ax1)
ax1.set_title('Average Death Rate by Region')
ax1.set_xlabel('Region')
ax1.set_ylabel('Death Rate (%)')

regional_summary[['%Rep_State', '%Dem_State']].plot(kind='bar', ax=ax2)
ax2.set_title('Average Political Control by Region')
ax2.set_xlabel('Region')
ax2.set_ylabel('Percentage')
ax2.legend(['Republican', 'Democrat'])

plt.tight_layout()
plt.savefig('Results/regional_analysis.png')
plt.close()

# Clustering Analysis
kmeans = KMeans(n_clusters=3, random_state=42)
model_data['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(12, 8))
for cluster in range(3):
    cluster_data = model_data[model_data['Cluster'] == cluster]
    plt.scatter(cluster_data['%Rep_State'], cluster_data['%Death_Rate'], 
                label=f'Cluster {cluster}')
plt.xlabel('Republican State Control (%)')
plt.ylabel('Death Rate (%)')
plt.title('Clusters of Regions by Political Control and Death Rate')
plt.legend()
plt.tight_layout()
plt.savefig('Results/clustering_analysis.png')
plt.close()

with open('Results/analysis_summary.txt', 'w') as f:
    f.write("Data Analysis Summary\n")
    f.write("===================\n\n")
    f.write("1. Regression Score: {:.3f}\n".format(reg.score(X_scaled, y)))
    f.write("\n2. Feature Coefficients:\n")
    f.write(coefficients.to_string())
    f.write("\n\n3. Regional Summary:\n")
    f.write(regional_summary.to_string())

#PCA Analysis
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                     c=model_data['%Death_Rate'], cmap='viridis')
plt.colorbar(scatter, label='Death Rate (%)')
first_component = features[np.abs(pca.components_[0]).argmax()]
second_component = features[np.abs(pca.components_[1]).argmax()]

plt.xlabel(f'{first_component} ({pca.explained_variance_ratio_[0]:.2%} variance explained)')
plt.ylabel(f'{second_component} ({pca.explained_variance_ratio_[1]:.2%} variance explained)')
plt.title('PCA of Political Control Features Colored by Death Rate')
plt.tight_layout()
plt.savefig('Results/pca_analysis.png')
plt.close()

components_df = pd.DataFrame(
    pca.components_,
    columns=features,
    index=[first_component, second_component]
)
with open('Results/analysis_summary.txt', 'a') as f:
    f.write("\n\n4. PCA Components:\n")
    f.write(components_df.to_string())

print("Analysis completed. Results saved in the Results folder.")
