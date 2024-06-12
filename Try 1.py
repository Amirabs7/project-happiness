import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import scipy.stats as stats
import plotly.express as px

# Load the data
data = pd.read_csv("merged_df_happy.csv")

# Data Exploration
print("First few rows of the data")
print(data.head())

print("Shape of the data")
print(data.shape)

print("Column names")
print(data.columns.tolist())

print("Data types")
print(data.dtypes)

# Missing values
missing_percentage = (data.isnull().sum() / len(data)) * 100
print("Percentage of missing data in each variable:")
print(missing_percentage)

# Dropping specific columns
variables_to_drop = [
    'Standard error of ladder score', 'upperwhisker', 'lowerwhisker', 'Ladder score in Dystopia',
    'Explained by: Log GDP per capita', 'Explained by: Social support', 'Explained by: Healthy life expectancy',
    'Explained by: Freedom to make life choices', 'Explained by: Generosity', 'Explained by: Perceptions of corruption',
    'Dystopia + residual'
]
data.drop(columns=variables_to_drop, inplace=True)

print("Remaining columns after dropping specified variables:")
print(data.columns.tolist())

# Imputing missing values
imputer = SimpleImputer(strategy='mean')
columns_with_missing = ['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                        'Freedom to make life choices', 'Generosity', 'Perceptions of corruption',
                        'Positive affect', 'Negative affect']
data[columns_with_missing] = imputer.fit_transform(data[columns_with_missing])

# Data Visualization
life_ladder = data['Life Ladder']
plt.figure(figsize=(8, 6))
plt.hist(life_ladder, bins=20, color='skyblue', edgecolor='black')
plt.title('Distribution of Life Ladder')
plt.xlabel('Life Ladder')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

stats.probplot(life_ladder, dist="norm", plot=plt)
plt.title('Q-Q Plot for Life Ladder')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.show()

statistic, p_value = stats.shapiro(life_ladder)
print("\nShapiro-Wilk Test Statistic for Life Ladder:", statistic)
print("P-value for Life Ladder:", p_value)

df_grouped = data.groupby(['Country name', 'Regional indicator']).agg({
    'Life Ladder': 'mean',
    'Healthy life expectancy at birth': 'mean'
}).reset_index()
fig = px.sunburst(df_grouped, path=['Regional indicator', 'Country name'], values='Life Ladder',
                  color='Healthy life expectancy at birth', color_continuous_scale='RdBu',
                  color_continuous_midpoint=np.average(df_grouped['Healthy life expectancy at birth'],
                                                       weights=df_grouped['Life Ladder']))
fig.update_layout(margin=dict(t=10, l=10, r=10, b=10))
fig.show()

df_top5 = data.groupby('year').apply(lambda x: x.nlargest(5, 'Life Ladder')).reset_index(drop=True)
fig_top5 = px.line(df_top5, x='year', y='Life Ladder', color='Country name', line_group='Country name',
                   hover_name='Country name', labels={'Life Ladder': 'Life Ladder Score', 'year': 'Year'},
                   title='Top 5 Happiest Countries (2005-2021)')
fig_top5.show()

df_bottom5 = data.groupby('year').apply(lambda x: x.nsmallest(5, 'Life Ladder')).reset_index(drop=True)
fig_bottom5 = px.line(df_bottom5, x='year', y='Life Ladder', color='Country name', line_group='Country name',
                      hover_name='Country name', labels={'Life Ladder': 'Life Ladder Score', 'year': 'Year'},
                      title='Top 5 Unhappiest Countries (2005-2021)')
fig_bottom5.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=data, x='Log GDP per capita', y='Life Ladder')
plt.title('Scatter Plot of Life Ladder vs Log GDP per capita')
plt.xlabel('Log GDP per capita')
plt.ylabel('Life Ladder')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Regional indicator', y='Life Ladder')
plt.title('Box Plot of Life Ladder by Regional Indicator')
plt.xlabel('Regional Indicator')
plt.ylabel('Life Ladder')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 8))
correlation_matrix = data[['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                           'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Linear Regression Model
label_encoder = LabelEncoder()
data['Regional indicator'] = label_encoder.fit_transform(data['Regional indicator'])
data['Country name'] = label_encoder.fit_transform(data['Country name'])

X = data[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
          'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
y = data['Life Ladder']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_reg_model = LinearRegression()
linear_reg_model.fit(X_train, y_train)
y_train_pred_lr = linear_reg_model.predict(X_train)
y_test_pred_lr = linear_reg_model.predict(X_test)

print("Linear Regression Model Metrics:")
print("Training R²:", r2_score(y_train, y_train_pred_lr))
print("Training MAE:", mean_absolute_error(y_train, y_train_pred_lr))
print("Training MSE:", mean_squared_error(y_train, y_train_pred_lr))
print("Training RMSE:", np.sqrt(mean_squared_error(y_train, y_train_pred_lr)))
print("Testing R²:", r2_score(y_test, y_test_pred_lr))
print("Testing MAE:", mean_absolute_error(y_test, y_test_pred_lr))
print("Testing MSE:", mean_squared_error(y_test, y_test_pred_lr))
print("Testing RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred_lr)))

feature_importance_lr = pd.Series(linear_reg_model.coef_, index=X.columns)
feature_importance_sorted_lr = feature_importance_lr.sort_values(ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x=feature_importance_sorted_lr.values, y=feature_importance_sorted_lr.index, palette='viridis')
plt.title('Feature Importance (Coefficients) for Linear Regression')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.xticks('rotation')