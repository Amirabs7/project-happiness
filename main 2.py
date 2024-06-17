## Installing Data and Importing Libraries

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import scipy.stats
from PIL import Image
import requests
from io import BytesIO
import base64
import os
import pickle


# Set up the page
st.set_page_config(
    page_title="International Happiness Accross Nations",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# Function to load an image from file path
def load_image_from_path(image_path):
    try:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            return img
        else:
            st.error(f"Image file not found at: {image_path}")
            return None
    except (IOError, OSError) as e:
        st.error(f"Error opening image: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None


# File path of the local image
fallback_image_path = "fallback_image.jpg"

# Load the image from file path
image = load_image_from_path(fallback_image_path)

# Display the image in Streamlit if successfully loaded
if image:
    st.image(image, caption="Fallback Image", use_column_width=True)
else:
    st.write("Failed to load the image from file path.")

# Sidebar Configuration and other pages omitted for brevity


# Sidebar Configuration
st.sidebar.title("Summary")
pages = ["Introduction", "Data Exploration", "Data Visualization", "Modelling", "Prediction and conclusion"]
page = st.sidebar.radio("Go to", pages)
st.sidebar.markdown(
    """
    - **Course**: Data Analyst
    - **Type**: Bootcamp
    - **Month**: April 2024
    - **Group**:
        - Amira Ben Salem
        - Julian Buß
        - Subash Chandra Pal
    """
)

# Introduction Page
if page == pages[0]:
    st.caption("""**Course**: Data Analyst
    | **Type**: Bootcamp
    | **Month**: April 2024
    | **Group**: Amira Ben Salem, Julian Buß, Subash Chandra Pal
    """)

    # Introduction page content
    st.markdown("""
    <div class="intro-container">
        <h1 class="intro-header">👋 Welcome to the International Happiness Report Analysis </h1>
        <h2 class="intro-subheader">Discover What Makes People Happy accross nations </h2>
    </div>
    """, unsafe_allow_html=True)

# Load data
data = pd.read_csv("merged_df_happy (1).csv")

############################################# slightly different preprocessing due to sickness and time restrictions
# Define a dictionary mapping countries to regions
merged_df_happy_JB= data

country_to_region = {'Angola': 'Sub-Saharan Africa',
                     'Belize': 'Latin America and Caribbean',
                     'Bhutan': 'South Asia',
                     'Central African Republic': 'Sub-Saharan Africa',
                     'Congo (Kinshasa)': 'Sub-Saharan Africa',
                     'Cuba': 'Latin America and Caribbean',
                     'Djibouti': 'Sub-Saharan Africa',
                     'Guyana': 'Latin America and Caribbean',
                     'Oman': 'Middle East and North Africa',
                     'Qatar': 'Middle East and North Africa',
                     'Somalia': 'Sub-Saharan Africa',
                     'Somaliland region': 'Sub-Saharan Africa',
                     'South Sudan': 'Sub-Saharan Africa',
                     'Sudan': 'Sub-Saharan Africa',
                     'Suriname': 'Latin America and Caribbean',
                     'Syria': 'Middle East and North Africa',
                     'Trinidad and Tobago': 'Latin America and Caribbean'}

# Iterate over the dataset and fill missing regions based on the dictionary
for country, region in country_to_region.items():
  merged_df_happy_JB. loc[merged_df_happy_JB['Country name'] == country, 'Regional indicator'] = region

# Train-Test-Split
y=merged_df_happy_JB['Life Ladder']
x=merged_df_happy_JB.drop('Life Ladder',axis=1)

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42)
print('Train set', x_train.shape)
print('Test set', x_test.shape)

#Fill NaN


num_col=['Standard error of ladder score', 'upperwhisker', 'lowerwhisker',
       'Log GDP per capita', 'Social support',
       'Healthy life expectancy at birth', 'Freedom to make life choices',
       'Generosity', 'Perceptions of corruption', 'Ladder score in Dystopia',
       'Explained by: Log GDP per capita', 'Explained by: Social support',
       'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption',
       'Dystopia + residual', 'Positive affect', 'Negative affect']

for col in num_col:
    # Fill missing values in x_train with the mean of the group in x_train
    mean_values = x_train.groupby('Regional indicator')[col].transform('mean')
    x_train[col] = x_train[col].fillna(mean_values)

    # Fill missing values in x_test with the mean of the group in x_train
    mean_values_test = x_test['Regional indicator'].map(x_train.groupby('Regional indicator')[col].mean())
    x_test[col] = x_test[col].fillna(mean_values_test)


x_train= x_train.reset_index(drop=True)
x_test= x_test.reset_index(drop=True)
y_train= y_train.reset_index(drop=True)
y_test= y_test.reset_index(drop=True)


x_train=pd.get_dummies(x_train)
x_test=pd.get_dummies(x_test)
all_train = x_train.columns.tolist()
all_test = x_test.columns.tolist()
max_len = max(len(all_train), len(all_test))
for i in range(max_len):
    train_col = all_train[i] if i < len(all_train) else ""
    test_col = all_test[i] if i < len(all_test) else ""
x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
columns_to_drop = ['Standard error of ladder score', 'upperwhisker', 'lowerwhisker',
                   'Ladder score in Dystopia', 'Explained by: Log GDP per capita',
                   'Explained by: Social support', 'Explained by: Healthy life expectancy',
                   'Explained by: Freedom to make life choices', 'Explained by: Generosity',
                   'Explained by: Perceptions of corruption', 'Dystopia + residual',
                   'Positive affect', 'Negative affect']

x_train_JB = x_train.drop(columns_to_drop, axis=1)
x_test_JB = x_test.drop(columns_to_drop, axis=1)

y_train_JB= y_train
y_test_JB= y_test
####################################################################################################


# Define the list of variables to drop
variables_to_drop = [
    'Standard error of ladder score',
    'upperwhisker',
    'lowerwhisker',
    'Ladder score in Dystopia',
    'Explained by: Log GDP per capita',
    'Explained by: Social support',
    'Explained by: Healthy life expectancy',
    'Explained by: Freedom to make life choices',
    'Explained by: Generosity',
    'Explained by: Perceptions of corruption',
    'Dystopia + residual'
]

# Drop the specified variables from the DataFrame
data.drop(columns=variables_to_drop, inplace=True)

from sklearn.impute import SimpleImputer

# Initialize SimpleImputer with mean strategy
imputer = SimpleImputer(strategy='mean')

# Define columns with missing values
columns_with_missing = ['Log GDP per capita', 'Social support',
                        'Healthy life expectancy at birth',
                        'Freedom to make life choices',
                        'Generosity', 'Perceptions of corruption',
                        'Positive affect', 'Negative affect']

# Impute missing values
data[columns_with_missing] = imputer.fit_transform(data[columns_with_missing])

# List of countries without regional indicators
countries_in_question = ['Angola', 'Belize', 'Bhutan', 'Central African Republic',
                         'Congo (Kinshasa)', 'Cuba', 'Djibouti', 'Guyana', 'Oman', 'Qatar',
                         'Somalia', 'Somaliland region', 'South Sudan', 'Sudan', 'Suriname',
                         'Syria', 'Trinidad and Tobago']

# Check if any of the countries in question have missing regional indicators
missing_regions = data[data['Country name'].isin(countries_in_question) & data['Regional indicator'].isnull()]

# Data Audit, object types
data_types = data.dtypes

# Data Exploration Page
if page == pages[1]:
    st.header("Data Exploration")
    st.subheader("Dataset on happiness around the world between 2005 et 2021")

    st.write("Here are the first few rows of the dataset, providing an initial glimpse of the data:")
    st.write(data.head())

    st.write("The shape of the dataset indicates the number of rows and columns present:")
    st.write("- Number of rowns :", data.shape[0])
    st.write("- Number of columns :", data.shape[1])

    st.write("These are the columns present in the dataset, each representing different attributes:")
    st.write(data.columns)

    st.write("The data types of each column describe the format and nature of the data stored:")
    st.write(data.dtypes)

    logged_gdp_per_capita = data['Log GDP per capita']

    life_ladder = data['Life Ladder']

    plt.figure(figsize=(8, 6))
    plt.hist(life_ladder, bins=20, color='skyblue', edgecolor='black')
    plt.title('Distribution of Life Ladder')
    plt.xlabel('Life Ladder')
    plt.ylabel('Frequency')
    st.pyplot(plt)
    st.write(
        "The histogram above illustrates the distribution of the 'Life Ladder' variable, which represents happiness scores.")
    st.write(
        "The distribution appears to be approximately normal, suggesting a central tendency around certain happiness levels.")
    # Shapiro-Wilk Test
    # # Extract the "Life Ladder" data from the dataset
    life_ladder_data = data['Life Ladder']
    # Perform Shapiro-Wilk test for Life Ladder
    statistic, p_value = stats.shapiro(life_ladder_data)
    # Explanation for the Shapiro-Wilk Test
    st.write("""We will use the Shapiro-Wilk test to confirm the distribution of the life ladder variable. This test checks if a sample comes from a normal distribution, providing a test statistic (0.9877) and a p-value (1.8879e-12).
             - The test statistic close to 1 suggests normality.
             - The very small p-value (< 0.05) strongly rejects the null hypothesis of normality.
             - Despite the high test statistic, the tiny p-value indicates that the "life_ladder" data does not follow a normal distribution.
             """)
    st.write(f"**Shapiro-Wilk Test Statistic for Life Ladder:** {statistic}")
    st.write(f"**P-value for Life Ladder:** {p_value}")

# Data Visualisation Page
if page == pages[2]:
    st.header("Data Visualisation")
    # Filter data for the year 2021
    df_2021 = data[data['year'] == 2021]
    # Filter data for the year 2021
    # # Fill missing values using the regional average
    df_2021['Life Ladder'] = df_2021.groupby('Regional indicator')['Life Ladder'].transform(
        lambda x: x.fillna(x.mean()))
    # Identify countries still missing Life Ladder values
    missing_countries = df_2021[df_2021['Life Ladder'].isnull()]['Country name'].tolist()
    # Ensure the correct column names and no remaining NaN values
    df_2021 = df_2021[['Country name', 'Life Ladder']]
    # Create a copy of the dataframe to work with
    df_filled = df_2021.copy()
    # Set a placeholder value for missing data (choose a value outside the expected range)
    placeholder_value = -999
    # Fill NaN values with the placeholder value
    df_filled['Life Ladder'].fillna(placeholder_value, inplace=True)
    # Create the map with conditional coloring
    fig = px.choropleth(
        df_filled,
        locations="Country name",
        locationmode='country names',
        color="Life Ladder",
        hover_name="Country name",
        color_continuous_scale=px.colors.sequential.Plasma,  # Use a predefined colorscale
        range_color=(df_filled['Life Ladder'].min(), df_filled['Life Ladder'].max()),  # Specify actual data range
        labels={'Life Ladder': 'Happiness Index'},
        color_continuous_midpoint=placeholder_value, )  # Ensure the midpoint aligns with the placeholder value)
    # Update colorscale to map the placeholder_value to gray
    fig.update_traces(marker_line_color='rgb(8,48,107)', marker_line_width=1)
    fig.update_layout(
        title_text='Happiness around the world in 2021',  # Title above the graph
        geo=dict(showframe=False, showcoastlines=False, projection_type='equirectangular')
    )
    # Streamlit app
    st.write("This map shows the Happiness Index (Life Ladder) for different countries in the year 2021. \
             Countries with no data are shown in white.")
    st.plotly_chart(fig)
    st.markdown("###### Healthy Life Expectancy at Birth for the period 2005 to 2021")
    data = data.dropna(subset=['Life Ladder', 'Healthy life expectancy at birth'])
    df_grouped = data.groupby(['Country name', 'Regional indicator']).agg({
        'Life Ladder': 'mean',
        'Healthy life expectancy at birth': 'mean'
    }).reset_index()

    fig = px.sunburst(df_grouped,
                      path=['Regional indicator', 'Country name'],
                      values='Life Ladder',
                      color='Healthy life expectancy at birth',
                      color_continuous_scale='RdBu',
                      color_continuous_midpoint=np.average(df_grouped['Healthy life expectancy at birth'],
                                                           weights=df_grouped['Life Ladder']))
    st.plotly_chart(fig)
    fig.update_layout(title_text='Happiness around the world in 2021')
    st.write("""
    The graph below shows the average life ladder score for several countries including the United States, Denmark, Netherlands, Canada, Sweden, Australia, Finland, Switzerland, Norway, New Zealand, Costa Rica, Israel, Venezuela, Austria, and Iceland. The life ladder score is a measure of subjective well-being that ranges from 0 to 10, with higher scores indicating greater happiness. The graph shows that Denmark has the highest average life ladder score over the time period covered by the graph, followed by Finland and Switzerland. The United States ranks tenth.
    """)

    df = data.dropna(subset=['Life Ladder'])
    df_top5 = df.groupby('year').apply(lambda x: x.nlargest(5, 'Life Ladder')).reset_index(drop=True)
    fig = px.line(df_top5,
                  x='year',
                  y='Life Ladder',
                  color='Country name',
                  line_group='Country name',
                  hover_name='Country name',
                  labels={'Life Ladder': 'Life Ladder Score', 'year': 'Year'},
                  title='Top 5 Happiest Countries (2005-2021)')
    st.plotly_chart(fig)

    st.write("""
    Given their consistently low scores, Burundi, Central African Republic, South Sudan, and Afghanistan likely remained among the unhappiest throughout the period.
    """)

    df_bottom5 = df.groupby('year').apply(lambda x: x.nsmallest(5, 'Life Ladder')).reset_index(drop=True)
    fig = px.line(df_bottom5,
                  x='year',
                  y='Life Ladder',
                  color='Country name',
                  line_group='Country name',
                  hover_name='Country name',
                  labels={'Life Ladder': 'Life Ladder Score', 'year': 'Year'},
                  title='Top 5 Unhappiest Countries (2005-2021)')
    st.plotly_chart(fig)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='Regional indicator', y='Life Ladder')
    plt.title('Box Plot of Life Ladder by Regional Indicator')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    plt.figure(figsize=(8, 6), dpi=100)
    average_life_ladder_by_region = data.groupby('Regional indicator')['Life Ladder'].mean().sort_values(
        ascending=False).reset_index()
    sns.barplot(data=average_life_ladder_by_region, x='Life Ladder', y='Regional indicator')
    plt.title('Average Life Ladder by Regional Indicator')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='Log GDP per capita', y='Life Ladder')
    plt.title('Scatter Plot of Life Ladder vs Log GDP per capita')
    st.pyplot(plt)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='Generosity', y='Life Ladder', alpha=0.7)
    plt.title('Scatter Plot: Life Ladder vs Generosity')
    st.pyplot(plt)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x='Freedom to make life choices', y='Life Ladder', alpha=0.7)
    plt.title('Scatter Plot: Life Ladder vs Freedom to make life choices')
    st.pyplot(plt)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=data, x='Perceptions of corruption', y='Life Ladder', alpha=0.7)
    plt.title('Scatter Plot: Life Ladder vs Perceptions of corruption')
    st.pyplot(plt)

    plt.figure(figsize=(10, 8))
    correlation_matrix = data[
        ['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
         'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

# Modelling Page
if page == pages[3]:
    st.header(" Modelling")

    st.subheader("Objectif")
    st.write(
        "To determine the best machine learning model for predicting the happiness score, or life ladder, using the available data from the World Happiness Report ")

    st.write("")
    st.write("")

    if st.button("Models developped "):
        st.subheader("Models ")
        st.markdown("""

                1- Linear Regression Model

                2- Decision Tree Model

                3- Gradiant Boosting Model

                4- LassoCV Model

                5- Random Forest 

                6- Ridge 


        """)

        st.write("")
        st.write("")

        st.subheader("Models creation and development")
        st.markdown("""
                    For each developed model, we followed these steps: :
                    1. Model instantiation
                    2. Training each model on training set  X_train and  y_train (80%, 20%)
                    3. Making predictions on test set  X_test and  y_test
                    4. Eavaluating model performance using specific metrics 
                    5. Interpreting features importance for each model
                    6. Visualizing and analyzing the results  
                """)

    st.write("")
    st.write("")
    if st.button("Decision Tree Model "):
        if 'dt_button' not in st.session_state:
            st.session_state.dt_button = False

        label_encoder = LabelEncoder()
        data['Regional indicator'] = label_encoder.fit_transform(data['Regional indicator'])
        data['Country name'] = label_encoder.fit_transform(data['Country name'])
        X = data.drop(['Life Ladder'], axis=1)
        y = data['Life Ladder']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        decision_tree_model = DecisionTreeRegressor(random_state=42)
        decision_tree_model.fit(X_train, y_train)
        train_predictions = decision_tree_model.predict(X_train)
        test_predictions = decision_tree_model.predict(X_test)
        r2_train = r2_score(y_train, train_predictions)
        mae_train = mean_absolute_error(y_train, train_predictions)
        mse_train = mean_squared_error(y_train, train_predictions)
        rmse_train = np.sqrt(mse_train)
        r2_test = r2_score(y_test, test_predictions)
        mae_test = mean_absolute_error(y_test, test_predictions)
        mse_test = mean_squared_error(y_test, test_predictions)
        rmse_test = np.sqrt(mse_test)
        st.subheader("Metrics Results ")
        st.write("Decision Tree Model training Results:")
        st.write("R²:", r2_train)
        st.write("MAE:", mae_train)
        st.write("MSE:", mse_train)
        st.write("RMSE:", rmse_train)
        st.write("Decision Tree Model testing Results::")
        st.write("R²:", r2_test)
        st.write("MAE:", mae_test)
        st.write("MSE:", mse_test)
        st.write("RMSE:", rmse_test)
        st.subheader("Features Importance ")
        feature_importances = decision_tree_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
        plt.title('Feature Importance')
        st.pyplot(plt)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train, train_predictions, color='blue', alpha=0.5)
        plt.title('Training Data: Target vs Prediction')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        st.subheader("Training and Testing : Target vs Prediction ")
        # 6 - Plot scatter plot of target vs prediction for training data
        plt.figure(figsize=(8, 6))
        plt.scatter(y_train, train_predictions, color='blue', alpha=0.5)
        plt.title('Training Data: Target vs Prediction')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        plt.grid(True)
        st.pyplot(plt)  # Use st.pyplot() instead of plt.show()
        # Plot scatter plot of target vs prediction for testing data
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, test_predictions, color='green', alpha=0.5)
        plt.title('Testing Data: Target vs Prediction')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        plt.grid(True)
        st.pyplot(plt)  # Use st.pyplot() instead of plt.show()
        st.write("""
     The initial creation of the decision tree model resulted in an R² score of 1, indicating a perfect fit. The three most important features within the model were GDP log per capita, healthy life expectancy at birth, and social support.
                 To address overfitting in the Decision Tree Model, several steps were taken:
                 **Limiting Maximum Depth**: 
                 The maximum depth of the decision tree was set to 5. This restricts the number of levels in the tree, simplifying the model and reducing overfitting.
                 - **Training and Evaluation**: The Decision Tree Model was trained on the imputed training data and evaluated on both the training and testing sets. R² scores were calculated to assess the model's performance in capturing the variance in the target variable.
                 - **Visualiztion of the Decision Tree**: The decision tree was visualized using the `plot_tree` function from the `sklearn.tree` module. This visualization provides insights into the structure of the decision tree and how it makes predictions, aiding in understanding its behavior and potential areas of improvement.
                 Overall, these adjustments help in mitigating overfitting and improving the generalization performance of the Decision Tree Model.
""")
        st.subheader("Adjusting Overfitting ")
        from sklearn.impute import SimpleImputer

        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        # Initialize a Decision Tree Regressor with a maximum depth
        max_depth = 5  # Set the maximum depth of the decision tree
        decision_tree_model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        # Train the Decision Tree Model
        decision_tree_model.fit(X_train_imputed, y_train)
        # Make predictions on the training and testing sets
        train_predictions = decision_tree_model.predict(X_train_imputed)
        test_predictions = decision_tree_model.predict(X_test_imputed)
        # Calculate R² for training and testing data
        train_r2 = r2_score(y_train, train_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Calculate MAE, MSE, and RMSE for training and testing data
        train_mae = mean_absolute_error(y_train, train_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        train_mse = mean_squared_error(y_train, train_predictions)
        test_mse = mean_squared_error(y_test, test_predictions)
        train_rmse = np.sqrt(train_mse)
        test_rmse = np.sqrt(test_mse)
        st.write("Training R²:", train_r2)
        st.write("Testing R²:", test_r2)
        st.write("Training MAE:", train_mae)
        st.write("Testing MAE:", test_mae)
        st.write("Training MSE:", train_mse)
        st.write("Testing MSE:", test_mse)
        st.write("Training RMSE:", train_rmse)
        st.write("Testing RMSE:", test_rmse)
        import streamlit as st
        import matplotlib.pyplot as plt
        from sklearn.tree import DecisionTreeRegressor, plot_tree

        # Plot the decision tree
        plt.figure(figsize=(12, 6))
        plot_tree(decision_tree_model, filled=True, feature_names=X.columns)
        st.pyplot(plt)
        if st.button("Toggle Decision Tree Model"):
            st.session_state.dt_button = not st.session_state.dt_button
        if st.session_state.dt_button:
            decision_tree_model()

    if st.button("Linear Regression Model "):
        label_encoder = LabelEncoder()
        data['Regional indicator'] = label_encoder.fit_transform(data['Regional indicator'])
        data['Country name'] = label_encoder.fit_transform(data['Country name'])
        X = data.drop(['Life Ladder'], axis=1)
        y = data['Life Ladder']
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Fit the linear regression model
        linear_reg_model = LinearRegression()
        linear_reg_model.fit(X_train, y_train)
        # Predict on training and testing sets
        y_train_pred = linear_reg_model.predict(X_train)
        y_test_pred = linear_reg_model.predict(X_test)
        # Calculate metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        mse = mean_squared_error(y_test, y_test_pred)
        rmse = mean_squared_error(y_test, y_test_pred, squared=False)
        st.subheader("Metrics Results ")
        st.write("Linear Regression Model Training Metrics:")
        st.write("R²:", train_r2)
        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("RMSE:", rmse)
        st.write("Linear Regression Model Testing Metrics:")
        st.write("R²:", test_r2)
        st.write("MAE:", mae)
        st.write("MSE:", mse)
        st.write("RMSE:", rmse)
        # Plot feature importance (coefficients)
        st.subheader("Features Importance ")
        feature_importance = pd.Series(linear_reg_model.coef_, index=X.columns)
        feature_importance_sorted = feature_importance.sort_values(ascending=False)
        plt.figure(figsize=(10, 8))  # Increase the figure size for better visibility
        fig, ax = plt.subplots(figsize=(10, 8))  # Create a matplotlib figure and axis
        sns.barplot(x=feature_importance_sorted.values, y=feature_importance_sorted.index, palette='viridis', ax=ax)
        ax.set_title('Feature Importance (Absolute Coefficients) for Linear Regression')
        ax.set_xlabel('Absolute Coefficient Value')
        ax.set_ylabel('Feature')
        ax.tick_params(axis='x', rotation=45)  # Rotate the x-axis labels for better readability
        st.pyplot(fig)  # Display the plot in Streamlit using st.pyplot()
        # Create scatter plot for training data
        st.subheader("Training and Testing : Target vs Prediction ")
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(y_train, y_train_pred, color='blue', label='Actual vs Predicted')
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--',
                 label='Ideal Line')
        plt.title('Linear Regression: Training Data - Actual vs Predicted')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        plt.legend()
        st.pyplot(fig)  # Display the plot in Streamlit using st.pyplot()
        # Create scatter plot for testing data
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.scatter(y_test, y_test_pred, color='green', label='Actual vs Predicted')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--',
                 label='Ideal Line')
        plt.title('Linear Regression: Testing Data - Actual vs Predicted')
        plt.xlabel('Actual Life Ladder')
        plt.ylabel('Predicted Life Ladder')
        plt.legend()
        st.pyplot(fig)  # Display the plot in Streamlit using st.pyplot()
        st.write("Conclusion on Linear Regression Model:")
        st.write(
            "Training R² of 0.769 and Testing R² of 0.733 suggest that the model explains approximately 76.9% and 73.3% of the variance in the target variable, respectively. These values are relatively high, indicating a good fit of the model to the data.")
        st.write(
            "Overall, these metrics indicate that the linear regression model performs reasonably well in predicting the Life Ladder score based on the provided features. However, there is still room for improvement, especially considering potential complexities and nuances in the data that may not be captured by a linear model.")

    # Buttons for individual models
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    # Ensure Streamlit session state
    if st.button("Gradiant Boosting Model "):
        # Define target variable (y) and features (X)
        y = data['Life Ladder']
        X = data[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                  'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Define the parameter grid
        param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10]}
        # Initialize the Gradient Boosting Regressor
        gbr = GradientBoostingRegressor(random_state=42)
        # Initialize Grid Search with cross-validation
        grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=-1)
        # Fit Grid Search to the data
        grid_search.fit(X_train, y_train)
        # Best parameter found by Grid Search
        best_max_depth = grid_search.best_params_['max_depth']
        st.write(f"Best max_depth: {best_max_depth}")
        # Train the Gradient Boosting Regressor with the best max_depth
        gbr = GradientBoostingRegressor(max_depth=best_max_depth, random_state=42)
        gbr.fit(X_train, y_train)
        # Predictions
        y_train_pred_gbr = gbr.predict(X_train)
        y_test_pred_gbr = gbr.predict(X_test)
        # Metrics for Gradient Boosting Regressor

        st.markdown("### Metrics Results")
        r2_train_gbr = r2_score(y_train, y_train_pred_gbr)
        st.write("R² Train = 0.987827")
        r2_test_gbr = r2_score(y_test, y_test_pred_gbr)
        st.write("R² Test = 0.850765")
        mae_test_gbr = mean_absolute_error(y_test, y_test_pred_gbr)
        st.write("MAE Test = 0.326058")
        mae_train_gbr = mean_absolute_error(y_train, y_train_pred_gbr)
        st.write("MAE Train = 0.091736")
        mse_test_gbr = mean_squared_error(y_test, y_test_pred_gbr)
        st.write("MSE Test = 0.186756")
        mse_train_gbr = mean_squared_error(y_train, y_train_pred_gbr)
        st.write("MSE Train = 0.015003")
        rmse_test_gbr = np.sqrt(mse_test_gbr)
        st.write("RMSE Test = 0.432153")
        rmse_train_gbr = np.sqrt(mse_train_gbr)
        st.write("RMSE Train = 0.122488")
        # Feature importances from Gradient Boosting Regressor
        st.markdown("### Feature Importance")
        feature_importances = gbr.feature_importances_
        # Plotting feature importances
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.barplot(x=feature_importances, y=X.columns, ax=ax1)
        ax1.set_title('Feature Importances from Gradient Boosting Regressor')
        ax1.set_xlabel('Importance')
        ax1.set_ylabel('Feature')
        st.pyplot(fig1)

        # Scatter plot of target vs predictions for Gradient Boosting Regressor
        st.markdown("### Training and Testing")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(y_test, y_test_pred_gbr, alpha=0.5)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax2.set_title('Gradient Boosting Regressor: Target vs Predictions')
        ax2.set_xlabel('Actual')
        ax2.set_ylabel('Predicted')
        st.pyplot(fig2)

        # Display model metrics
        st.markdown("""
         <div style="font-size: 14px;">
         <b>Gradient Boosting:</b><br>
         - <b>R² Train and Test</b> The Gradient Boosting model has a higher R² value for both training (0.9878) and testing (0.8508) compared to the LassoCV model (training: 0.7423, testing: 0.7055). This indicates that the Gradient Boosting model explains a higher proportion of the variance in the target variable both in training and testing sets. The Gradient Boosting model's R² values show a small decrease from training to testing, suggesting it generalizes well to unseen data.<br>
         - <b>MAE Train and Test:</b>  The Gradient Boosting model has a significantly lower MAE for both training (0.0917) and testing (0.3261) compared to the LassoCV model (training: 0.4446, testing: 0.4728). This indicates that the Gradient Boosting model's predictions are, on average, closer to the actual values.<br>
         - <b>MSE and RMSE :</b> The Gradient Boosting model also has a much lower MSE for both training (0.0150) and testing (0.1868) compared to the LassoCV model (training: 0.3176, testing: 0.3685). Lower MSE values indicate better model performance in terms of the average squared difference between the predicted and actual values.Similarly, the Gradient Boosting model has a lower RMSE for both training (0.1225) and testing (0.4322) compared to the LassoCV model (training: 0.5636, testing: 0.6070). Lower RMSE values indicate that the Gradient Boosting model has better overall predictive accuracy.<br>
         - <b>Conclusion :</b>Overall Performance: The Gradient Boosting model outperforms the LassoCV model across all metrics (R², MAE, MSE, RMSE) for both training and testing datasets. It has higher R² values and lower error metrics, indicating more accurate and reliable predictions. Generalization: The Gradient Boosting model shows good generalization from training to testing data, as indicated by the relatively small decrease in R² and moderate increase in error metrics. Prediction Accuracy: The significantly lower MAE, MSE, and RMSE values for the Gradient Boosting model suggest it is more precise in predicting the target variable compared to the LassoCV model.
         </div>
         """, unsafe_allow_html=True)

    if st.button("LassoCV Model"):
        from sklearn.linear_model import LassoCV
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import make_regression

        # Define target variable (y) and features (X)
        y = data['Life Ladder']
        X = data[['Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                  'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']]
        from sklearn.model_selection import train_test_split

        # Split data into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        from sklearn.linear_model import LassoCV

        # Training the LassoCV model
        lasso = LassoCV(cv=5, random_state=42)
        lasso.fit(X_train, y_train)
        # Predictions
        y_train_pred_lasso = lasso.predict(X_train)
        y_test_pred_lasso = lasso.predict(X_test)
        from sklearn.metrics import r2_score
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error

        # Metrics for LassoCV
        r2_train_lasso = r2_score(y_train, y_train_pred_lasso)
        r2_test_lasso = r2_score(y_test, y_test_pred_lasso)
        mae_test_lasso = mean_absolute_error(y_test, y_test_pred_lasso)
        mae_train_lasso = mean_absolute_error(y_train, y_train_pred_lasso)
        mse_test_lasso = mean_squared_error(y_test, y_test_pred_lasso)
        mse_train_lasso = mean_squared_error(y_train, y_train_pred_lasso)
        rmse_train_lasso = np.sqrt(mse_train_lasso)
        rmse_test_lasso = np.sqrt(mse_test_lasso)
        # Feature Importance
        st.subheader('Feature Importance from LassoCV Model')
        feature_importances = np.abs(lasso.coef_)
        fig_feat, ax_feat = plt.subplots(figsize=(10, 6))
        sns.barplot(x=feature_importances, y=X.columns, ax=ax_feat)
        ax_feat.set_title('Feature Importances from LassoCV Model')
        ax_feat.set_xlabel('Importance')
        ax_feat.set_ylabel('Feature')
        plt.tight_layout()
        st.pyplot(fig_feat)
        # Plotting the results
        st.subheader('LassoCV: Target vs Predictions')
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_test_pred_lasso, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax.set_title('LassoCV: Target vs Predictions')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig)
        st.subheader('Metrics Results ')
        st.write('R² Train:', r2_train_lasso)
        st.write('R² Test:', r2_test_lasso)
        st.write('MAE Train:', mae_train_lasso)
        st.write('MAE Test:', mae_test_lasso)
        st.write('MSE Train:', mse_train_lasso)
        st.write('MSE Test:', mse_test_lasso)
        st.write('RMSE Train:', rmse_train_lasso)
        st.write('RMSE Test:', rmse_test_lasso)
        st.subheader('Conclusion ')
        st.write(
            "R² Train (0.742295): This indicates that the LassoCV model explains approximately 74.23% of the variance in the training data. This is significantly lower than Gradient Boosting, suggesting that LassoCV is not capturing the relationship as well.")
        st.write(
            "R² Test (0.705542): This indicates that the model explains approximately 70.55% of the variance in the test data. Although this is slightly lower than the training R², it is close, indicating reasonable generalization.")
        st.write(
            "MAE Test(0.472798): The mean absolute error is higher compared to Gradient Boosting, suggesting that on average, the model's predictions are off by about 0.473 units.")
        st.write(
            "MSE Test(0.368493) and RMSE Test(0.432153): These metrics are higher than those for Gradient Boosting, indicating that LassoCV has worse predictive performance.")
        st.write(
            "MSE Train(0.317628) and RMSE Train(0.607036): These metrics are higher than those for Gradient Boosting, indicating that LassoCV has worse predictive performance.")

        st.write(
            "Overall, these metrics indicate that the LassoCV model performs reasonably well in predicting the Life Ladder score based on the provided features. However, there is still room for improvement, especially considering potential complexities and nuances in the data that may not be captured by a linear model.")

    if st.button("Random Forest"):
         from sklearn.ensemble import RandomForestRegressor
         from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
         #################################################### Train Random Forest model with max_depth=2
         rf_model_JB = RandomForestRegressor(max_depth=2,n_estimators=100, random_state=42)
         rf_model_JB.fit(x_train_JB, y_train_JB)
         # Predictions
         rf_predictions_test_JB = rf_model_JB.predict(x_test_JB)
         rf_predictions_train_JB = rf_model_JB.predict(x_train_JB)
         # Evaluation training set
         rf_mse_train_JB = mean_squared_error(y_train_JB, rf_predictions_train_JB)
         rf_rmse_train_JB = np.sqrt(rf_mse_train_JB)  # Calculate RMSE
         rf_r2_train_JB = r2_score(y_train_JB, rf_predictions_train_JB)
         rf_mae_train_JB = mean_absolute_error(y_train_JB, rf_predictions_train_JB)
         # Evaluation test set
         rf_mse_test_JB = mean_squared_error(y_test_JB, rf_predictions_test_JB)
         rf_rmse_test_JB = np.sqrt(rf_mse_test_JB)  # Calculate RMSE
         rf_r2_test_JB = r2_score(y_test_JB, rf_predictions_test_JB)
         rf_mae_test_JB = mean_absolute_error(y_test_JB, rf_predictions_test_JB)
         # Font size adjustments
         base_fontsize = 10
         new_fontsize = base_fontsize * 2  # Increase by 100%
         # Calculate the required height to maintain a 3:4 aspect ratio for the combined plot
         original_plot_width = 8  # Set this to the original width
         combined_width = original_plot_width * 2  # Since we have two plots side by side
         combined_height = combined_width * 3 / 4
         st.subheader('Random Forest: Target vs Predictions')
         # Create a figure with two subplots side by side
         fig_rf, ax_rf = plt.subplots(1, 2, figsize=(combined_width, combined_height))
         # Scatter plot for Random Forest predictions on training set without Scaling
         ax_rf[0].scatter(y_train_JB, rf_predictions_train_JB, color='blue', label='Random Forest Predictions', alpha=0.5)
         ax_rf[0].plot([min(y_train_JB), max(y_train_JB)], [min(y_train_JB), max(y_train_JB)], color='red', linestyle='--', label='Perfect Predictions')
         ax_rf[0].set_title('RF: Training Set (max_depth=2)', fontsize=new_fontsize)
         ax_rf[0].set_xlabel('Actual Life Ladder', fontsize=new_fontsize)
         ax_rf[0].set_ylabel('Predicted Life Ladder', fontsize=new_fontsize)
         ax_rf[0].legend(fontsize=new_fontsize)
         ax_rf[0].grid(True)
         ax_rf[0].tick_params(axis='both', which='major', labelsize=new_fontsize)
         # Scatter plot for Random Forest predictions on test set without Scaling
         ax_rf[1].scatter(y_test_JB, rf_predictions_test_JB, color='blue', label='Random Forest Predictions', alpha=0.5)
         ax_rf[1].plot([min(y_test_JB), max(y_test_JB)], [min(y_test_JB), max(y_test_JB)], color='red', linestyle='--', label='Perfect Predictions')
         ax_rf[1].set_title('RF: Testing Set (max_depth=2)', fontsize=new_fontsize)
         ax_rf[1].set_xlabel('Actual Life Ladder', fontsize=new_fontsize)
         ax_rf[1].set_ylabel('Predicted Life Ladder', fontsize=new_fontsize)
         ax_rf[1].legend(fontsize=new_fontsize)
         ax_rf[1].grid(True)
         ax_rf[1].tick_params(axis='both', which='major', labelsize=new_fontsize)
         # Match the axis limits
         ax_rf[1].set_xlim(ax_rf[0].get_xlim())
         ax_rf[1].set_ylim(ax_rf[0].get_ylim())
         plt.tight_layout()
         plt.show()
         st.pyplot(fig_rf)
         # Feature Importance
         st.subheader('Feature Importance from Random Forest Model')
         feature_importances_rf = rf_model_JB.feature_importances_
         fig_feat_rf, ax_feat_rf = plt.subplots(figsize=(10, 6))
         sns.barplot(x=feature_importances_rf, y=x.columns, ax=ax_feat_rf)
         ax_feat_rf.set_title('Feature Importances from Random Forest Model')
         ax_feat_rf.set_xlabel('Importance')
         ax_feat_rf.set_ylabel('Feature')
         plt.tight_layout()
         st.pyplot(fig_feat_rf)
         # Conclusion
         st.subheader('Metrics')
         st.write(f"R² Train: {rf_r2_train_JB:.4f}")
         st.write(f"R² Test: {rf_r2_test_JB:.4f}")
         st.write(f"MAE Train: {rf_mae_train_JB:.4f}")
         st.write(f"MAE Test: {rf_mae_test_JB:.4f}")
         st.write(f"MSE Train: {rf_mse_train_JB:.4f}")
         st.write(f"MSE Test: {rf_mse_test_JB:.4f}")
         st.write(f"RMSE Train: {rf_rmse_train_JB:.4f}")
         st.write(f"RMSE Test: {rf_rmse_test_JB:.4f}")
         st.subheader('Conclusion')
         st.write("The plots show a rather low congruene of the model with the red dotted line. The metrics confirm a moderate fit and generalization for the Random Forest Regression Model, suggesting that there is room for improvement in prediction accuracy.")

         ################################################################################ Train Random Forest model with max_depth=7
         rf_model_JB = RandomForestRegressor(max_depth=7,n_estimators=100, random_state=42)
         rf_model_JB.fit(x_train_JB, y_train_JB)
         # Predictions
         rf_predictions_test_JB = rf_model_JB.predict(x_test_JB)
         rf_predictions_train_JB = rf_model_JB.predict(x_train_JB)
         # Evaluation training set
         rf_mse_train_JB = mean_squared_error(y_train_JB, rf_predictions_train_JB)
         rf_rmse_train_JB = np.sqrt(rf_mse_train_JB)  # Calculate RMSE
         rf_r2_train_JB = r2_score(y_train_JB, rf_predictions_train_JB)
         rf_mae_train_JB = mean_absolute_error(y_train_JB, rf_predictions_train_JB)
         # Evaluation test set
         rf_mse_test_JB = mean_squared_error(y_test_JB, rf_predictions_test_JB)
         rf_rmse_test_JB = np.sqrt(rf_mse_test_JB)  # Calculate RMSE
         rf_r2_test_JB = r2_score(y_test_JB, rf_predictions_test_JB)
         rf_mae_test_JB = mean_absolute_error(y_test_JB, rf_predictions_test_JB)
         # Font size adjustments
         base_fontsize = 10
         new_fontsize = base_fontsize * 2  # Increase by 100%
         # Calculate the required height to maintain a 3:4 aspect ratio for the combined plot
         original_plot_width = 8  # Set this to the original width
         combined_width = original_plot_width * 2  # Since we have two plots side by side
         combined_height = combined_width * 3 / 4
         st.subheader('Random Forest Adjustment: Target vs Predictions')
         # Create a figure with two subplots side by side
         fig_rf, ax_rf = plt.subplots(1, 2, figsize=(combined_width, combined_height))
         # Scatter plot for Random Forest predictions on training set without Scaling
         ax_rf[0].scatter(y_train_JB, rf_predictions_train_JB, color='blue', label='Random Forest Predictions', alpha=0.5)
         ax_rf[0].plot([min(y_train_JB), max(y_train_JB)], [min(y_train_JB), max(y_train_JB)], color='red', linestyle='--', label='Perfect Predictions')
         ax_rf[0].set_title('RF: Training Set (max_depth=7)', fontsize=new_fontsize)
         ax_rf[0].set_xlabel('Actual Life Ladder', fontsize=new_fontsize)
         ax_rf[0].set_ylabel('Predicted Life Ladder', fontsize=new_fontsize)
         ax_rf[0].legend(fontsize=new_fontsize)
         ax_rf[0].grid(True)
         ax_rf[0].tick_params(axis='both', which='major', labelsize=new_fontsize)
         # Scatter plot for Random Forest predictions on test set without Scaling
         ax_rf[1].scatter(y_test_JB, rf_predictions_test_JB, color='blue', label='Random Forest Predictions', alpha=0.5)
         ax_rf[1].plot([min(y_test_JB), max(y_test_JB)], [min(y_test_JB), max(y_test_JB)], color='red', linestyle='--', label='Perfect Predictions')
         ax_rf[1].set_title('RF: Testing Set (max_depth=7)', fontsize=new_fontsize)
         ax_rf[1].set_xlabel('Actual Life Ladder', fontsize=new_fontsize)
         ax_rf[1].set_ylabel('Predicted Life Ladder', fontsize=new_fontsize)
         ax_rf[1].legend(fontsize=new_fontsize)
         ax_rf[1].grid(True)
         ax_rf[1].tick_params(axis='both', which='major', labelsize=new_fontsize)
         # Match the axis limits
         ax_rf[1].set_xlim(ax_rf[0].get_xlim())
         ax_rf[1].set_ylim(ax_rf[0].get_ylim())
         plt.tight_layout()
         plt.show()
         st.pyplot(fig_rf)
         # Feature Importance
         # st.subheader('Feature Importance from Random Forest Model')
         # feature_importances_rf = rf_model_JB.feature_importances_
         # fig_feat_rf, ax_feat_rf = plt.subplots(figsize=(10, 6))
         # sns.barplot(x=feature_importances_rf, y=x.columns, ax=ax_feat_rf)
         # ax_feat_rf.set_title('Feature Importances from Random Forest Model')
         # ax_feat_rf.set_xlabel('Importance')
         # ax_feat_rf.set_ylabel('Feature')
         # plt.tight_layout()
         # st.pyplot(fig_feat_rf)
         # Conclusion
         st.subheader('Metrics')
         st.write(f"R² Train: {rf_r2_train_JB:.4f}")
         st.write(f"R² Test: {rf_r2_test_JB:.4f}")
         st.write(f"MAE Train: {rf_mae_train_JB:.4f}")
         st.write(f"MAE Test: {rf_mae_test_JB:.4f}")
         st.write(f"MSE Train: {rf_mse_train_JB:.4f}")
         st.write(f"MSE Test: {rf_mse_test_JB:.4f}")
         st.write(f"RMSE Train: {rf_rmse_train_JB:.4f}")
         st.write(f"RMSE Test: {rf_rmse_test_JB:.4f}")
         st.subheader('Conclusion')
         st.write("After adjusting the tree depth, the model performs much better. The metrics suggest that the Random Forest model performs well in predicting the Life Ladder score based on the provided features. The R² values indicate that the model explains a significant portion of the variance in both training and test datasets. Additionally, the feature importance plot highlights which features have the most impact on the predictions.")
        

    if st.button("Ridge Model"):
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        ridge_model_JB = Ridge(alpha=1.0, random_state=42)
        ridge_model_JB.fit(x_train_JB, y_train_JB)
        # Predictions
        ridge_predictions_test_JB = ridge_model_JB.predict(x_test_JB)
        ridge_predictions_train_JB = ridge_model_JB.predict(x_train_JB)
        # Evaluation training set
        ridge_mse_train_JB = mean_squared_error(y_train_JB, ridge_predictions_train_JB)
        ridge_rmse_train_JB = np.sqrt(ridge_mse_train_JB)  # Calculate RMSE
        ridge_r2_train_JB = r2_score(y_train_JB, ridge_predictions_train_JB)
        ridge_mae_train_JB = mean_absolute_error(y_train_JB, ridge_predictions_train_JB)
        # Evaluation test set
        ridge_mse_test_JB = mean_squared_error(y_test_JB, ridge_predictions_test_JB)
        ridge_rmse_test_JB = np.sqrt(ridge_mse_test_JB)  # Calculate RMSE
        ridge_r2_test_JB = r2_score(y_test_JB, ridge_predictions_test_JB)
        ridge_mae_test_JB = mean_absolute_error(y_test_JB, ridge_predictions_test_JB)
        # Font size adjustments
        base_fontsize = 10
        new_fontsize = base_fontsize * 2  # Increase by 100%
        # Calculate the required height to maintain a 3:4 aspect ratio for the combined plot
        original_plot_width = 8  # Set this to the original width
        combined_width = original_plot_width * 2  # Since we have two plots side by side
        combined_height = combined_width * 3 / 4
        st.subheader('Ridge: Target vs Predictions')
        # Create a figure with two subplots side by side
        fig_ridge, ax_ridge = plt.subplots(1, 2, figsize=(combined_width, combined_height))
        # Scatter plot for Ridge predictions on training set without Scaling
        ax_ridge[0].scatter(y_train_JB, ridge_predictions_train_JB, color='blue', label='Ridge Predictions', alpha=0.5)
        ax_ridge[0].plot([min(y_train_JB), max(y_train_JB)], [min(y_train_JB), max(y_train_JB)], color='red', linestyle='--', label='Peridgeect Predictions')
        ax_ridge[0].set_title('Ridge: Training Set (alpha=1.0)', fontsize=new_fontsize)
        ax_ridge[0].set_xlabel('Actual Life Ladder', fontsize=new_fontsize)
        ax_ridge[0].set_ylabel('Predicted Life Ladder', fontsize=new_fontsize)
        ax_ridge[0].legend(fontsize=new_fontsize)
        ax_ridge[0].grid(True)
        ax_ridge[0].tick_params(axis='both', which='major', labelsize=new_fontsize)
        # Scatter plot for Ridge predictions on test set without Scaling
        ax_ridge[1].scatter(y_test_JB, ridge_predictions_test_JB, color='blue', label='Ridge Predictions', alpha=0.5)
        ax_ridge[1].plot([min(y_test_JB), max(y_test_JB)], [min(y_test_JB), max(y_test_JB)], color='red', linestyle='--', label='Perfect Predictions')
        ax_ridge[1].set_title('Ridge: Testing Set (alpha=1.0)', fontsize=new_fontsize)
        ax_ridge[1].set_xlabel('Actual Life Ladder', fontsize=new_fontsize)
        ax_ridge[1].set_ylabel('Predicted Life Ladder', fontsize=new_fontsize)
        ax_ridge[1].legend(fontsize=new_fontsize)
        ax_ridge[1].grid(True)
        ax_ridge[1].tick_params(axis='both', which='major', labelsize=new_fontsize)
        # Match the axis limits
        ax_ridge[1].set_xlim(ax_ridge[0].get_xlim())
        ax_ridge[1].set_ylim(ax_ridge[0].get_ylim())
        plt.tight_layout()
        plt.show()
        st.pyplot(fig_ridge)
        # Feature Importance
    
        # Plot feature importances (coefficients) for Ridge model
        ridge_coefficients = ridge_model_JB.coef_
        st.subheader('Feature Importances from Ridge Model')
        fig_feat_ridge, ax_feat_ridge = plt.subplots(figsize=(10, 6))
        
        # Sorting coefficients by absolute magnitude for better visualization
        sorted_coefficient_indices = np.argsort(np.abs(ridge_coefficients))
        sorted_coefficients = ridge_coefficients[sorted_coefficient_indices]
        sorted_features = x.columns[sorted_coefficient_indices]


        # Plotting using seaborn barplot
        sns.barplot(x=sorted_coefficients, y=sorted_features, ax=ax_feat_ridge, palette="viridis")


        # Customize plot labels and layout
        ax_feat_ridge.set_title('Feature Importances from Ridge')
        ax_feat_ridge.set_xlabel('Coefficient Magnitude')
        ax_feat_ridge.set_ylabel('Feature')
        plt.tight_layout()

        st.pyplot(fig_feat_ridge)


        # Conclusion
        st.subheader('Metrics')
        st.write(f"R² Train: {ridge_r2_train_JB:.4f}")
        st.write(f"R² Test: {ridge_r2_test_JB:.4f}")
        st.write(f"MAE Train: {ridge_mae_train_JB:.4f}")
        st.write(f"MAE Test: {ridge_mae_test_JB:.4f}")
        st.write(f"MSE Train: {ridge_mse_train_JB:.4f}")
        st.write(f"MSE Test: {ridge_mse_test_JB:.4f}")
        st.write(f"RMSE Train: {ridge_rmse_train_JB:.4f}")
        st.write(f"RMSE Test: {ridge_rmse_test_JB:.4f}")
        st.subheader('Conclusion')
        st.write("Overall, these metrics suggest that the Ridge regression model performs reasonably well in predicting the Life Ladder score based on the provided features. The R² values indicate that the model explains a significant portion of the variance in both training and test datasets.")

        


    # Function to display the conclusion and table
    def display_conclusion_and_table():
        data = {
            "Model Name": ["Linear Regression", "Decision Tree", "Gradient Boosting", "LassoCV", "Random Forest",
                           "Ridge"],
            "Parameters": ["Default", "MaxDepth=7", "MaxDepth=7", "Alpha = 1", "MaxDepth=7", "Alpha=1"],
            "R2 Train": [0.7694, 0.8342, 0.9878, 0.7423, 0.9192, 0.9029],
            "R2 Test": [0.7328, 0.765, 0.850765, 0.7328, 0.8533, 0.8647],
            "MAE Train": [0.445, 0.3523, 0.091736, 0.444561, 0.2441, 0.2627],
            "MAE Test": [0.445, 0.4196, 0.3261, 0.4728, 0.327, 0.3048],
            "MSE Train": [0.3344, 0.2044, 0.015003, 0.317628, 0.0996, 0.1197],
            "MSE Test": [0.3344, 0.2935, 0.186756, 0.3685, 0.1836, 0.1693],
            "RMSE Train": [0.5782, 0.4521, 0.122488, 0.563585, 0.3155, 0.346],
            "RMSE Test": [0.5782, 0.5417, 0.432153, 0.607, 0.4285, 0.4115]
        }
        # Convert the data to DataFrame
        df = pd.DataFrame(data)
        # Adjust the index to start from 1
        df.index = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
        # Display the conclusion
        st.write("**Conclusion:**")
        st.write(
            "Considering the data in the comparative chart above, we can observe that for the test set the **Ridge Model** created the best result throughout all metrics of all models:")
        st.write(
            "It accurately explained 86.47% of the variance of the test set and also minimizes the error metrics to 0.4114 RMSE and 0.3048 MAE.")
        st.write("This indicates that the Ridge model performs the best among the considered models.")
        # Filter out None values which are causing empty rows
        df_filtered = df.dropna(how='all')
        # Manually format Ridge model metrics to be bold
        ridge_idx = df_filtered[df_filtered['Model Name'] == 'Ridge'].index
        styled_df = df_filtered.style.applymap(
            lambda x: 'font-weight: bold' if df_filtered.loc[ridge_idx, :].isin([x]).any().any() else '',
            subset=pd.IndexSlice[ridge_idx, :]
        )
        # Display the table
        st.write("**Comparative Table of Model Metrics:**")
        st.write(styled_df)


    if __name__ == "__main__":
        # Button to display conclusion and table
        if st.button("Conclusion"):
            display_conclusion_and_table()




if page == pages[4]:
     
    import streamlit as st
    import numpy as np
    import pickle
    import pandas as pd
    
    # Function to load the model
    def charger_modele():
        try:
            with open('modele_rfr_up.pkl', 'rb') as file_model:
                model = pickle.load(file_model)
            return model
        except FileNotFoundError:
            st.error("Model file 'modele_rfr_up.pkl' not found. Please ensure the model file is in the same directory as this script.")
            return None
  
    # Function to load the scaler
    def charger_scaler():
        try:
            with open('scaler.pkl', 'rb') as fichier_scaler:
                scaler = pickle.load(fichier_scaler)
            return scaler
        except FileNotFoundError:
            st.error("Scaler file 'scaler.pkl' not found. Please ensure the scaler file is in the same directory as this script.")
            return None

    # Header and subheader for prediction
    st.header("Prediction")
    st.subheader('Prediction simulation with Ridge Model')

    # List of regions
    regions = ["Region_Central_and_Eastern_Europe", "Region_Commonwealth_of_Independent_States", 
               "Region_East_Asia", "Region_Latin_America_and_Caribbean", "Region_Middle_East_and_North_Africa",
               "Region_North_America_and_ANZ", "Region_South_Asia", "Region_Southeast_Asia", 
               "Region_Sub-Saharan_Africa", "Region_Western_Europe"]
    region_values = {region: 0 for region in regions}

    # Select region
    selected_region = st.selectbox('Select a region:', regions)
    region_values[selected_region] = 1
    region_values_list = list(region_values.values())

    # Input features
    caracteristique1 = float(st.slider("Log GDP per capita", 6.0, 13.0, 9.4))
    caracteristique2 = float(st.slider("Social support", 0.0, 1.0, 0.8))
    caracteristique3 = float(st.slider("Healthy Life expectancy at birth", 30.0, 80.0, 63.8))
    caracteristique4 = st.slider("Positive affect", 0.0, 1.0, 0.7)

    # Combine features into a single array
    caracteristiques = np.array([[caracteristique1, caracteristique2, caracteristique3, caracteristique4] + region_values_list])

    # Load model and scaler
    modele = charger_modele()
    scaler = charger_scaler()

    # Make prediction if model and scaler are loaded successfully
    if modele is not None and scaler is not None:
        try:
            caracteristiques_scaled = scaler.transform(caracteristiques)
            prediction = modele.predict(caracteristiques_scaled)
            prediction = np.round(prediction, 3)
            st.markdown(f"<p style='font-size:24px; font-weight:bold;'>Le Ladder Score serait de : {prediction[0]}</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

    # Example data for concrete cases
    data_pred = {
        'Variables': ['Log GDP per capita', 'Social support', 'Healthy Life expectancy at birth', 'Positive affect', 'Life Ladder'],
        'Costa Rica (2018 - Latin America and Caribbean)': [9.874, 0.876, 71.3, 0.870, 7.141],
        'Croatia (2018 - Central and Eastern Europe)': [10.224, 0.910, 70.2, 0.582, 5.536]
    }

    if st.checkbox("Check a concrete case"):
        tab = pd.DataFrame.from_dict(data_pred, orient='index')
        tab.columns = tab.iloc[0]
        tab = tab[1:]
        st.table(tab)

    
