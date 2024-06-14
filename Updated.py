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
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import scipy.stats
from PIL import Image
import requests
from io import BytesIO
import base64
import os



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
fallback_image_path = "C:/Users/beami/Desktop/Project Final/fallback_image.jpg"

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
pages = ["Introduction", "Data Exploration", "Data Visualization", "Modelling", "Prediction", "Conclusion"]
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


#Data Audit, object types
data_types = data.dtypes



# Data Exploration Page
if page == pages[1]:
    st.header("🔍 Data Exploration")
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
    st.write("The histogram above illustrates the distribution of the 'Life Ladder' variable, which represents happiness scores.")
    st.write("The distribution appears to be approximately normal, suggesting a central tendency around certain happiness levels.")
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
    st.header("📊 Data Visualisation")

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

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x='Log GDP per capita', y='Life Ladder')
    plt.title('Scatter Plot of Life Ladder vs Log GDP per capita')
    st.pyplot(plt)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='Regional indicator', y='Life Ladder')
    plt.title('Box Plot of Life Ladder by Regional Indicator')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    plt.figure(figsize=(8, 6), dpi=100)
    average_life_ladder_by_region = data.groupby('Regional indicator')['Life Ladder'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=average_life_ladder_by_region, x='Life Ladder', y='Regional indicator')
    plt.title('Average Life Ladder by Regional Indicator')
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
    correlation_matrix = data[['Life Ladder', 'Log GDP per capita', 'Social support', 'Healthy life expectancy at birth',
                               'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

 # Modelling Page
if page == pages[3]:
    st.header(" Modelling")

    st.subheader("Objectif")
    st.write("Predict the happiness score, or life ladder, of a geographical region using the best machine learning model. ")
    
    st.write("")
    st.write("")

    if st.button("Models developped ") :
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
    if st.button("Decision Tree Model ") :
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
    if st.button("Linear Regression Model ") :
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
            #Plot feature importance (coefficients)
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
            plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], color='red', linestyle='--', label='Ideal Line')
            plt.title('Linear Regression: Training Data - Actual vs Predicted')
            plt.xlabel('Actual Life Ladder')
            plt.ylabel('Predicted Life Ladder')
            plt.legend()
            st.pyplot(fig)  # Display the plot in Streamlit using st.pyplot()
            # Create scatter plot for testing data
            fig, ax = plt.subplots(figsize=(10, 8))
            plt.scatter(y_test, y_test_pred, color='green', label='Actual vs Predicted')
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Line')
            plt.title('Linear Regression: Testing Data - Actual vs Predicted')
            plt.xlabel('Actual Life Ladder')
            plt.ylabel('Predicted Life Ladder')
            plt.legend()
            st.pyplot(fig)  # Display the plot in Streamlit using st.pyplot()
            st.write("Conclusion on Linear Regression Model:")
            st.write("Training R² of 0.769 and Testing R² of 0.733 suggest that the model explains approximately 76.9% and 73.3% of the variance in the target variable, respectively. These values are relatively high, indicating a good fit of the model to the data.")
            st.write("Overall, these metrics indicate that the linear regression model performs reasonably well in predicting the Life Ladder score based on the provided features. However, there is still room for improvement, especially considering potential complexities and nuances in the data that may not be captured by a linear model.")
    # Buttons for individual models
    if st.button("Gradient Boosting Model"):
        st.write("Add code for Gradient Boosting Model here.")
    if st.button("LassoCV Model"):
        st.write("Add code for LassoCV Model here.")
    if st.button("Random Forest"):
        st.write("Add code for Random Forest Model here.")
    if st.button("Ridge"):
        st.write("Add code for Ridge Model here.")
    import streamlit as st
    import pandas as pd
    # Function to display the conclusion and table
    def display_conclusion_and_table():
        data = {
        "Model Name": ["Linear Regression", "Decision Tree", "Gradient Boosting", "LassoCV", "Random Forest", "Ridge"],
        "Parameters": ["Default", "MaxDepth=7", "MaxDepth=7", "Alpha = 1", "MaxDepth=7", "Alpha=1"],
        "R2 Train": [0.7694, 0.8342, 0.9878, 0.7423, 0.9192, 0.9029],
        "R2 Test": [0.7328, 0.765, None, 0.7328, 0.8533, 0.8647],
        "MAE Train": [0.445, 0.3523, None, None, 0.2441, 0.2627],
        "MAE Test": [0.445, 0.4196, None, None, 0.327, 0.3048],
        "MSE Train": [0.3344, 0.2044, 0.3261, 0.4728, 0.0996, 0.1197],
        "MSE Test": [0.3344, 0.2935, None, 0.3685, 0.1836, 0.1693],
        "RMSE Train": [0.5782, 0.4521, 0.1868, 0.607, 0.3155, 0.346],
        "RMSE Test": [0.5782, 0.5417, None, 0.607, 0.4285, 0.4115]
    }
        # Convert the data to DataFrame
        df = pd.DataFrame(data)
        # Adjust the index to start from 1
        df.index = pd.RangeIndex(start=1, stop=len(df) + 1, step=1)
        # Display the conclusion
        st.write("**Conclusion:**")
        st.write("Considering the data in the comparative chart above, we can observe that for the test set the **Ridge Model** created the best result throughout all metrics of all models:")
        st.write("It accurately explained 86.47% of the variance of the test set and also minimizes the error metrics to 0.4114 RMSE and 0.3048 MAE.")
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
    
    

