import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

# Load the best models (which include the pipeline)
best_models = {
    'log_reg': joblib.load('models/log_reg_best_model.joblib'),
    'knn': joblib.load('models/knn_best_model.joblib'),
    'svm': joblib.load('models/svm_best_model.joblib'),
    'rf': joblib.load('models/rf_best_model.joblib'),
    'xgb': joblib.load('models/xgb_best_model.joblib'),
}


# Apply custom CSS
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
    padding: 10px;
    border-radius: 5px;
}
.stButton > button {
    background-color: #4CAF50;
    color: white;
    border-radius: 5px;
    padding: 10px 20px;
}
.stDataFrame {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)




# Sidebar for model selection
st.sidebar.title('Model Selection')
model_choice = st.sidebar.selectbox('Choose a model:', list(best_models.keys()))
st.sidebar.write("Select a model and upload your data.")


# Main app title
st.title('🚦Predictive Analysis of Traffic Crashes in Chicago: Crash Types')
st.write("Upload your data, choose a model, and start exploring the insights!")


# Display chosen model
st.write(f'You selected the {model_choice.upper()} model.')

# File upload for user input
# uploaded_file = st.file_uploader("Upload a CSV file with data points", type=["csv"])

uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])


if uploaded_file:
    # Read the CSV file into a pandas DataFrame
    df_used = pd.read_csv(uploaded_file)

    st.write("### Uploaded Data")
    st.write(df_used.head())
    st.write(f"Data shape: {df_used.shape}")

    df = df_used.drop("CRASH_TYPE",axis = 1)

    # Convert columns to datetime
    st.write("### Preprocessing Uploaded Data")
    try:
        df['CRASH_DATE'] = pd.to_datetime(df['CRASH_DATE'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        df['DATE_POLICE_NOTIFIED'] = pd.to_datetime(df['DATE_POLICE_NOTIFIED'], format='%m/%d/%Y %I:%M:%S %p', errors='coerce')
        st.success("Datetime conversion successful!")
    except Exception as e:
        st.error(f"Error in datetime conversion: {e}")

    # Show missing data summary
    missing_data = df.isnull().sum()
    if missing_data.any():
        st.warning("### Missing Data Summary")
        st.write(missing_data[missing_data > 0])

    # Data visualization options
    st.write("### Data Insights")
    if st.checkbox("Show Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 10))
        correlation_matrix = df.select_dtypes(include = ['int64','float64']).corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='viridis', square=True, cbar_kws={"shrink": .8})
        st.pyplot(fig)

    if st.checkbox("Show Data Distribution"):
        col_to_plot = st.selectbox("Select a column to visualize:", df.select_dtypes(include=['number']).columns)
        fig, ax = plt.subplots()
        sns.histplot(df[col_to_plot].dropna(), kde=True, ax=ax)
        st.pyplot(fig)

    if st.checkbox("Show Top Contributory Causes"):
        fig, ax = plt.subplots(figsize=(8, 5))
        top_causes = df_used['PRIM_CONTRIBUTORY_CAUSE'].value_counts().head(10)
        sns.barplot(y=top_causes.index, x=top_causes.values, palette="coolwarm", ax=ax)
        ax.set_title("Top 10 Contributory Causes of Crashes")
        ax.set_xlabel("Number of Crashes")
        ax.set_ylabel("Contributory Cause")
        st.pyplot(fig)

    if st.checkbox("Show Crashes by Day of the Week"):
        fig, ax = plt.subplots(figsize=(8, 5))
        day_counts = df_used['CRASH_DAY_OF_WEEK'].value_counts(sort=False)
        sns.barplot(x=day_counts.index, y=day_counts.values, palette="crest", ax=ax)
        ax.set_title("Crashes by Day of the Week")
        ax.set_xlabel("Day of the Week (1=Monday, 7=Sunday)")
        ax.set_ylabel("Number of Crashes")
        st.pyplot(fig)

    if st.checkbox("Show Weather Conditions Impact"):
       fig, ax = plt.subplots(figsize=(8, 5))
       weather_counts = df_used['WEATHER_CONDITION'].value_counts()
       sns.barplot(y=weather_counts.index, x=weather_counts.values, palette="mako", ax=ax)
       ax.set_title("Impact of Weather Conditions on Crashes")
       ax.set_xlabel("Number of Crashes")
       ax.set_ylabel("Weather Condition")
       st.pyplot(fig)

    if st.checkbox("Show Hourly Crash Frequency"):
       fig, ax = plt.subplots(figsize=(10, 8))
       heatmap_data = df_used.pivot_table(index='CRASH_DAY_OF_WEEK', columns='CRASH_HOUR', aggfunc='size', fill_value=0)
       sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt="d", ax=ax)
       ax.set_title("Hourly Crash Frequency by Day of the Week")
       ax.set_xlabel("Hour of the Day")
       ax.set_ylabel("Day of the Week")
       st.pyplot(fig)



    if st.checkbox("Show Lighting Conditions Impact"):
       fig, ax = plt.subplots(figsize=(8, 5))
       lighting_counts = df_used['LIGHTING_CONDITION'].value_counts()
       sns.barplot(y=lighting_counts.index, x=lighting_counts.values, palette="rocket", ax=ax)
       ax.set_title("Impact of Lighting Conditions on Crashes")
       ax.set_xlabel("Number of Crashes")
       ax.set_ylabel("Lighting Condition")
       st.pyplot(fig)


    if st.checkbox("Show Geospatial Distribution of Crashes"):
       st.map(df_used[['LATITUDE', 'LONGITUDE']].dropna())



# Prediction button
if st.button('Make Prediction'):
    if uploaded_file is not None:
        try:
            # Get the selected model pipeline
            model_pipeline = best_models[model_choice]
            
            # Ensure required preprocessing is applied (e.g., drop unnecessary columns)
            # required_columns = model_pipeline.named_steps['preprocessor'].get_feature_names_out()
            # processed_df = df[required_columns]
            
            # Make predictions
            prediction = model_pipeline.predict(df)

            prediction_labels = ["NO INJURY / DRIVE AWAY" if p == 0 else "INJURY AND / OR TOW DUE TO CRASH" for p in prediction]

            
            # # Show the prediction results
            # st.write("### Prediction Results")
            # st.write(prediction_labels)

            actual_results = df_used['CRASH_TYPE'].map({
                    0: "NO INJURY / DRIVE AWAY",
                    1: "INJURY AND / OR TOW DUE TO CRASH"
                })

            results_df = pd.DataFrame({
                # "Index": range(len(prediction)),
                "Actual Result": actual_results,
                "Prediction": prediction_labels
            })

            # Show the prediction results as a table
            st.write("### Prediction Results")
            st.dataframe(results_df)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Please upload a CSV file before making predictions.")
