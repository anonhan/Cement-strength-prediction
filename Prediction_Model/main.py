import streamlit as st
import pandas as pd
from Prediction_Model.data_validation_insertion.train_validate_insert import Train_Validation
from Prediction_Model.data_validation_insertion.predict_validate_insert import Prediction_Validation
from Prediction_Model.train_model.train import TrainModel
from Prediction_Model.config.config import TRAINING_FILES_DIR, PREDICTION_FILES_DIR

# Function to load and display uploaded CSV file
def load_data(file):
    data = pd.read_csv(file)
    return data

# Main function
def main():
    # Page title and description
    st.title("Cement Strength Prediction App")
    st.write("Select a tab to perform an action.")

    # Sidebar for tab selection
    st.sidebar.title("Navigation")
    selected_tab = st.sidebar.radio("Select Tab", ["Train Model", "Predict Cement Strength"])
    if selected_tab == "Train Model":
        # Train Model Tab
        st.header("Train Model")
        st.write("Click below to start model training.")

        # Button to trigger model training
        if st.button("Train Model"):
            try:
                st.info("Starting the model training. Pleas wait...")
                train_data_val = Train_Validation(path=TRAINING_FILES_DIR)
                train_data_val.validate_training_data()

                # st.info("Model training process has started. Please wait...")
                # train_model = TrainModel()
                # train_model.start_training()

                st.success("Model training completed successfully!")
            
            except Exception as e:
                st.error("The model has been trained already!!")
                raise Exception("Error occurred during model training.")

    elif selected_tab == "Predict Cement Strength":
        # Predict Cement Strength Tab
        st.header("Predict Cement Strength")
        st.write("Upload a CSV file to get predictions.")
        if st.button("Pred Model"):
            try:
                st.info("Starting the model prediction validation. Pleas wait...")
                train_data_val = Prediction_Validation(path=PREDICTION_FILES_DIR)
                train_data_val.validate_prediction_data()

                # st.info("Model training process has started. Please wait...")
                # train_model = TrainModel()
                # train_model.start_training()

                st.success("Model training completed successfully!")
            
            except Exception as e:
                st.error("The model has been trained already!!")
                raise Exception("Error occurred during model training.")

        # # File upload for prediction data
        # file_predict = st.file_uploader("Upload Data for Prediction (CSV)", type=['csv'])

        # if file_predict is not None:
        #     # Load and display the first few rows of the data
        #     data = load_data(file_predict)
        #     st.write("Preview of uploaded data:")
        #     st.write(data.head())

        #     # Button to trigger predictions
        #     if st.button("Make Predictions"):
        #         # Placeholder for prediction function
        #         st.write("Predictions: [Placeholder]")

# Run the app
if __name__ == "__main__":
    main()