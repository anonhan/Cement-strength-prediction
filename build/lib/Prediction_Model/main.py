import streamlit as st
import pandas as pd
from io import BytesIO
from Prediction_Model.data_validation_insertion.train_validate_insert import Train_Validation
from Prediction_Model.data_validation_insertion.predict_validate_insert import Prediction_Validation
from Prediction_Model.train_model.train import TrainModel
from Prediction_Model.predict.predict import MakePredictions
from Prediction_Model.config.config import TRAINING_FILES_DIR, PREDICTION_FILES_DIR, PREDICTION_OUTPUT_FILE

# Main function
def main():
    st.title("Cement Strength Prediction App")
    st.write("Select a tab to perform an action.")

    st.sidebar.title("Navigation")
    selected_tab = st.sidebar.radio("Select Tab", ["Train Model", "Predict Cement Strength", "Create Bulk Predictions"])

    if selected_tab == "Train Model":
        st.header("Train Model")
        st.write("Click below to start model training.")

        if st.button("Train Model"):
            try:
                st.info("Starting the model training. Please wait...")
                Train_Validation(path=TRAINING_FILES_DIR).validate_training_data()
                TrainModel().start_training()
                st.success("Model training completed successfully!")
            except Exception as e:
                st.error("Error occurred during model training: " + str(e))

    elif selected_tab == "Predict Cement Strength":
        st.header("Predict Cement Strength")
        st.write("Get the predictions based on")

        if st.button("Predict Cement Strength"):
            try:
                st.info("Starting the model prediction. Please wait...")
                Prediction_Validation(path=PREDICTION_FILES_DIR).validate_prediction_data()
                predictor = MakePredictions()
                is_successful = predictor.start_prediction()
                if is_successful:
                    st.success("Predictions made successfully!")
                    st.info("Displaying predictions...")
                    preds = pd.read_csv(PREDICTION_OUTPUT_FILE)
                    st.write(preds.head())
                else:
                    st.error("Error occurred while making predictions.")
            except Exception as e:
                st.error("Error occurred during prediction generation: " + str(e))
        
    elif selected_tab == "Create Bulk Predictions":
        st.header("Upload CSV File")
        st.write("Upload a CSV file below to get predictions.")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            st.info("Analysing the file please wait...")
            prediction_df = pd.read_csv(uploaded_file)
            try:
                st.info("Validating the uploaded data. Please wait...")
                # Convert DataFrame to BytesIO object
                csv_buffer = BytesIO()
                prediction_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                file_name = uploaded_file.name
                # Pass the BytesIO object to the validation function
                Prediction_Validation(path=csv_buffer).validate_prediction_data(is_file_from_path=False,uploaded_file=csv_buffer, file_name=file_name)
                st.success("Data validation completed successfully!")
                
                predictor = MakePredictions()
                is_successful = predictor.start_prediction()
                if is_successful:
                    st.success("Predictions made successfully!")
                    st.info("Displaying predictions...")
                    preds = pd.read_csv(PREDICTION_OUTPUT_FILE)
                    st.write(preds.head())
                else:
                    st.error("Error occurred while making predictions.")
            except Exception as e:
                st.error("Error occurred during data validation: " + str(e))

if __name__ == "__main__":
    main()
