import streamlit as st
import pandas as pd
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
    selected_tab = st.sidebar.radio("Select Tab", ["Train Model", "Predict Cement Strength"])

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
        st.write("Upload a CSV file to get predictions.")

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

if __name__ == "__main__":
    main()
