from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
from datetime import datetime
import uvicorn
import sys
import os
import numpy as np
import joblib
import pandas as pd
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.feature_engineering import (
    FeatureEngineering,
)  # Import feature engineering class
from scripts.credit_scoring_model import CreditScoreRFM  # Import RFM class

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Load the model using joblib (instead of pickle)
try:
    with open("api/model/best_model.pkl", "rb") as f:
        model = joblib.load(f)
    logging.info("Model loaded successfully.")
    print(f"Loaded model type: {type(model)}")

except FileNotFoundError:
    logging.error(
        "Model file not found. Ensure the model exists in the 'model' directory."
    )
    raise


# Create FastAPI app
app = FastAPI()


# Input schema
class InputData(BaseModel):
    TransactionId: int
    CustomerId: int
    ProductCategory: int
    ChannelId: str
    Amount: float
    TransactionStartTime: datetime
    PricingStrategy: int


@app.get("/")
async def read_root():
    return {
        "message": "🎉 Welcome to Bati Bank Credit Scoring API! 🎉",
        "description": "This API allows you to make predictions about credit scoring.📊",
        "author": "©️ Amen Zelealem",
        "instructions": (
            "To explore the API and test its endpoints, please visit "
            "the Swagger documentation at: /docs 📖"
        ),
        "note": "Make sure to use the base URL followed by /docs. 🔗"
    }

@app.post("/predict")
async def predict(input_data: InputData):
    try:
        # Log received data
        logging.info(f"Received input data: {input_data}")

        # Prepare input data as a DataFrame
        input_data_dict = {
            "TransactionId": input_data.TransactionId,
            "CustomerId": input_data.CustomerId,
            "ProductCategory": input_data.ProductCategory,
            "ChannelId": input_data.ChannelId,
            "Amount": input_data.Amount,
            "TransactionStartTime": input_data.TransactionStartTime,
            "PricingStrategy": input_data.PricingStrategy,
        }
        input_df = pd.DataFrame([input_data_dict])

        # Log preprocessing start
        logging.info("Starting feature engineering...")

        # Feature Engineering
        fe = FeatureEngineering()
        input_df = fe.create_aggregate_features(input_df)
        input_df = fe.create_transaction_features(input_df)
        input_df = fe.extract_time_features(input_df)

        # Encode categorical features
        categorical_cols = ["ProductCategory", "ChannelId"]
        input_df = fe.encode_categorical_features(input_df, categorical_cols)

        # Normalize numerical features
        numeric_cols = input_df.select_dtypes(include="number").columns.tolist()
        exclude_cols = ["Amount", "TransactionId"]
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        input_df = fe.normalize_numerical_features(
            input_df, numeric_cols, method="standardize"
        )

        # RFM Calculation
        rfm = CreditScoreRFM(input_df.reset_index())
        rfm_df = rfm.calculate_rfm()

        # Merge RFM features with the input data
        final_df = pd.merge(input_df, rfm_df, on="CustomerId", how="left")

        # Ensure all required features exist in the final_df
        required_features = [
            "ProductCategory",
            "PricingStrategy",
            "Transaction_Count",
            "Transaction_Month",
            "Transaction_Year",
            "Recency",
            "Frequency",
        ]

        # Fill missing features with default values (e.g., 0 for numerical features)
        for feature in required_features:
            if feature not in final_df.columns:
                final_df[feature] = 0  # Fill with default value

        # Reindex the final_df to match the training feature order
        final_df = final_df.reindex(columns=required_features, fill_value=0)

        # Log prediction start
        logging.info("Making prediction...")

        # Make prediction
        prediction = model.predict(final_df)
        predicted_risk = "Good" if prediction[0] == 0 else "Bad"

        # Log prediction result
        logging.info(
            f"Prediction complete: Customer ID {input_data.CustomerId}, Predicted Risk: {predicted_risk}"
        )

        # Return response
        return {"customer_id": input_data.CustomerId, "predicted_risk": predicted_risk}

    except ValidationError as ve:
        logging.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Validation error: {ve}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(e)}"
        )
