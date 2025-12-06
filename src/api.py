"""
FastAPI REST API for Real-time Fraud Detection
Serves fraud risk scores for UPI transactions
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="UPI Fraud Detection API",
    description="Real-time fraud risk scoring for UPI transactions",
    version="1.0.0"
)

# Load model and components
try:
    model = joblib.load('../models/fraud_model.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    engineer = joblib.load('../models/feature_engineer.joblib')
    config = joblib.load('../models/config.joblib')
    THRESHOLD = config['threshold']
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    MODEL_LOADED = False
    THRESHOLD = 0.5

# Request/Response Models
class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    sender_upi: str
    receiver_upi: str
    amount: float
    transaction_type: str = "P2M"
    merchant_category: str = "grocery"
    timestamp: Optional[str] = None
    device_type: str = "android"
    device_model: str = "Samsung Galaxy"
    device_id: str = "unknown"
    app_version: str = "1.0.0"
    location_city: str = "Mumbai"

class FraudResponse(BaseModel):
    transaction_id: str
    risk_score: float
    is_fraud: bool
    risk_level: str
    recommendation: str
    features_flagged: list

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold: float

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health status"""
    return HealthResponse(
        status="healthy",
        model_loaded=MODEL_LOADED,
        threshold=THRESHOLD
    )

@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(transaction: TransactionRequest):
    """
    Predict fraud probability for a single transaction
    Returns risk score and recommendation
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Parse timestamp
        if transaction.timestamp:
            ts = pd.to_datetime(transaction.timestamp)
        else:
            ts = datetime.now()

        # Create transaction dataframe
        txn_data = {
            'transaction_id': [transaction.transaction_id],
            'user_id': [transaction.user_id],
            'sender_upi': [transaction.sender_upi],
            'receiver_upi': [transaction.receiver_upi],
            'amount': [transaction.amount],
            'transaction_type': [transaction.transaction_type],
            'merchant_category': [transaction.merchant_category],
            'timestamp': [ts],
            'hour': [ts.hour],
            'day_of_week': [ts.weekday()],
            'is_weekend': [ts.weekday() >= 5],
            'device_type': [transaction.device_type],
            'device_model': [transaction.device_model],
            'device_id': [transaction.device_id],
            'app_version': [transaction.app_version],
            'location_city': [transaction.location_city],
            'is_fraud': [0]  # Placeholder for feature engineering
        }

        df = pd.DataFrame(txn_data)

        # Apply feature engineering
        df = engineer.transform(df)

        # Get feature columns
        feature_cols = engineer.get_feature_columns()
        X = df[feature_cols].fillna(0)

        # Predict
        risk_score = float(model.predict_proba(X)[0, 1])
        is_fraud = risk_score >= THRESHOLD

        # Determine risk level
        if risk_score < 0.3:
            risk_level = "LOW"
            recommendation = "APPROVE - Transaction appears normal"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
            recommendation = "REVIEW - Manual verification recommended"
        elif risk_score < 0.8:
            risk_level = "HIGH"
            recommendation = "FLAG - Requires immediate review"
        else:
            risk_level = "CRITICAL"
            recommendation = "BLOCK - High probability of fraud"

        # Identify flagged features
        features_flagged = []
        if df['is_night'].values[0] == 1:
            features_flagged.append("Unusual transaction time (night)")
        if df['is_unusual_amount'].values[0] == 1:
            features_flagged.append("Amount significantly above user average")
        if df['is_new_device'].values[0] == 1:
            features_flagged.append("New device detected")
        if df['is_different_city'].values[0] == 1:
            features_flagged.append("Transaction from different city")
        if df['txn_count_1h'].values[0] > 5:
            features_flagged.append("High transaction velocity")

        return FraudResponse(
            transaction_id=transaction.transaction_id,
            risk_score=round(risk_score, 4),
            is_fraud=is_fraud,
            risk_level=risk_level,
            recommendation=recommendation,
            features_flagged=features_flagged
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(transactions: list[TransactionRequest]):
    """
    Predict fraud for multiple transactions
    """
    results = []
    for txn in transactions:
        result = await predict_fraud(txn)
        results.append(result)
    return results

# Example usage endpoint
@app.get("/example")
async def example_request():
    """Returns an example request for testing"""
    return {
        "transaction_id": "TXN123456789",
        "user_id": "USER_00001",
        "sender_upi": "rahul123@paytm",
        "receiver_upi": "bigbazaar.merchant@sbi",
        "amount": 2500.00,
        "transaction_type": "P2M",
        "merchant_category": "grocery",
        "timestamp": "2024-06-15T14:30:00",
        "device_type": "android",
        "device_model": "Samsung Galaxy",
        "device_id": "abc123def456",
        "app_version": "2.1.0",
        "location_city": "Mumbai"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
