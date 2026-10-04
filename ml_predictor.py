import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_risk_model(file_path):
    """Trains a Random Forest model to predict if a customer is a churn risk."""
    df = pd.read_csv(file_path)
    
    # 1. Define the Target Variable (1 = At Risk, 0 = Safe)
    df['At_Risk'] = ((df['Feedback Score (1-5)'] <= 2) | (df['Visit Frequency'] == 'Low')).astype(int)
    
    # 2. Select Features
    X = df[['AMOUNT SPENT ON THAT DAY(20/10/2024)', 'Payment Method', 'Favorite Dish']].copy()
    y = df['At_Risk']
    
    # 3. One-Hot Encode categorical variables (Payment Method, Favorite Dish)
    X_encoded = pd.get_dummies(X)
    
    # 4. Train the Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)
    
    # Return the trained model and the exact column structure needed for future predictions
    return model, X_encoded.columns

def predict_customer_risk(model, model_columns, amount, payment, dish):
    """Takes a single customer profile and predicts their risk status."""
    # Create a dataframe for the single input
    input_df = pd.DataFrame({
        'AMOUNT SPENT ON THAT DAY(20/10/2024)': [amount],
        'Payment Method': [payment],
        'Favorite Dish': [dish]
    })
    
    # Encode the input to match the training data format
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    # Predict (returns 1 or 0)
    prediction = model.predict(input_encoded)[0]
    return "High Risk of Churn" if prediction == 1 else "Loyal Customer"