import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def train_risk_model(file_path):

    df = pd.read_csv(file_path)
    
                                                           
    df['At_Risk'] = ((df['Feedback Score (1-5)'] <= 2) | (df['Visit Frequency'] == 'Low')).astype(int)
    
                        
    X = df[['AMOUNT SPENT ON THAT DAY(20/10/2024)', 'Payment Method', 'Favorite Dish']].copy()
    y = df['At_Risk']
    
                                                                             
    X_encoded = pd.get_dummies(X)
    
                        
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)
    
                                                                                           
    return model, X_encoded.columns

def predict_customer_risk(model, model_columns, amount, payment, dish):

                                             
    input_df = pd.DataFrame({
        'AMOUNT SPENT ON THAT DAY(20/10/2024)': [amount],
        'Payment Method': [payment],
        'Favorite Dish': [dish]
    })
    
                                                        
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    
                              
    prediction = model.predict(input_encoded)[0]
    return "High Risk of Churn" if prediction == 1 else "Loyal Customer"