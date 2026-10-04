import pandas as pd

def get_sentiment_from_score(score):
    """Categorize 1-5 scores into clean labels."""
    if score >= 4:
        return "Positive"
    elif score <= 2:
        return "Negative"
    else:
        return "Neutral"

def process_reviews(file_path):
    """Loads data and maps numerical scores to sentiment columns."""
    df = pd.read_csv(file_path)
    
    # Apply human-readable labels based on the numeric score
    df['sentiment'] = df['Feedback Score (1-5)'].apply(get_sentiment_from_score)
    
    return df