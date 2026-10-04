import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

def generate_executive_summary(dataframe, topic_context="selected"):
    """
    Takes a Pandas dataframe of customer profiles and uses Gemini to summarize the core trends.
    """
    if dataframe.empty:
        return "No data available to summarize."
        
    # Isolate the most relevant columns and take a sample (e.g., top 15) to keep the prompt concise
    sample_df = dataframe[['AMOUNT SPENT ON THAT DAY(20/10/2024)', 'Payment Method', 'Favorite Dish', 'Visit Frequency']].head(15)
    
    # Convert the dataframe sample to a readable string table
    data_string = sample_df.to_string(index=False)
    
    # Construct the prompt for tabular data
    prompt = f"""
    You are an expert restaurant consultant. Analyze the following sample of {topic_context} customer profiles.
    Provide a concise, 3-bullet-point executive summary highlighting trends in their favorite dishes, spending habits, and payment methods.
    Keep it professional and actionable.
    
    Customer Data:
    {data_string}
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating summary: {e}"


def generate_targeted_campaign(favorite_dish, payment_method, amount_spent):
    """Uses Gemini to generate a personalized SMS marketing campaign to win back a customer."""
    
    prompt = f"""
    You are an AI marketing specialist for a restaurant. 
    Write a short, engaging SMS text message (max 2 sentences) to win back a customer who hasn't visited recently.
    
    Customer Profile:
    - Their favorite dish is: {favorite_dish}
    - They usually pay with: {payment_method}
    - Their last bill was: ₹{amount_spent}
    
    Offer them a compelling discount on their favorite dish. Do not use placeholders like [Name], just start the message directly.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating campaign: {e}"