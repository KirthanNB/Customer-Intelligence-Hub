import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()

def generate_executive_summary(dataframe, topic_context="selected"):

    if dataframe.empty:
        return "No data available to summarize."
        
                                                                                                   
    sample_df = dataframe[['AMOUNT SPENT ON THAT DAY(20/10/2024)', 'Payment Method', 'Favorite Dish', 'Visit Frequency']].head(15)
    
                                                             
    data_string = sample_df.to_string(index=False)
    
                                           
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


def generate_targeted_campaign(favorite_dish, payment_method, amount_spent, feedback_score):
    """Uses Gemini to generate a personalized SMS based on whether the customer was happy or angry."""
    
    # AI Logic: Change the instructions based on their past experience
    if feedback_score <= 2:
        strategy = "The customer had a BAD experience last time. Apologize sincerely for falling short and offer a massive discount on their favorite dish to make it right."
    else:
        strategy = "The customer had a GREAT experience last time but hasn't visited recently. Tell them we miss them and offer a friendly discount on their favorite dish."

    prompt = f"""
    You are an expert restaurant marketing manager. Write a short SMS text message (max 2 sentences).
    
    Customer Profile:
    - Favorite dish: {favorite_dish}
    - Last feedback score: {feedback_score} out of 5
    
    Your Goal: {strategy}
    Do not use placeholders like [Name]. Be human, empathetic, and persuasive.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Error generating campaign: {e}"