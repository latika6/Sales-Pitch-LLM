import openai
import requests
from bs4 import BeautifulSoup
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import language_tool_python
from textstat import flesch_reading_ease
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Set your OpenAI API key
openai.api_key = ""

# Function to scrape data from HiDevs website
def scrape_hidevs_website():
    try:
        url = ""
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract relevant data
            events = [event.text.strip() for event in soup.find_all("h3", class_="event-title")]
            descriptions = [desc.text.strip() for desc in soup.find_all("p", class_="event-description")]
            
            data = {
                "events": events,
                "descriptions": descriptions
            }
            return data
        else:
            return {"error": f"Failed to fetch data. HTTP Status Code: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Function to generate sales pitch using GPT
def generate_sales_pitch(target_audience, event_type, key_points, website_data):
    try:
        events = ", ".join(website_data.get("events", []))
        descriptions = " ".join(website_data.get("descriptions", []))
        
        prompt = f"""
        Create a professional and persuasive sales pitch for a {event_type}. 
        The pitch is targeted at {target_audience}. 
        Focus on the following key points: {key_points}.
        
        Reference the following information about HiDevs:
        Events: {events}.
        Descriptions: {descriptions}.
        
        Ensure the tone is professional, engaging, and tailored to the audience.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7
        )
        
        pitch = response['choices'][0]['message']['content']
        return pitch.strip()
    except Exception as e:
        return f"Error generating sales pitch: {e}"

# Performance Metrics
def evaluate_pitch(pitch, target_audience, key_points):
    metrics = {}

    # Relevance Score (Cosine Similarity)
    vectorizer = CountVectorizer().fit_transform([pitch, target_audience + " " + key_points])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0][1]
    metrics["Relevance Score"] = round(cosine_sim * 100, 2)

    # Sentiment Analysis
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(pitch)
    metrics["Sentiment (Positive)"] = round(sentiment["pos"] * 100, 2)
    metrics["Sentiment (Negative)"] = round(sentiment["neg"] * 100, 2)

    # Readability Score
    readability = flesch_reading_ease(pitch)
    metrics["Readability Score"] = round(readability, 2)

    # Grammar Check
    tool = language_tool_python.LanguageTool("en-US")
    matches = tool.check(pitch)
    metrics["Grammar Issues"] = len(matches)

    # Length Check
    word_count = len(pitch.split())
    metrics["Word Count"] = word_count

    return metrics

# Streamlit app UI
def main():
    st.title("AI-Powered Sales Pitch Generator with Performance Metrics")
    st.write("Generate customized sales pitches using website content and evaluate their quality.")

    # Scrape data from HiDevs website
    st.subheader("Website Data")
    website_data = scrape_hidevs_website()
    
    if "error" in website_data:
        st.error(website_data["error"])
    else:
        st.write("**Events Extracted:**")
        st.write(website_data["events"])
        st.write("**Descriptions Extracted:**")
        st.write(website_data["descriptions"])
    
    # Input fields
    target_audience = st.text_input("Target Audience", placeholder="e.g., Data Science Community, Colleges, Companies")
    event_type = st.text_input("Event Type", placeholder="e.g., Workshop, Webinar, Seminar")
    key_points = st.text_area("Key Points", placeholder="e.g., Advanced Machine Learning, Career Growth, Hands-on Learning")

    # Generate pitch button
    if st.button("Generate Sales Pitch"):
        if target_audience and event_type and key_points and website_data:
            with st.spinner("Generating your sales pitch..."):
                pitch = generate_sales_pitch(target_audience, event_type, key_points, website_data)
                st.subheader("Generated Sales Pitch:")
                st.success(pitch)

                # Evaluate the pitch
                st.subheader("Performance Metrics")
                metrics = evaluate_pitch(pitch, target_audience, key_points)
                for metric, value in metrics.items():
                    st.write(f"**{metric}:** {value}")
        else:
            st.error("Please fill in all the input fields or ensure website data is fetched correctly!")

# Run the app
if __name__ == '__main__':
    main()
