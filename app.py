import streamlit as st
import google.generativeai as genai
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# MongoDB Connection
MONGO_URI = "mongodb+srv://your_user:your_password@your-cluster.mongodb.net/business_rag"
client = MongoClient(MONGO_URI)
db = client["business_rag"]
collection = db["questions"]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set Gemini API Key
GEMINI_API_KEY = "your_gemini_api_key_here"
genai.configure(api_key=GEMINI_API_KEY)

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to retrieve similar questions
def search_questions(query, top_n=5):
    query_embedding = embedding_model.encode(query).tolist()
    all_questions = list(collection.find({}, {"question": 1, "category": 1, "subcategory": 1, "embedding": 1}))

    # Compute similarity scores
    for q in all_questions:
        q["similarity"] = cosine_similarity(query_embedding, q["embedding"])

    # Sort by similarity and return top_n results
    sorted_questions = sorted(all_questions, key=lambda x: x["similarity"], reverse=True)[:top_n]

    return [q["question"] for q in sorted_questions]

# Function to refine questions with Gemini
def refine_questions_with_gemini(questions, user_query):
    prompt = f"""
    You are an expert in business analysis. Based on the user's input: "{user_query}",
    refine the following questions to make them more precise and context-aware:

    {questions}

    Provide a structured and relevant set of questions.
    """

    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)

    return response.text if response else "Error retrieving response from Gemini API"

# Streamlit UI
st.title("RAG-Powered Business Question Generator (Gemini)")

# User input
user_query = st.text_input("Enter business type or key details:", "")

if st.button("Generate Questions"):
    if user_query:
        retrieved_questions = search_questions(user_query)

        if retrieved_questions:
            st.subheader("Refining Questions with Gemini AI...")
            refined_questions = refine_questions_with_gemini(retrieved_questions, user_query)

            st.subheader("AI-Generated Questions:")
            st.write(refined_questions)
        else:
            st.warning("No relevant questions found!")
    else:
        st.warning("Please enter a query!")
