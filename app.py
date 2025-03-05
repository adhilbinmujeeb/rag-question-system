import streamlit as st
import google.generativeai as genai
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import json

# MongoDB Connection
MONGO_URI = "mongodb+srv://adhilbinmujeeb:admin123@cluster0.uz62z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["business_rag"]
collection = db["questions"]

# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Set Gemini API Key
GEMINI_API_KEY = "AIzaSyB5zaK5_IqRH1K5Rk9ibwSGP-nk1icWUIo"
genai.configure(api_key=GEMINI_API_KEY)

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Function to retrieve similar questions
def search_questions(query, top_n=10):
    query_embedding = embedding_model.encode(query).tolist()
    all_questions = list(collection.find({}, {"question": 1, "category": 1, "subcategory": 1, "embedding": 1}))
    
    # Compute similarity scores
    for q in all_questions:
        q["similarity"] = cosine_similarity(query_embedding, q["embedding"])
    
    # Sort by similarity and return top_n results
    sorted_questions = sorted(all_questions, key=lambda x: x["similarity"], reverse=True)[:top_n]
    return [q["question"] for q in sorted_questions]

# Function to generate the first batch of questions with Gemini
def generate_initial_questions(business_type, num_questions=5):
    prompt = f"""
    You are an expert business analyst. I'm analyzing a {business_type} business.
    Generate {num_questions} important questions to understand this business better.
    The questions should cover different aspects like business model, target market, competition, revenue streams,
    challenges, and growth potential.
    
    Format your response as a JSON array of question objects like this:
    [
        {{"id": 1, "question": "What is your primary revenue stream?", "category": "Finance"}},
        {{"id": 2, "question": "Who is your target audience?", "category": "Market"}}
    ]
    
    Make the questions specific, insightful and relevant to the {business_type} business type.
    """
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    try:
        # Parse the JSON response
        questions = json.loads(response.text)
        return questions
    except:
        # Fallback if JSON parsing fails
        fallback_questions = [
            {"id": 1, "question": f"What specific products or services does your {business_type} offer?", "category": "Business Model"},
            {"id": 2, "question": "Who is your target customer?", "category": "Market"},
            {"id": 3, "question": "What are your primary revenue streams?", "category": "Finance"},
            {"id": 4, "question": "Who are your main competitors?", "category": "Competition"},
            {"id": 5, "question": "What are your biggest challenges right now?", "category": "Challenges"}
        ]
        return fallback_questions

# Function to generate follow-up questions based on previous answers
def generate_follow_up_question(business_type, previous_qa, remaining_questions=3):
    # Format the previous Q&A for context
    qa_context = ""
    for qa in previous_qa:
        qa_context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
    
    prompt = f"""
    You are an expert business analyst helping me understand a {business_type} business.
    Based on the following information I've gathered so far:

    {qa_context}

    Generate {remaining_questions} follow-up questions that would help me better understand this business.
    The questions should dig deeper into areas mentioned in previous answers or explore important aspects not yet covered.
    
    Format your response as a JSON array of question objects like this:
    [
        {{"id": 1, "question": "Based on your target audience, what marketing channels have been most effective?", "category": "Marketing"}},
        {{"id": 2, "question": "You mentioned challenges with X. How are you addressing these challenges?", "category": "Strategy"}}
    ]
    
    Make the questions specific, insightful and relevant based on the previous answers.
    """
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    try:
        # Parse the JSON response
        questions = json.loads(response.text)
        return questions
    except:
        # Fallback if JSON parsing fails
        fallback_questions = [
            {"id": 1, "question": "Can you elaborate more on your business model?", "category": "Business Model"},
            {"id": 2, "question": "What are your plans for scaling the business?", "category": "Growth"},
            {"id": 3, "question": "What metrics do you use to measure success?", "category": "Performance"}
        ]
        return fallback_questions

# Function to generate business report
def generate_business_report(business_type, qa_pairs):
    # Format the Q&A for context
    qa_context = ""
    for qa in qa_pairs:
        qa_context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"
    
    prompt = f"""
    You are an expert business analyst. Based on the following information about a {business_type} business:

    {qa_context}

    Create a comprehensive business report that would be useful for investors and founders.
    
    The report should include:
    1. Executive Summary
    2. Business Model Analysis
    3. Market Analysis and Target Customer
    4. Competitive Landscape
    5. Financial Overview and Revenue Streams
    6. Challenges and Risks
    7. Growth Opportunities
    8. Recommendations
    
    Format the report in markdown with clear headings and bullet points where appropriate.
    Be specific, data-driven where possible, and provide actionable insights.
    """
    
    model = genai.GenerativeModel("gemini-2.0-pro")
    response = model.generate_content(prompt)
    
    return response.text

# Initialize session state variables
if 'started' not in st.session_state:
    st.session_state.started = False

if 'current_question_index' not in st.session_state:
    st.session_state.current_question_index = 0

if 'questions' not in st.session_state:
    st.session_state.questions = []

if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = []

if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

if 'business_type' not in st.session_state:
    st.session_state.business_type = ""

# Function to start the questionnaire
def start_questionnaire():
    st.session_state.started = True
    st.session_state.current_question_index = 0
    st.session_state.questions = generate_initial_questions(st.session_state.business_type)
    st.session_state.qa_pairs = []
    st.session_state.report_generated = False

# Function to handle next question
def next_question():
    # Save current answer
    current_q = st.session_state.questions[st.session_state.current_question_index]
    answer = st.session_state.current_answer
    
    # Add to QA pairs
    st.session_state.qa_pairs.append({
        "question": current_q["question"],
        "answer": answer,
        "category": current_q.get("category", "General")
    })
    
    # Move to next question or generate more if needed
    st.session_state.current_question_index += 1
    
    # If we've reached the end of current questions, generate more
    if st.session_state.current_question_index >= len(st.session_state.questions):
        # After every 5 questions, generate new follow-up questions
        new_questions = generate_follow_up_question(
            st.session_state.business_type, 
            st.session_state.qa_pairs,
            remaining_questions=3
        )
        
        # Add IDs to the new questions to continue the sequence
        for i, q in enumerate(new_questions):
            q["id"] = len(st.session_state.questions) + i + 1
            
        st.session_state.questions.extend(new_questions)

# Function to generate final report
def generate_report():
    st.session_state.report_generated = True

# Streamlit UI
st.title("Interactive Business Analysis Tool")

if not st.session_state.started:
    st.subheader("Tell us about your business")
    business_type = st.text_input("What type of business are you analyzing?", key="business_type_input")
    
    if st.button("Start Analysis"):
        if business_type:
            st.session_state.business_type = business_type
            start_questionnaire()
        else:
            st.warning("Please enter a business type!")

elif not st.session_state.report_generated:
    # Display progress
    total_questions_expected = 10  # We'll ask at least this many questions
    current_progress = min(100, int((st.session_state.current_question_index / total_questions_expected) * 100))
    st.progress(current_progress / 100)
    st.write(f"Question {st.session_state.current_question_index + 1}")

    # Display current question
    current_q = st.session_state.questions[st.session_state.current_question_index]
    st.subheader(current_q["question"])
    st.text_area("Your answer:", key="current_answer")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("Next Question"):
            next_question()
    
    # Allow generating report after answering at least 5 questions
    if len(st.session_state.qa_pairs) >= 5:
        with col2:
            if st.button("Generate Business Report"):
                generate_report()
    
    # Display answered questions (collapsible)
    if st.session_state.qa_pairs:
        with st.expander("Previous Answers"):
            for qa in st.session_state.qa_pairs:
                st.markdown(f"**Q: {qa['question']}** ({qa['category']})")
                st.markdown(f"A: {qa['answer']}")
                st.divider()

else:
    # Generate and display report
    st.subheader("Business Analysis Report")
    report = generate_business_report(st.session_state.business_type, st.session_state.qa_pairs)
    st.markdown(report)
    
    # Add option to download report
    st.download_button(
        label="Download Report",
        data=report,
        file_name=f"{st.session_state.business_type.replace(' ', '_')}_business_analysis.md",
        mime="text/markdown"
    )
    
    # Option to restart
    if st.button("Start New Analysis"):
        st.session_state.started = False
        st.session_state.current_question_index = 0
        st.session_state.questions = []
        st.session_state.qa_pairs = []
        st.session_state.report_generated = False
        st.session_state.business_type = ""
        st.experimental_rerun()
