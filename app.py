import streamlit as st
import google.generativeai as genai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import json
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import os

# MongoDB Connection
MONGO_URI = "mongodb+srv://adhilbinmujeeb:admin123@cluster0.uz62z.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(MONGO_URI)
db = client["business_rag"]
business_data_collection = db["business_attributes"]
question_collection = db["questions"]

# Embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini API Key
GEMINI_API_KEY = "AIzaSyB5zaK5_IqRH1K5Rk9ibwSGP-nk1icWUIo"
genai.configure(api_key=GEMINI_API_KEY)

# --- Data Loading Functions (MongoDB Only) ---

@st.cache_data
def load_business_data_from_mongodb():
    """
    Loads business data from MongoDB into a Pandas DataFrame.
    """
    try:
        data = list(business_data_collection.find()) # Fetch all documents
        if not data:
            st.warning("No business data found in MongoDB. Please run the data insertion script (upload_data_mongodb.py).")
            return None
        df = pd.DataFrame(data)
        if '_id' in df.columns: # Remove MongoDB's ObjectId if present
            df = df.drop(columns=['_id'])
        return df
    except Exception as e:
        st.error(f"Error loading business data from MongoDB: {e}")
        return None

# --- Helper Functions (Cosine Similarity, Search Questions, etc.) ---
# (These remain the same as in your previous complete code)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_questions(query, top_n=10):
    query_embedding = embedding_model.encode(query).tolist()
    all_questions = list(question_collection.find({}, {"question": 1, "category": 1, "subcategory": 1, "embedding": 1}))

    # Compute similarity scores
    for q in all_questions:
        q["similarity"] = cosine_similarity(query_embedding, q["embedding"])

    # Sort by similarity and return top_n results
    sorted_questions = sorted(all_questions, key=lambda x: x["similarity"], reverse=True)[:top_n]
    return [q["question"] for q in sorted_questions]

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

def recommend_similar_businesses(business_attributes, top_n=5):
    df = load_business_data_from_mongodb() # Load from MongoDB
    if df is None:
        return []

    # Extract key attributes to match against
    industry = business_attributes.get("Industry", "")
    revenue_model = business_attributes.get("Revenue Model", "")
    development_stage = business_attributes.get("Development Stage", "")

    # Filter based on attributes (with fallbacks if no exact matches)
    filtered_df = df

    if industry:
        filtered_df = filtered_df[filtered_df["Business Attributes.Business Fundamentals.Industry Classification.Primary Industry"].str.contains(industry, case=False, na=False)]

    # If we have too few results, don't filter further
    if len(filtered_df) < 10 and len(filtered_df) > 0:
        similar_businesses = filtered_df.head(top_n)
    else:
        # Continue filtering if we have enough businesses
        if revenue_model and len(filtered_df) > 10:
            temp_df = filtered_df[filtered_df["Business Attributes.Business Fundamentals.Business Model.Primary Revenue Model"].str.contains(revenue_model, case=False, na=False)]
            if len(temp_df) > 0:
                filtered_df = temp_df

        if development_stage and len(filtered_df) > 10:
            temp_df = filtered_df[filtered_df["Business Attributes.Business Fundamentals.Development Stage"].str.contains(development_stage, case=False, na=False)]
            if len(temp_df) > 0:
                filtered_df = temp_df

        similar_businesses = filtered_df.head(top_n)

    # Return list of similar businesses with key attributes
    result = []
    for _, row in similar_businesses.iterrows():
        result.append({
            "name": row["business_name"],
            "industry": row["Business Attributes.Business Fundamentals.Industry Classification.Primary Industry"],
            "revenue_model": row["Business Attributes.Business Fundamentals.Business Model.Primary Revenue Model"],
            "development_stage": row["Business Attributes.Business Fundamentals.Development Stage"],
            "target_market": row["Business Attributes.Business Fundamentals.Business Model.Target Market Segment"],
            "revenue_bracket": row["Business Attributes.Financial Metrics.Revenue Brackets (Annual)"]
        })

    return result

def extract_business_attributes(qa_pairs, business_type):
    # Format the Q&A for context
    qa_context = ""
    for qa in qa_pairs:
        qa_context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"

    prompt = f"""
    You are an expert business analyst. Based on the following Q&A about a {business_type} business:

    {qa_context}

    Extract and identify the following business attributes:
    1. Industry
    2. Revenue Model
    3. Development Stage
    4. Target Market
    5. Estimated Annual Revenue
    6. Team Size
    7. Competition Level
    8. Growth Rate
    9. Key Challenges
    10. Innovation Level

    Format your response as a JSON object with these attributes as keys.
    If an attribute is not mentioned or unclear from the context, use "Unknown" as the value.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    try:
        # Parse the JSON response
        attributes = json.loads(response.text)
        return attributes
    except:
        # Fallback if JSON parsing fails
        fallback_attributes = {
            "Industry": business_type,
            "Revenue Model": "Unknown",
            "Development Stage": "Unknown",
            "Target Market": "Unknown",
            "Estimated Annual Revenue": "Unknown",
            "Team Size": "Unknown",
            "Competition Level": "Unknown",
            "Growth Rate": "Unknown",
            "Key Challenges": "Unknown",
            "Innovation Level": "Unknown"
        }
        return fallback_attributes

def generate_benchmark_insights(business_attributes):
    df = load_business_data_from_mongodb() # Load from MongoDB
    if df is None:
        return "Unable to generate insights: Dataset not available"

    industry = business_attributes.get("Industry", "")
    revenue_model = business_attributes.get("Revenue Model", "")
    development_stage = business_attributes.get("Development Stage", "")

    # Create context about the industry benchmarks
    industry_context = ""
    if industry:
        industry_df = df[df["Business Attributes.Business Fundamentals.Industry Classification.Primary Industry"].str.contains(industry, case=False, na=False)]

        if len(industry_df) > 0:
            # Revenue distribution
            revenue_counts = industry_df["Business Attributes.Financial Metrics.Revenue Brackets (Annual)"].value_counts()
            industry_context += f"Revenue brackets in this industry (from {len(industry_df)} companies):\n"
            for bracket, count in revenue_counts.items():
                industry_context += f"- {bracket}: {count} companies ({count/len(industry_df)*100:.1f}%)\n"

            # Growth rates
            growth_counts = industry_df["Business Attributes.Growth & Scalability.Growth Rate"].value_counts()
            industry_context += f"\nGrowth rates in this industry:\n"
            for rate, count in growth_counts.items():
                industry_context += f"- {rate}: {count} companies ({count/len(industry_df)*100:.1f}%)\n"

            # Risk profile
            market_risks = industry_df["Business Attributes.Risk Assessment.Market Risks"].value_counts().head(3)
            industry_context += f"\nTop market risks in this industry:\n"
            for risk, count in market_risks.items():
                industry_context += f"- {risk}: {count} companies\n"

            financial_risks = industry_df["Business Attributes.Risk Assessment.Financial Risks"].value_counts().head(3)
            industry_context += f"\nTop financial risks in this industry:\n"
            for risk, count in financial_risks.items():
                industry_context += f"- {risk}: {count} companies\n"

    prompt = f"""
    You are an expert business analyst with access to industry benchmark data. Based on the following:

    Business Details:
    - Industry: {business_attributes.get("Industry", "Unknown")}
    - Revenue Model: {business_attributes.get("Revenue Model", "Unknown")}
    - Development Stage: {business_attributes.get("Development Stage", "Unknown")}
    - Target Market: {business_attributes.get("Target Market", "Unknown")}
    - Estimated Annual Revenue: {business_attributes.get("Estimated Annual Revenue", "Unknown")}
    - Team Size: {business_attributes.get("Team Size", "Unknown")}
    - Competition Level: {business_attributes.get("Competition Level", "Unknown")}
    - Growth Rate: {business_attributes.get("Growth Rate", "Unknown")}
    - Key Challenges: {business_attributes.get("Key Challenges", "Unknown")}
    - Innovation Level: {business_attributes.get("Innovation Level", "Unknown")}

    Industry Benchmarks:
    {industry_context}

    Generate a concise benchmark analysis that:
    1. Compares the business to industry averages
    2. Identifies areas where the business is outperforming or underperforming
    3. Provides 3-5 actionable recommendations based on industry best practices

    Format the response as markdown with clear headings.
    """

    model = genai.GenerativeModel("gemini-2.0-pro")
    response = model.generate_content(prompt)

    return response.text

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

def generate_industry_insights(industry):
    df = load_business_data_from_mongodb() # Load from MongoDB
    if df is None:
        return "Unable to generate industry insights: Dataset not available"

    # Filter by industry
    industry_df = df[df["Business Attributes.Business Fundamentals.Industry Classification.Primary Industry"].str.contains(industry, case=False, na=False)]

    if len(industry_df) < 5:
        return f"Not enough data available for the {industry} industry. Please check industry name or try a broader category."

    # Gather industry statistics
    stats = {
        "business_count": len(industry_df),
        "revenue_models": industry_df["Business Attributes.Business Fundamentals.Business Model.Primary Revenue Model"].value_counts().to_dict(),
        "development_stages": industry_df["Business Attributes.Business Fundamentals.Development Stage"].value_counts().to_dict(),
        "revenue_brackets": industry_df["Business Attributes.Financial Metrics.Revenue Brackets (Annual)"].value_counts().to_dict(),
        "growth_rates": industry_df["Business Attributes.Growth & Scalability.Growth Rate"].value_counts().to_dict(),
        "market_risks": industry_df["Business Attributes.Risk Assessment.Market Risks"].value_counts().head(5).to_dict(),
        "financial_risks": industry_df["Business Attributes.Risk Assessment.Financial Risks"].value_counts().head(5).to_dict(),
        "innovation_levels": industry_df["Business Attributes.Product/Service Attributes.Innovation Level"].value_counts().to_dict(),
    }

    prompt = f"""
    You are an expert business analyst. Based on the following statistics about the {industry} industry:

    - Total businesses analyzed: {stats['business_count']}
    - Top revenue models: {stats['revenue_models']}
    - Development stages: {stats['development_stages']}
    - Revenue brackets: {stats['revenue_brackets']}
    - Growth rates: {stats['growth_rates']}
    - Top market risks: {stats['market_risks']}
    - Top financial risks: {stats['financial_risks']}
    - Innovation levels: {stats['innovation_levels']}

    Generate a concise industry insights summary that would help entrepreneurs and investors understand:
    1. The current state of the {industry} industry
    2. Common business models and their effectiveness
    3. Typical growth trajectories
    4. Key risks and challenges
    5. Success factors based on the data

    Format your response as markdown with clear section headings.
    """

    model = genai.GenerativeModel("gemini-2.0-pro")
    response = model.generate_content(prompt)

    return response.text

def generate_business_report(business_type, qa_pairs):
    # Extract business attributes
    business_attributes = extract_business_attributes(qa_pairs, business_type)

    # Get similar businesses
    similar_businesses = recommend_similar_businesses(business_attributes)

    # Get benchmark insights
    benchmark_insights = generate_benchmark_insights(business_attributes)

    # Format the Q&A for context
    qa_context = ""
    for qa in qa_pairs:
        qa_context += f"Q: {qa['question']}\nA: {qa['answer']}\n\n"

    # Format similar businesses for context
    similar_context = "## Similar Businesses\n"
    for business in similar_businesses:
        similar_context += f"- **{business['name']}**: {business['industry']}, {business['revenue_model']}, {business['development_stage']}, Revenue: {business['revenue_bracket']}\n"

    prompt = f"""
    You are an expert business analyst. Based on the following information about a {business_type} business:

    {qa_context}

    And industry comparison data:

    {similar_context}

    Create a comprehensive business report that would be useful for investors and founders.

    The report should include:
    1. Executive Summary
    2. Business Model Analysis
    3. Market Analysis and Target Customer
    4. Competitive Landscape
    5. Financial Overview and Revenue Streams
    6. Challenges and Risks
    7. Growth Opportunities
    8. Industry Benchmark Comparison
    9. Strategic Recommendations

    Format the report in markdown with clear headings and bullet points where appropriate.
    Be specific, data-driven where possible, and provide actionable insights.
    """

    model = genai.GenerativeModel("gemini-2.0-pro")
    response = model.generate_content(prompt)

    # Add benchmark insights as a separate section
    full_report = response.text + "\n\n## Industry Benchmark Analysis\n\n" + benchmark_insights

    return full_report, business_attributes, similar_businesses

def generate_industry_visualizations(industry):
    df = load_business_data_from_mongodb() # Load from MongoDB
    if df is None:
        return None, None, None

    # Filter by industry
    industry_df = df[df["Business Attributes.Business Fundamentals.Industry Classification.Primary Industry"].str.contains(industry, case=False, na=False)]

    if len(industry_df) < 5:
        return None, None, None

    # 1. Revenue Model Distribution
    revenue_model_counts = industry_df["Business Attributes.Business Fundamentals.Business Model.Primary Revenue Model"].value_counts()
    revenue_model_fig = px.pie(
        names=revenue_model_counts.index,
        values=revenue_model_counts.values,
        title=f"Revenue Models in {industry}",
        hole=0.4
    )

    # 2. Revenue Brackets Distribution
    revenue_bracket_counts = industry_df["Business Attributes.Financial Metrics.Revenue Brackets (Annual)"].value_counts()
    revenue_bracket_fig = px.bar(
        x=revenue_bracket_counts.index,
        y=revenue_bracket_counts.values,
        title=f"Revenue Brackets in {industry}",
        labels={'x': 'Revenue Bracket', 'y': 'Number of Businesses'}
    )

    # 3. Growth Rate Distribution
    growth_rate_counts = industry_df["Business Attributes.Growth & Scalability.Growth Rate"].value_counts()
    growth_rate_fig = px.bar(
        x=growth_rate_counts.index,
        y=growth_rate_counts.values,
        title=f"Growth Rates in {industry}",
        labels={'x': 'Growth Rate', 'y': 'Number of Businesses'}
    )

    return revenue_model_fig, revenue_bracket_fig, growth_rate_fig

def generate_competitive_landscape(business_attributes):
    df = load_business_data_from_mongodb() # Load from MongoDB
    if df is None:
        return None

    industry = business_attributes.get("Industry", "")

    if not industry or industry == "Unknown":
        return None

    # Filter by industry
    industry_df = df[df["Business Attributes.Business Fundamentals.Industry Classification.Primary Industry"].str.contains(industry, case=False, na=False)]

    if len(industry_df) < 5:
        return None

    # Create a scatter plot of businesses by development stage and innovation level
    dev_stage_map = {
        "Concept/Idea Only": 1,
        "Early Research & Development": 2,
        "Prototype Development": 3,
        "Beta Testing/Market Testing": 4,
        "Initial Market Launch": 5,
        "Revenue Generating": 6,
        "Growth Phase": 7,
        "Established Business": 8
    }

    innovation_map = {
        "Incremental Improvement": 1,
        "Moderate Innovation": 2,
        "Disruptive Innovation": 3,
        "Revolutionary/Breakthrough": 4
    }

    # Map values to numeric for plotting
    industry_df["dev_stage_numeric"] = industry_df["Business Attributes.Business Fundamentals.Development Stage"].map(
        lambda x: dev_stage_map.get(x, 0)
    )

    industry_df["innovation_numeric"] = industry_df["Business Attributes.Product/Service Attributes.Innovation Level"].map(
        lambda x: innovation_map.get(x, 0)
    )

    # Filter out rows with missing mappings
    plot_df = industry_df[(industry_df["dev_stage_numeric"] > 0) & (industry_df["innovation_numeric"] > 0)]

    if len(plot_df) < 5:
        return None

    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x="dev_stage_numeric",
        y="innovation_numeric",
        hover_name="business_name",
        size_max=10,
        color="Business Attributes.Financial Metrics.Revenue Brackets (Annual)",
        title=f"Competitive Landscape: {industry}",
        labels={
            "dev_stage_numeric": "Development Stage",
            "innovation_numeric": "Innovation Level"
        }
    )

    # Update x-axis ticks
    fig.update_xaxes(
        tickmode='array',
        tickvals=list(dev_stage_map.values()),
        ticktext=list(dev_stage_map.keys())
    )

    # Update y-axis ticks
    fig.update_yaxes(
        tickmode='array',
        tickvals=list(innovation_map.values()),
        ticktext=list(innovation_map.keys())
    )

    return fig

# --- Streamlit UI and Session State (No Changes Needed in this Section) ---
# (This section remains the same as in your previous complete code)

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

if 'business_attributes' not in st.session_state:
    st.session_state.business_attributes = {}

if 'similar_businesses' not in st.session_state:
    st.session_state.similar_businesses = []

if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "questionnaire"

# Function to start the questionnaire
def start_questionnaire():
    st.session_state.started = True
    st.session_state.current_question_index = 0
    st.session_state.questions = generate_initial_questions(st.session_state.business_type)
    st.session_state.qa_pairs = []
    st.session_state.report_generated = False

def next_question():
    current_q = st.session_state.questions[st.session_state.current_question_index]
    answer = st.session_state.current_answer

    # Add to QA pairs
    st.session_state.qa_pairs.append({
        "question": current_q["question"],
        "answer": answer,
        "category": current_q.get("category", "General")
    })

    # Move to the next question
    st.session_state.current_question_index += 1

    # If we have reached the end of available questions, generate new ones
    if st.session_state.current_question_index >= len(st.session_state.questions):
        new_questions = generate_follow_up_question(
            st.session_state.business_type,
            st.session_state.qa_pairs,
            remaining_questions=3
        )

        # Remove duplicates before adding new questions
        existing_questions = {q["question"] for q in st.session_state.questions}
        unique_new_questions = [q for q in new_questions if q["question"] not in existing_questions]

        st.session_state.questions.extend(unique_new_questions)


# Function to generate final report
def generate_report():
    st.session_state.report_generated = True

# Streamlit UI
st.set_page_config(layout="wide", page_title="Business Insights Platform")

# Sidebar for navigation
with st.sidebar:
    st.title("Business Insights")

    # Only show full navigation when report is generated
    if st.session_state.report_generated:
        selected = option_menu(
            "Navigation",
            ["Questionnaire", "Business Report", "Industry Insights", "Competitive Analysis", "Similar Businesses"],
            icons=["chat-dots", "file-earmark-text", "graph-up", "diagram-3", "people"],
            default_index=0,
        )
        st.session_state.current_tab = selected.lower().replace(" ", "_")
    else:
        selected = option_menu(
            "Navigation",
            ["Questionnaire"],
            icons=["chat-dots"],
            default_index=0,
        )
        st.session_state.current_tab = "questionnaire"

    # Show industry explorer option always
    if st.checkbox("Industry Explorer", value=False):
        st.session_state.current_tab = "industry_explorer"

    # Load dataset info from MongoDB
    df = load_business_data_from_mongodb() # Load from MongoDB
    if df is not None:
        st.write(f"Dataset: {len(df)} businesses")

        # Show top industries
        top_industries = df["Business Attributes.Business Fundamentals.Industry Classification.Primary Industry"].value_counts().head(10)
        with st.expander("Top Industries"):
            for industry, count in top_industries.items():
                st.write(f"{industry}: {count}")

# Main content based on selected tab
if st.session_state.current_tab == "questionnaire":
    st.title("Interactive Business Analysis")

    if not st.session_state.started:
        col1, col2 = st.columns([3, 2])

        with col1:
            st.subheader("Tell us about your business")
            business_type = st.text_input("What type of business are you analyzing?", key="business_type_input")

            if st.button("Start Analysis"):
                if business_type:
                    st.session_state.business_type = business_type
                    start_questionnaire()
                else:
                    st.error("Please enter a business type to continue")

        with col2:
            st.subheader("How it works")
            st.write("""
            1. Tell us what kind of business you're analyzing
            2. Answer targeted questions about your business
            3. Get a comprehensive business report with benchmarks and insights
            4. Explore industry data and competitive landscape
            """)

            st.image("https://image.freepik.com/free-vector/business-analysis-concept-illustration_114360-1512.jpg", width=250)


    else:
        # Display header with progress
        total_questions = max(8, len(st.session_state.questions))  # At least 8 questions expected
        progress = st.session_state.current_question_index / total_questions

        st.subheader(f"Analyzing: {st.session_state.business_type}")
        st.progress(progress)

        col1, col2 = st.columns([4, 1])

        with col1:
            # If we haven't answered all the questions and there are questions to answer
            if st.session_state.current_question_index < len(st.session_state.questions):
                current_q = st.session_state.questions[st.session_state.current_question_index]

                st.subheader(f"Question {current_q['id']}: {current_q['question']}")

                # See if there are similar questions in the database
                similar_questions = search_questions(current_q["question"])
                if similar_questions:
                    with st.expander("See similar questions others have asked"):
                        for q in similar_questions[:5]:
                            st.write(f"- {q}")

                # Answer input
                st.text_area("Your answer:", height=100, key="current_answer")

                if st.button("Next Question"):
                    next_question()
                    st.rerun()

            # If we've answered enough questions, show generate report button
            elif not st.session_state.report_generated and len(st.session_state.qa_pairs) >= 5:
                st.write("You've completed the initial questions! Generate your report or continue with more questions.")

                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Generate Business Report"):
                        report, attributes, similar = generate_business_report(
                            st.session_state.business_type,
                            st.session_state.qa_pairs
                        )
                        st.session_state.report = report
                        st.session_state.business_attributes = attributes
                        st.session_state.similar_businesses = similar
                        generate_report()
                        st.rerun()

                with col2:
                    if st.button("Continue with More Questions"):
                        # Generate new follow-up questions
                        new_questions = generate_follow_up_question(
                            st.session_state.business_type,
                            st.session_state.qa_pairs,
                            remaining_questions=3
                        )

                        # Add IDs to the new questions
                        for i, q in enumerate(new_questions):
                            q["id"] = len(st.session_state.questions) + i + 1

                        st.session_state.questions.extend(new_questions)
                        st.rerun()

        with col2:
            if st.session_state.current_question_index < len(st.session_state.questions):
                st.subheader("Question Categories")
                categories = {}
                for qa in st.session_state.qa_pairs:
                    cat = qa.get("category", "General")
                    categories[cat] = categories.get(cat, 0) + 1
                for q_index in range(st.session_state.current_question_index):
                    cat = st.session_state.questions[q_index].get("category", "General")
                    categories[cat] = categories.get(cat, 0) + 1

                if categories:
                    category_list = sorted(categories.items(), key=lambda item: item[1], reverse=True)
                    for category, count in category_list:
                        st.write(f"- {category}: {count} questions")
                else:
                    st.write("Answering your first questions...")
            else:
                st.empty() # placeholder if needed

elif st.session_state.current_tab == "business_report":
    if st.session_state.report_generated:
        st.title("Business Report")
        st.download_button(
            label="Download Full Report",
            data=st.session_state.report,
            file_name=f"BusinessReport_{st.session_state.business_type.replace(' ', '_')}.md",
            mime="text/markdown"
        )
        st.markdown(st.session_state.report)
    else:
        st.write("Report not generated yet. Please complete the questionnaire.")

elif st.session_state.current_tab == "industry_insights":
    st.title("Industry Insights")
    industry_for_insights = st.text_input("Enter industry name to explore:", st.session_state.business_type)

    if st.button("Generate Industry Insights", key="industry_insights_button"):
        if industry_for_insights:
            with st.spinner(f"Generating insights for {industry_for_insights}..."):
                insights = generate_industry_insights(industry_for_insights)
                st.session_state.industry_insights = insights # Store insights for persistence
                st.session_state.industry_insights_industry = industry_for_insights # Store industry name
                st.session_state.industry_visuals = generate_industry_visualizations(industry_for_insights) # Store visuals
        else:
            st.error("Please enter an industry name.")

    if 'industry_insights' in st.session_state and st.session_state.industry_insights_industry == industry_for_insights:
        st.subheader(f"Industry Insights for: {st.session_state.industry_insights_industry}")
        st.markdown(st.session_state.industry_insights)

    if st.session_state.industry_visuals:
        revenue_model_fig, revenue_bracket_fig, growth_rate_fig = st.session_state.industry_visuals

        if revenue_model_fig:
            st.plotly_chart(revenue_model_fig)
        if revenue_bracket_fig:
            st.plotly_chart(revenue_bracket_fig)
        if growth_rate_fig:
            st.plotly_chart(growth_rate_fig)

elif st.session_state.current_tab == "competitive_analysis":
    st.title("Competitive Landscape Analysis")

    if st.session_state.business_attributes:
        with st.spinner("Generating competitive landscape visualization..."):
            landscape_fig = generate_competitive_landscape(st.session_state.business_attributes)
            st.session_state.landscape_fig = landscape_fig # store the figure

        if st.session_state.landscape_fig:
            st.plotly_chart(st.session_state.landscape_fig)
        else:
            st.write("Could not generate competitive landscape visualization for this industry.  Ensure industry is correctly identified in the questionnaire.")
    else:
        st.write("Competitive analysis requires business attributes. Please generate the Business Report first.")

elif st.session_state.current_tab == "similar_businesses":
    st.title("Similar Businesses")
    if st.session_state.similar_businesses:
        st.write("Here are businesses similar to the one you are analyzing, based on our dataset:")
        for business in st.session_state.similar_businesses:
            with st.expander(business['name']):
                st.write(f"**Industry:** {business['industry']}")
                st.write(f"**Revenue Model:** {business['revenue_model']}")
                st.write(f"**Development Stage:** {business['development_stage']}")
                st.write(f"**Target Market:** {business['target_market']}")
                st.write(f"**Revenue Bracket:** {business['revenue_bracket']}")
    else:
        st.write("Similar businesses data is not available yet. Please generate the Business Report first.")

elif st.session_state.current_tab == "industry_explorer":
    st.title("Industry Explorer")
    df = load_business_data_from_mongodb() # Load from MongoDB
    if df is None:
        st.error("Dataset not loaded.")
    else:
        industries = df["Business Attributes.Business Fundamentals.Industry Classification.Primary Industry"].unique()
        selected_industry_explorer = st.selectbox("Select an Industry to Explore", industries)

        if selected_industry_explorer:
            st.subheader(f"Exploring Industry: {selected_industry_explorer}")
            industry_insights_explorer = generate_industry_insights(selected_industry_explorer)
            st.markdown(industry_insights_explorer)
            revenue_model_fig_explorer, revenue_bracket_fig_explorer, growth_rate_fig_explorer = generate_industry_visualizations(selected_industry_explorer)

            if revenue_model_fig_explorer:
                st.plotly_chart(revenue_model_fig_explorer)
            if revenue_bracket_fig_explorer:
                st.plotly_chart(revenue_bracket_fig_explorer)
            if growth_rate_fig_explorer:
                st.plotly_chart(growth_rate_fig_explorer)
