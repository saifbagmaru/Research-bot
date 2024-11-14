import os
import json
import psycopg2
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.llms import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from mistralai_azure import MistralAzure
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

# Azure AI Configuration
AZURE_AI_ENDPOINT = os.environ.get("AZURE_AI_ENDPOINT")
AZURE_AI_API_KEY = os.environ.get("AZURE_AI_API_KEY")

client = MistralAzure(azure_endpoint=AZURE_AI_ENDPOINT, azure_api_key=AZURE_AI_API_KEY)

DB_CONFIG = {
    "host": os.environ.get('HOST'),
    "database": os.environ.get('DATABASE_NAME'),
    "user": "bugbuster_admin",
    "password": os.environ.get('PASSWORD'),
    "port": os.environ.get('PORT'),
}


def serialize_datetime(obj: Any) -> str:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError("Type not serializable")

def execute_query(query: str, params: tuple = None) -> List[Dict]:
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                columns = [desc[0] for desc in cur.description]
                return [
                    {col: serialize_datetime(val) if isinstance(val, datetime) else val
                     for col, val in zip(columns, row)}
                    for row in cur.fetchall()
                ]
    except psycopg2.Error as error:
        print(f"Database error: {error}")
        return []

def retrieve_user_details(user_id: str) -> str:
    query = """
    SELECT u.*, up.*, p.*, su.* 
    FROM usermanagement_user u 
    LEFT JOIN usermanagement_userprofile up ON u.id = up.user_id 
    LEFT JOIN program_programs p ON u.id = p.user_id_id::uuid
    LEFT JOIN submission_submission su ON u.id = su.user_id::uuid
    WHERE u.id = %s;
    """
    results = execute_query(query, (user_id,))
    
    desired_keys = ["first_name", "last_name", "username", "role", "program_id", 
                    "program_type", "program_package", "program_title", "submission_title", 
                    "project_description", "expected_vulnerability_types", "submissions_id", 
                    "submission_title", "submission_status"]
    
    filtered_results = [{key: result[key] for key in desired_keys if key in result} for result in results]
    return json.dumps(filtered_results, default=serialize_datetime, indent=4)

def retrieve_submission_details(submission_id: str) -> str:
    query = """
    SELECT su.*, u.first_name, u.last_name, u.username, p.program_title, p.program_type, p.project_description
    FROM submission_submission su
    JOIN usermanagement_user u ON su.user_id::uuid = u.id
    JOIN program_programs p ON su.program_id::uuid = p.id
    WHERE su.id = %s;
    """
    results = execute_query(query, (submission_id,))
    return json.dumps(results, default=serialize_datetime, indent=4)

def retrieve_program_details(program_id: str) -> str:
    query = """
    SELECT p.*, u.first_name, u.last_name, u.username
    FROM program_programs p
    LEFT JOIN usermanagement_user u ON p.user_id_id::uuid = u.id
    WHERE p.id = %s;
    """
    results = execute_query(query, (program_id,))
    return json.dumps(results, default=serialize_datetime, indent=4)

def fetch_programs() -> List[Dict]:
    query = """
    SELECT id, program_type, program_title, project_description, project_tags, 
           expected_vulnerability_types
    FROM program_programs
    WHERE program_status = 'approved' AND is_deleted = False
    """
    return execute_query(query)

def fetch_filtered_programs():
    query = """
    select program_title, project_description, project_tags, scope_title, expected_vulnerability_types
    FROM program_programs
    WHERE program_status = 'approved' AND is_deleted = False
    """

    return execute_query(query)

def fetch_certifications(user_id: str) -> List[tuple]:
    query = """
    SELECT certification_title, certification_type, skill_title, skill_level, certification_short_description 
    FROM skill_certification 
    WHERE user_id_id = %s
    """
    return execute_query(query, (user_id,))

def suggest_programs(user_id: str) -> pd.DataFrame:
    programs = fetch_programs()
    certifications = fetch_certifications(user_id)

    if programs is None:
        print("No programs or certifications found.")
        return pd.DataFrame()

    df_programs = pd.DataFrame(programs)
    df_certifications = pd.DataFrame(certifications, columns=['certification_title', 'certification_type', 'skill_title', 'skill_level', 'certification_short_description'])

    if df_certifications.empty or df_certifications['skill_title'].isnull().all():
        print("No valid skills found for the user.")
        return pd.DataFrame()

    user_skills = ' '.join(df_certifications['skill_title'].dropna().tolist() + 
                           df_certifications['certification_short_description'].dropna().tolist())

    program_descriptions = df_programs.apply(lambda row: ' '.join(filter(None, [
        str(row['program_type']),
        str(row['program_title']),
        str(row['project_description']),
        str(row['project_tags']),
        str(row['expected_vulnerability_types'])
    ])), axis=1)

    if program_descriptions.empty:
        print("No valid program descriptions found.")
        return pd.DataFrame()

    tfidf_vectorizer = TfidfVectorizer()
    
    try:
        user_tfidf_matrix = tfidf_vectorizer.fit_transform([user_skills])
        program_tfidf_matrix = tfidf_vectorizer.transform(program_descriptions)
    except ValueError as e:
        print(f"Error during vectorization: {e}")
        return pd.DataFrame()

    cosine_similarities = cosine_similarity(user_tfidf_matrix, program_tfidf_matrix)
    top_indices = cosine_similarities[0].argsort()[-3:][::-1]  # Get top 3 matches
    
    suggested_programs = df_programs.iloc[top_indices]
    return suggested_programs[['program_type', 'program_title', 'project_description']]


def generate_llm_response(context_type, details, question, additional_info):
    system_message = f"""
   You are an AI-based cybersecurity assistant bot for researchers. Your task is to answer the user's question based on the provided information.

    Context Type: {context_type}

    Details:
    {details}

    
    Additional Information (if applicable):
    {additional_info}

    Instructions:
    1. Analyze the provided details and the question carefully.
    2. If the context is about a user:
       - Consider the user's profile, including their role, programs, and submissions.
       - If asked about program recommendations, explain why these programs are suitable based on the user's skills and certifications.
       - Provide a concise summary of each recommended program and its relevance to the user's profile.
    3. If the context is about a specific submission:
       - Provide relevant information about the submission, including its status, associated program, and user details.
       - If applicable, suggest next steps or improvements based on the submission status.
    4. If the context is about a specific program:
       - Provide details about the program, including its type, title, description, and associated user (if any).
       - Highlight key aspects of the program that might be relevant to potential participants.
    5. For other questions, provide accurate and relevant information based on the given details.
    6. Keep your responses concise, clear, and tailored to the cybersecurity context.
    7.Do not include or reference any personal identifiable information (PII) such as email addresses, phone numbers, user IDs, 
    or other sensitive personal details in your responses.
    
    Answer:
    """

    user_message = f"User question: {question}"

    resp = client.chat.complete(messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ], model="azureai")
    
    return resp.choices[0].message.content


def get_context_and_details(user_id: Optional[str] = None, submission_id: Optional[str] = None, program_id: Optional[str] = None) -> Tuple[str, str, str]:
    if user_id:
        context_type = "User Profile"
        details = retrieve_user_details(user_id)
        suggested_programs_df = suggest_programs(user_id)
        additional_info = suggested_programs_df.to_json(orient='records') if not suggested_programs_df.empty else ""
    elif submission_id:
        context_type = "Submission Details"
        details = retrieve_submission_details(submission_id)
        additional_info = ""
    elif program_id:
        context_type = "Program Details"
        details = retrieve_program_details(program_id)
        additional_info = ""
    else:
        raise ValueError("Either user_id, submission_id, or program_id must be provided")
    
    return context_type, details, additional_info


def main():
    st.title("Researcher Insights Bot")
    
    user_id = st.text_input("Enter User ID:")
    question = st.text_area("Ask a question based on user profile:")
    
    if st.button("Get Answer"):
        if user_id and question:
            context_type, details, additional_info = get_context_and_details(user_id=user_id)
            answer = generate_llm_response(context_type, details, question, additional_info)
            st.write("AI Response:")
            st.write(answer)
        else:
            st.error("Please provide both a user ID and a question.")
            
if __name__ == "__main__":
    main()
