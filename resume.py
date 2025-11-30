import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnableMap
import google.generativeai as genai
from dotenv import load_dotenv
import shutil
import re
import io
import csv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure Google AI API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    # if running via streamlit, show error in UI
    st.error("Please set your GOOGLE_API_KEY in a .env file")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Setup embedding model (keeps same model)
embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create or load Chroma vector store
VECTOR_STORE_DIR = "chroma_store"
if os.path.exists(VECTOR_STORE_DIR):
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)
else:
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vectorstore = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embedding_model)

# Extract text from uploaded files
def extract_text_from_resume(file):
    temp_file_path = f"temp_{file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())

    file_extension = os.path.splitext(file.name)[1].lower()
    try:
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        documents = loader.load()
        text = " ".join([doc.page_content for doc in documents])
        return text
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Text splitting
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents([text])

# Store resume analysis in vector store
def store_resume_analysis(resume_text, analysis, doc_id):
    documents = split_text(analysis)
    # Add prefixed IDs so they are unique
    vectorstore.add_documents(documents, ids=[f"{doc_id}_chunk_{i}" for i in range(len(documents))])
    vectorstore.persist()

# Extract percentage score from analysis text
def extract_suitability_score(text):
    # Searches for "Suitability Score: XX%" pattern
    match = re.search(r"Suitability Score:\s*(\d{1,3})\s*%", text)
    if match:
        # clamp between 0 and 100
        val = int(match.group(1))
        return max(0, min(100, val))
    return None

# Build chain (kept as a function to re-use)
def build_analysis_chain(google_api_key, model_name="gemini-2.0-flash", temperature=0.2):
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=google_api_key,
        temperature=temperature
    )
    prompt_template = PromptTemplate(
        input_variables=["job_requirements", "resume_text"],
        template="""
You are an expert HR and recruitment specialist. Analyze the resume below against the job requirements.

Job Requirements:
{job_requirements}

Resume:
{resume_text}

Provide a structured analysis of how well the resume matches the job requirements.
At the end, clearly state a "Suitability Score" as a percentage (0-100%) based on how well the resume aligns with the job.
Format: Suitability Score: XX%

Also include 2-3 short bullets listing the main strengths and 1-2 short bullets listing the gaps.
"""
    )

    chain = (
        RunnableMap({
            "job_requirements": lambda x: x["job_requirements"],
            "resume_text": lambda x: x["resume_text"]
        })
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain

# Main App
def main():
    st.set_page_config(page_title="Resume Screening & Ranking", layout="wide", page_icon="🧾")
    st.title("Resume Screening & Ranking")

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        model_choice = st.selectbox("LLM model", ["gemini-2.0-flash", "gemini-1.0"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2)
        persist_db = st.checkbox("Save analyses to vector DB (Chroma)", value=True)
        max_files = st.number_input("Max resumes to analyze at once", min_value=1, max_value=20, value=10)
        st.markdown("---")
        st.markdown("Make sure `.env` contains `GOOGLE_API_KEY`.")

    # Main input area
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Job Requirements")
        job_requirements = st.text_area("Enter job requirements", height=250)
        st.header("Upload Resumes (multiple)")
        uploaded_files = st.file_uploader(
            "Upload one or more resumes (pdf/docx/txt)",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        if uploaded_files:
            st.info(f"{len(uploaded_files)} file(s) selected — will analyze up to {max_files}.")
    with col2:
        st.header("Quick Actions")
        analyze_btn = st.button("Analyze & Rank Resumes", type="primary")
        st.markdown("---")
        st.header("Tips")
        st.write("- Upload multiple resumes to get a ranked list.")
        st.write("- Scores are produced by the LLM. You can review raw analysis for transparency.")
        st.write("- Persistent DB saves chunked analysis to `chroma_store/`.")

    # Run analysis when button clicked
    if analyze_btn:
        if not uploaded_files or len(uploaded_files) == 0:
            st.warning("Please upload at least one resume file.")
            st.stop()
        if not job_requirements or job_requirements.strip() == "":
            st.warning("Please enter job requirements for meaningful scoring.")
            st.stop()

        # Limit number of files analyzed at once
        files_to_process = uploaded_files[:int(max_files)]

        # Prepare chain
        chain = build_analysis_chain(GOOGLE_API_KEY, model_name=model_choice, temperature=temperature)

        results = []
        progress_bar = st.progress(0)
        total = len(files_to_process)
        for idx, file in enumerate(files_to_process, start=1):
            try:
                st.info(f"Processing {file.name} ({idx}/{total})")
                resume_text = extract_text_from_resume(file)

                # invoke LLM chain
                analysis = chain.invoke({
                    "job_requirements": job_requirements,
                    "resume_text": resume_text
                })

                # Extract suitability score
                score = extract_suitability_score(analysis)
                if score is None:
                    # if LLM didn't include a score, try to fallback by heuristics:
                    # (we can set 0 to indicate unknown)
                    score = 0

                # Optionally store analysis
                if persist_db:
                    try:
                        doc_id = f"{os.path.splitext(file.name)[0]}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
                        store_resume_analysis(resume_text, analysis, doc_id)
                    except Exception as e:
                        st.warning(f"Failed to persist analysis for {file.name}: {e}")

                # Add to results list
                results.append({
                    "filename": file.name,
                    "score": score,
                    "analysis": analysis,
                    "resume_text": resume_text
                })

                # Update progress
                progress_bar.progress(int((idx / total) * 100))

            except Exception as e:
                st.error(f"Error processing {file.name}: {e}")
                # still continue with next files

        progress_bar.empty()
        if len(results) == 0:
            st.warning("No analyses produced.")
            st.stop()

        # Sort results by score descending
        ranked = sorted(results, key=lambda r: r["score"], reverse=True)
        # Build leaderboard table
        table_rows = []
        for rank_idx, item in enumerate(ranked, start=1):
            preview = item["analysis"][:400].replace("\n", " ")
            table_rows.append({
                "Rank": rank_idx,
                "Filename": item["filename"],
                "Score (%)": item["score"],
                "Preview": preview
            })

        st.header("Ranked Resumes")
        st.table(table_rows)

        # Show each result with expanders and provide download buttons
        for rank_idx, item in enumerate(ranked, start=1):
            with st.expander(f"{rank_idx}. {item['filename']} — {item['score']}%"):
                st.markdown("**AI Analysis:**")
                st.write(item["analysis"])
                st.markdown("**Resume Extract (first 1000 chars):**")
                st.text(item["resume_text"][:1000])
                # Download analysis text
                st.download_button(
                    label="Download full analysis",
                    data=item["analysis"],
                    file_name=f"{os.path.splitext(item['filename'])[0]}_analysis.txt"
                )

        # Provide CSV download of leaderboard
        csv_buffer = io.StringIO()
        csv_writer = csv.writer(csv_buffer)
        csv_writer.writerow(["Rank", "Filename", "Score (%)", "Analysis (short)"])
        for r_idx, item in enumerate(ranked, start=1):
            csv_writer.writerow([r_idx, item["filename"], item["score"], item["analysis"][:400].replace("\n", " ")])

        csv_contents = csv_buffer.getvalue()
        st.download_button("Download leaderboard CSV", csv_contents, file_name="resume_leaderboard.csv", mime="text/csv")

if __name__ == "__main__":
    main()
