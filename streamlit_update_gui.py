import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
from fpdf import FPDF
import os
import re
st.markdown("""
<style>

/* Keep sidebar gradient */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #4B0082, #6a11cb, #2575fc) !important;
}

/* Sidebar title text color fix */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] span {
    color: white !important;
}

/* --- RADIO BUTTONS FIX (safe targeting) --- */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label > div {
    color: white !important;
    font-size: 18px !important;
    font-weight: 600 !important;
}

/* Extra safety: Streamlit sometimes wraps label text in span */
section[data-testid="stSidebar"] div[data-testid="stRadio"] label span {
    color: white !important;
}

/* Radio circle styling */
section[data-testid="stSidebar"] input[type="radio"] {
    accent-color: #FFD700 !important; /* gold */
}

</style>
""", unsafe_allow_html=True)



# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="PDF Bot Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("📄 PDF Bot Pro")
page = st.sidebar.radio("Select Functionality:", ["Home", "Summary", "Quiz", "Chatbot"])

# --- Helper: Load and Split PDFs ---
def load_and_split_pdfs(files):
    all_docs = []
    for uploaded_file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        all_docs.extend(splits)
    return all_docs


# --- For Display (Markdown Style in Streamlit) ---
def create_styled_pdf(content):
    """
    Convert plain text or MCQs into clean Markdown for Streamlit display.
    """
    content = re.sub(r"(\d+)\.", r"### Question \1", content)
    content = re.sub(r"\*Correct answer: (.+?)\*", r"**Answer: \1**", content)
    content = content.replace("---", "\n---\n")
    return content


# --- For PDF Download (Styled PDF) ---
def generate_styled_pdf(content, filename="output.pdf"):
    """
    Generate a clean, styled PDF with headings, spacing, and readable fonts.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("DejaVu", "", "fonts/DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "fonts/DejaVuSans-Bold.ttf", uni=True)
    pdf.set_auto_page_break(auto=True, margin=15)

    def add_text(line):
        line = line.strip()
        if not line:
            pdf.ln(6)
            return

        # Headings or numbered titles
        if re.match(r"^(###|[0-9]+\.)", line):
            pdf.set_font("DejaVu", "B", 14)
            pdf.set_text_color(0, 51, 153)
            line = re.sub(r"^(###\s*|[0-9]+\.)", "", line).strip()
            pdf.multi_cell(0, 8, line)
            pdf.ln(4)
        # Correct answers
        elif "Answer:" in line or "Correct answer" in line:
            pdf.set_font("DejaVu", "B", 12)
            pdf.set_text_color(34, 139, 34)
            pdf.multi_cell(0, 8, line)
            pdf.ln(3)
        # Section dividers
        elif line.strip() == "---":
            pdf.set_text_color(180, 180, 180)
            pdf.set_font("DejaVu", "", 10)
            pdf.cell(0, 5, "-" * 80, 0, 1, "C")
            pdf.ln(2)
        # Normal text
        else:
            pdf.set_font("DejaVu", "", 12)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 7, line)
            pdf.ln(2)

    for line in content.split("\n"):
        add_text(line)

    tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    pdf.output(tmp_path)
    return tmp_path


# --- File Upload ---
st.markdown("""
    <style>
        /* Style the uploader container */
        .file-upload-wrapper {
            background: linear-gradient(135deg, #6a11cb, #2575fc);
            padding: 25px;
            border-radius: 15px;
            text-align: center;
            color: white;
            font-size: 18px;
            font-weight: 500;
            font-family: 'Segoe UI', sans-serif;
            box-shadow: 0 6px 20px rgba(0,0,0,0.25);
            transition: all 0.3s ease;
        }

        .file-upload-wrapper:hover {
            transform: scale(1.02);
            box-shadow: 0 10px 35px rgba(0,0,0,0.35);
        }

        /* Hide default label and style it */
        .file-upload-wrapper label {
            color: white;
            font-weight: bold;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="file-upload-wrapper">
        📤 <b>Upload PDF files</b>
    </div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "",
    type=["pdf"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)
if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded successfully!")
    splits = load_and_split_pdfs(uploaded_files)


    # Create embeddings & vector store
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embedding_model)

    # Initialize LLM
    llm = ChatOpenAI(
        api_key=st.secrets["OPENROUTER_API_KEY"],  # Replace with your own key
        base_url="https://openrouter.ai/api/v1",
        model="openai/gpt-oss-safeguard-20b"
    )

    full_text = "\n\n".join(doc.page_content for doc in splits)


# --- HOME PAGE ---
if page == "Home":
    st.markdown("""
        <style>
            .home-title {
                text-align: center;
                color: #4B0082;
                font-size: 50px;
                font-weight: bold;
                font-family: 'Poppins', sans-serif;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                margin-bottom: 20px;
            }

            .home-container {
                background: linear-gradient(135deg, #4B0082 0%, #6a11cb 50%, #2575fc 100%);
                padding: 40px;
                border-radius: 20px;
                color: white;
                font-family: 'Segoe UI', sans-serif;
                box-shadow: 0px 6px 20px rgba(0,0,0,0.25);
                max-width: 800px;
                margin: auto;
            }

            .home-container h2 {
                text-align: center;
                font-size: 30px;
                margin-bottom: 15px;
                color: #FFD700;
                text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
            }

            .home-container p {
                font-size: 18px;
                text-align: center;
                line-height: 1.6;
            }

            .home-container ul {
                font-size: 17px;
                line-height: 1.9;
                list-style: none;
                padding: 0;
            }

            .home-container li::before {
                content: "✨ ";
            }

            .cta {
                text-align: center;
                margin-top: 20px;
                font-size: 18px;
                background-color: rgba(255,255,255,0.1);
                padding: 12px 20px;
                border-radius: 10px;
                display: inline-block;
                color: #FFF;
                transition: all 0.3s ease;
            }

            .cta:hover {
                background-color: rgba(255,255,255,0.25);
                transform: scale(1.05);
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 class='home-title'>📄 PDF Bot Pro</h1>", unsafe_allow_html=True)

    st.markdown("""
        <div class="home-container">
            <h2>🤖 Your Smart PDF Assistant</h2>
            <p>
                Welcome to <b>PDF Bot Pro</b> — the intelligent way to interact with your PDFs!
            </p>
            <ul>
                <li>Upload <b>multiple PDFs</b> in one go</li>
                <li>Generate concise <b>Summaries</b> and interactive <b>MCQs</b></li>
                <li>Chat directly with your PDF content</li>
                <li>Export your work as a <b>stylish downloadable PDF</b></li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# --- SUMMARY PAGE ---
if page == "Summary" and uploaded_files:
    st.markdown("<h2 style='color: #FF4500;'>📃Summary</h2>", unsafe_allow_html=True)
    summary_prompt_template = """
    Provide a concise and well-structured summary of the following text from PDFs.
    Avoid unnecessary repetition or unrelated content.

    Text:
    {text}
    """
    prompt = PromptTemplate.from_template(summary_prompt_template)
    summary_chain = (
        {"text": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    with st.spinner("Generating summary..."):
        summary = summary_chain.invoke(full_text)
        st.markdown(summary)

        # PDF download (styled)
        pdf_path = generate_styled_pdf(summary, filename="summary.pdf")
        with open(pdf_path, "rb") as f:
            st.markdown("""
    <style>
        /* Stylish Download Button */
        .stDownloadButton button {
            background-color: #6A11CB !important;  /* Purple shade */
            color: white !important;
            padding: 12px 22px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            transition: 0.3s ease;
        }

        .stDownloadButton button:hover {
            background-color: #2575FC !important; /* Light blue hover */
            transform: translateY(-2px); /* small hover lift effect */
            box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        }

        .stDownloadButton button:active {
            transform: scale(0.97);
        }
    </style>
""", unsafe_allow_html=True)

            st.download_button("Download Summary as PDF", data=f, file_name="summary.pdf", mime="application/pdf")


# --- MCQs PAGE ---
if page == "Quiz" and uploaded_files:
    st.markdown("<h2 style='color: #008B8B;'>📝Quiz Generator</h2>", unsafe_allow_html=True)
    mcq_prompt_template = """
    You are a teacher. Based ONLY on the text below, generate 5–12 MCQs.
    Each question should have 4 options (A, B, C, D) .

    Text:
    {text}
    """
    mcq_prompt = PromptTemplate.from_template(mcq_prompt_template)
    mcq_chain = (
        {"text": RunnablePassthrough()}
        | mcq_prompt
        | llm
        | StrOutputParser()
    )

    with st.spinner("Generating MCQs..."):
        mcqs = mcq_chain.invoke(full_text)
        styled_mcqs = create_styled_pdf(mcqs)
        st.markdown(styled_mcqs)

        # Styled PDF
        pdf_path = generate_styled_pdf(mcqs, filename="mcqs.pdf")
        with open(pdf_path, "rb") as f:
            st.markdown("""
    <style>
        /* Stylish Download Button */
        .stDownloadButton button {
            background-color: #6A11CB !important;  /* Purple shade */
            color: white !important;
            padding: 12px 22px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            transition: 0.3s ease;
        }

        .stDownloadButton button:hover {
            background-color: #2575FC !important; /* Light blue hover */
            transform: translateY(-2px); /* small hover lift effect */
            box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        }

        .stDownloadButton button:active {
            transform: scale(0.97);
        }
    </style>
""", unsafe_allow_html=True)

            st.download_button(" Download MCQs as PDF", data=f, file_name="mcqs.pdf", mime="application/pdf")


# --- CHATBOT PAGE ---
if page == "Chatbot" and uploaded_files:
    st.markdown("<h2 style='color: #6A5ACD;'>💬 Chatbot (PDF-based)</h2>", unsafe_allow_html=True)
    user_question = st.text_input("Ask something about your uploaded PDFs:")
    if st.button("Get Answer"):
        if user_question.strip():
            retriever = vectorstore.as_retriever()
            prompt_template = """
            Answer the following question based ONLY on the provided context.
            If not found in the text, respond with "I do not have that information in the document."

            Context:
            {context}

            Question:
            {question}
            """
            prompt = PromptTemplate.from_template(prompt_template)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            with st.spinner("Generating answer..."):
                answer = rag_chain.invoke(user_question)
                st.markdown(f"**Answer:** {answer}")
