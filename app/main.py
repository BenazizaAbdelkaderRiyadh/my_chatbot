from flask import Blueprint, render_template, session, redirect, url_for, request, jsonify, send_from_directory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from datetime import datetime
import re
import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

main_bp = Blueprint('main', __name__)

# --- ✨ 1. PDF Processing and Knowledge Base Setup ---

# --- Global variables for the dynamic knowledge base ---
KNOWLEDGE_CORPUS = ""
KNOWLEDGE_CHUNKS = []
TFIDF_VECTORIZER = TfidfVectorizer()

def extract_text_from_pdf(pdf_path):
    """Extracts text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        print(f"Successfully extracted text from {pdf_path}")
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

def chunk_text(text, chunk_size=500):
    """Splits text into smaller chunks of words."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def update_knowledge_base(new_text=""):
    """
    Updates the global knowledge base with new text, re-chunks, and re-fits the vectorizer.
    """
    global KNOWLEDGE_CORPUS, KNOWLEDGE_CHUNKS, TFIDF_VECTORIZER
    
    if new_text:
        KNOWLEDGE_CORPUS += "\n" + new_text
        
    KNOWLEDGE_CHUNKS = chunk_text(KNOWLEDGE_CORPUS)
    
    if KNOWLEDGE_CHUNKS:
        TFIDF_VECTORIZER = TfidfVectorizer().fit(KNOWLEDGE_CHUNKS)
        print(f"Knowledge base updated. Total chunks: {len(KNOWLEDGE_CHUNKS)}")
    else:
        TFIDF_VECTORIZER = TfidfVectorizer()
        print("Knowledge base is currently empty.")

def initial_pdf_load():
    """
    Loads text from specified PDFs and populates the knowledge base on startup.
    """
    global KNOWLEDGE_CORPUS
    pdf_files = ["app/first_file.pdf", "app/second_file.pdf", "app/third_file.pdf", "app/forth_file.pdf"]
    initial_corpus = ""
    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            initial_corpus += extract_text_from_pdf(pdf_path) + "\n"
        else:
            print(f"Warning: Initial PDF file not found at {pdf_path}")
            
    if initial_corpus:
        KNOWLEDGE_CORPUS = initial_corpus
        update_knowledge_base()
    else:
        print("Warning: No initial text was extracted from PDFs. The knowledge base is empty.")

initial_pdf_load()

def retrieve_relevant_chunks(user_question, top_k=3):
    """
    Retrieves the most relevant text chunks from the knowledge base.
    """
    if not user_question or not KNOWLEDGE_CHUNKS:
        return []
    
    question_vec = TFIDF_VECTORIZER.transform([user_question])
    chunk_vecs = TFIDF_VECTORIZER.transform(KNOWLEDGE_CHUNKS)
    
    similarities = cosine_similarity(question_vec, chunk_vecs).flatten()
    
    effective_top_k = min(top_k, len(KNOWLEDGE_CHUNKS))
    
    top_indices = similarities.argsort()[-effective_top_k:][::-1]
    return [KNOWLEDGE_CHUNKS[i] for i in top_indices]

# --- ✨ 2. NEW PDF Upload Route ---
@main_bp.route('/upload-pdf', methods=['POST'])
def upload_pdf():
    """
    Handles uploading a new PDF file and updating the knowledge base.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        upload_folder = "/content/uploads" 
        os.makedirs(upload_folder, exist_ok=True)
        file_path = os.path.join(upload_folder, file.filename)
        file.save(file_path)

        new_text = extract_text_from_pdf(file_path)
        if new_text:
            update_knowledge_base(new_text)
            print(f"📄 PDF loaded and knowledge base updated: {file.filename}")
            return jsonify({"status": "loaded", "filename": file.filename})
        else:
            return jsonify({"error": "Could not extract text from PDF"}), 400

    except Exception as e:
        print("❌ Upload error:", str(e))
        return jsonify({"error": str(e)}), 500

# --- 3. AI Chain Creation ---

def create_chain():
    """Creates and configures the LangChain LLM chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key="AIzaSyAdVflqSS1xlyKwfgzEfCYCLahuNtpa29k", # Replace with your key
        temperature=0.6
    )
    prompt = PromptTemplate(
        input_variables=["history", "user_input", "retrieved_context"],
        template="""
[System] You are ProjectPro, a professional project management assistant. Your primary role is to provide structured, professional advice on project management.

---
**BEHAVIORAL RULES:**

1.  **Greeting:** If the user gives a simple greeting (like "hi", "hello"), you MUST ONLY respond with: "Hi! I'm ProjectPro, your project management assistant. How can I help you today?". Do not add any other content.

2.  **Summarization:** If the user explicitly asks for a "summary" of a file or document, you MUST ONLY provide a concise summary of the `retrieved_context`. Do NOT use the detailed project format below. Start your response with "Here is a brief summary of the document:" and then provide the summary in one or two paragraphs.

3.  **Project Analysis (Default Behavior):** For all other requests related to analyzing a project, its risks, tasks, or scope, you MUST generate a structured, markdown-formatted answer. **STRICTLY follow the formatting rules and the example provided below.**

---

**Formatting Rules for Project Analysis:**

1.  Use `**bold section titles**`.
2.  Use **numbered lists** for main points. **End every list item and paragraph with two spaces** to force a markdown line break.
3.  Leave **one blank line** between list items and between sections for readability.
4.  Use sub-bullets with a hyphen (`-`) for details under a numbered item.

---

**Output Format Example for Project Analysis:**

**Project Overview**
This is a brief summary of the project. It describes the main goals and objectives in one or two sentences.

**Project Risks**
1.  **Human:** A potential risk related to the team or stakeholders.

2.  **Technical:** A potential risk related to technology or integration.
    - A sub-point detailing a specific technical challenge.

**Task Plan**
1.  **Phase 1: Discovery:** A high-level task for the first phase.

2.  **Phase 2: Development:** Another high-level task.

**Clarifying Questions**
1.  What is the primary success metric for this project?

2.  Are there any hard deadlines we must meet?

**Next Steps**
1.  The first action item for the team.

2.  The second action item.

---

**CONTEXT FOR YOUR RESPONSE:**

- **Relevant Context from Documents:**
  {retrieved_context}

- **Previous Conversation History:**
  {history}

- **User Input:**
  {user_input}

---

**Start your response below, strictly following the BEHAVIORAL RULES.**

Assistant:
"""
    )
    return LLMChain(llm=llm, prompt=prompt)

def sanitize_input(text):
    """Removes potentially unwanted characters from user input."""
    return re.sub(r'[^\w\s,.?!-]', '', text).strip()
    
def format_response_for_markdown(text):
    """
    Programmatically cleans and formats the AI's response to ensure proper Markdown rendering.
    """
    lines = text.split('\n')
    formatted_lines = []
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            formatted_lines.append("")
            continue

        if line.startswith('**') and line.endswith('**'):
            formatted_lines.append(line)
            formatted_lines.append("") 
        elif line.startswith('-'):
            formatted_lines.append(line + "  ")
        elif re.match(r'^\d+\.', line):
            formatted_lines.append(line + "  ")
            if i + 1 < len(lines) and lines[i+1].strip():
                 formatted_lines.append("")
        else:
            formatted_lines.append(line + "  ")

    final_output = "\n".join(formatted_lines)
    return re.sub(r'\n{3,}', '\n\n', final_output).strip()

def save_chat_history(user_id, history):
    """Saves the conversation history to a JSON file for a specific user."""
    history_folder = "chat_histories"
    os.makedirs(history_folder, exist_ok=True)
    file_path = os.path.join(history_folder, f"{user_id}_history.json")
    try:
        with open(file_path, 'w') as f:
            json.dump(history, f, indent=4)
    except Exception as e:
        print(f"Error saving chat history for user {user_id}: {e}")

def load_chat_history(user_id):
    """Loads the conversation history from a JSON file for a specific user."""
    history_folder = "chat_histories"
    file_path = os.path.join(history_folder, f"{user_id}_history.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading chat history for user {user_id}: {e}")
            return []
    return []
    
@main_bp.route('/')
def home():
    """Renders the main landing page."""
    return render_template('main.html', logged_in='user_ID' in session)

## MODIFICATION: New route to get the user's history file
@main_bp.route('/get-history')
def get_history():
    """Provides the user's chat history JSON file for review."""
    if 'user_ID' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    user_id = session['user_ID']
    history_folder = "chat_histories"
    filename = f"{user_id}_history.json"
    
    # Use an absolute path for security with send_from_directory
    history_dir = os.path.join(os.getcwd(), history_folder)

    if not os.path.exists(os.path.join(history_dir, filename)):
        return jsonify({"error": "No history found for this user."}), 404

    # send_from_directory is the secure way to send files from a directory
    return send_from_directory(directory=history_dir, path=filename, as_attachment=False)

@main_bp.route('/pm-chat', methods=['GET', 'POST'])
def pm_chat():
    """Handles chat page rendering and AJAX requests, with history persistence."""
    if 'user_ID' not in session:
        return redirect(url_for('auth.login'))

    user_id = session['user_ID']

    if 'conversation_history' not in session:
        session['conversation_history'] = load_chat_history(user_id)

    if request.method == 'POST':
        try:
            data = request.get_json()
            user_input = sanitize_input(data.get('message', ''))

            if not user_input:
                return jsonify({"error": "Message cannot be empty."}), 400

            session['conversation_history'].append({
                'role': 'user', 'content': user_input, 'timestamp': datetime.now().isoformat()
            })

            history = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in session['conversation_history'][:-1]
            )
            
            relevant_chunks = retrieve_relevant_chunks(user_input)
            retrieved_context = "\n---\n".join(relevant_chunks) if relevant_chunks else "No relevant context found."

            chain = create_chain()
            response = chain.invoke({
                "user_input": user_input, 
                "history": history,
                "retrieved_context": retrieved_context
            })
            ai_response = response.get('text', '').strip() or "I'm not sure how to help with that. Can you rephrase?"

            formatted_ai_response = format_response_for_markdown(ai_response)

            session['conversation_history'].append({
                'role': 'ai', 'content': formatted_ai_response, 'timestamp': datetime.now().isoformat()
            })
            
            save_chat_history(user_id, session['conversation_history'])
            session.modified = True

            return jsonify({"response": formatted_ai_response})

        except Exception as e:
            print(f"AI Chat Error: {str(e)}")
            return jsonify({
                "error": "The AI assistant is currently unavailable. Please try again later."
            }), 500

    return render_template('pm_chat.html',
                           logged_in=True,
                           conversation_history=session['conversation_history'])
