from flask import Blueprint, render_template, session, redirect, url_for, request, jsonify
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from datetime import datetime
import re
import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

main_bp = Blueprint('main', __name__)

# --- ✨ 1. PDF Processing and Knowledge Base Setup ---

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

def load_and_process_pdfs():
    """
    Loads text from specified PDFs, combines them, and splits into chunks.
    This function runs once on startup.
    """
    pdf_files = ["app/first_file.pdf", "app/second_file.pdf", "app/third_file.pdf", "app/forth_file.pdf"]
    full_corpus = ""
    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            full_corpus += extract_text_from_pdf(pdf_path) + "\n"
        else:
            print(f"Warning: PDF file not found at {pdf_path}")
            
    if not full_corpus:
        print("Warning: No text was extracted from PDFs. The knowledge base is empty.")
        return [], None
        
    chunks = chunk_text(full_corpus)
    vectorizer = TfidfVectorizer().fit(chunks) # Fit the vectorizer on the chunks
    print(f"PDFs processed into {len(chunks)} chunks.")
    return chunks, vectorizer

# --- Global variables for the knowledge base ---
# Load and process the PDFs once when the application starts.
KNOWLEDGE_CHUNKS, TFIDF_VECTORIZER = load_and_process_pdfs()

def retrieve_relevant_chunks(user_question, chunks, vectorizer, top_k=3):
    """
    Retrieves the most relevant text chunks from the knowledge base.
    """
    if not user_question or not chunks or not vectorizer:
        return []
    
    # You must transform both the question and chunks with the *same* fitted vectorizer
    question_vec = vectorizer.transform([user_question])
    chunk_vecs = vectorizer.transform(chunks)
    
    similarities = cosine_similarity(question_vec, chunk_vecs).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# --- 2. AI Chain Creation (with updated prompt) ---

def create_chain():
    """Creates and configures the LangChain LLM chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        # --- ⚠️ ACTION REQUIRED: It is highly recommended to use environment variables ---
        # Example: google_api_key=os.environ.get("GOOGLE_API_KEY")
        google_api_key="AIzaSyAdVflqSS1xlyKwfgzEfCYCLahuNtpa29k", # Replace with your key
        temperature=0.6
    )

    # --- ✨ MODIFICATION: Added 'retrieved_context' to the prompt ---
    prompt = PromptTemplate(
        input_variables=["history", "user_input", "retrieved_context"],
        template="""
[System] You are ProjectPro, a professional project management assistant.

Your task is to generate structured, markdown-formatted answers. **STRICTLY follow the formatting rules and the example provided below.**

---

**Formatting Rules:**

1.  Use `**bold section titles**`.  
2.  Use **numbered lists** for main points. **End every list item and paragraph with two spaces** to force a markdown line break.  
3.  Leave **one blank line** between list items and between sections for readability.  
4.  Use sub-bullets with a hyphen (`-`) for details under a numbered item.

---

**Output Format Example:**

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

**Start your well-structured, markdown-formatted response below.**

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
            # Keep intentional blank lines
            formatted_lines.append("")
            continue

        # Add two spaces to the end of any line that contains content 
        # but isn't a section title or a sub-bullet.
        # This creates the <br> line break in Markdown.
        if line.startswith('**') and line.endswith('**'):
            formatted_lines.append(line) # It's a title, leave it
            formatted_lines.append("") # Ensure a blank line after title
        elif line.startswith('-'):
             # It's a sub-bullet, add two spaces
            formatted_lines.append(line + "  ")
        elif re.match(r'^\d+\.', line):
             # It's a numbered list item, add two spaces and a blank line after
            formatted_lines.append(line + "  ")
            # Check if the next line is not blank to add a separator
            if i + 1 < len(lines) and lines[i+1].strip():
                 formatted_lines.append("")
        else:
            # Regular paragraph text
            formatted_lines.append(line + "  ")

    # Remove any potential duplicate blank lines
    final_output = "\n".join(formatted_lines)
    return re.sub(r'\n{3,}', '\n\n', final_output).strip()
@main_bp.route('/')
def home():
    """Renders the main landing page."""
    return render_template('main.html', logged_in='user_ID' in session)

@main_bp.route('/pm-chat', methods=['GET', 'POST'])
def pm_chat():
    """Handles the chat page rendering and the AJAX chat requests."""
    if 'user_ID' not in session:
        return redirect(url_for('auth.login'))

    if 'conversation_history' not in session:
        session['conversation_history'] = []

    if request.method == 'POST':
        try:
            data = request.get_json()
            user_input = sanitize_input(data.get('message', ''))

            if not user_input:
                return jsonify({"error": "Message cannot be empty."}), 400

            session['conversation_history'].append({
                'role': 'user', 'content': user_input, 'timestamp': datetime.now().isoformat()
            })
            session.modified = True

            history = "\n".join(
                f"{msg['role']}: {msg['content']}" for msg in session['conversation_history'][:-1]
            )
            
            relevant_chunks = retrieve_relevant_chunks(user_input, KNOWLEDGE_CHUNKS, TFIDF_VECTORIZER)
            retrieved_context = "\n---\n".join(relevant_chunks) if relevant_chunks else "No relevant context found."

            chain = create_chain()
            response = chain.invoke({
                "user_input": user_input, 
                "history": history,
                "retrieved_context": retrieved_context
            })
            ai_response = response.get('text', '').strip() or "I'm not sure how to help with that. Can you rephrase?"

            # --- ✨ KEY MODIFICATION: Apply formatting before sending to user ---
            formatted_ai_response = format_response_for_markdown(ai_response)

            session['conversation_history'].append({
                'role': 'ai', 'content': formatted_ai_response, 'timestamp': datetime.now().isoformat()
            })
            session.modified = True

            # Return the CLEANED response
            return jsonify({"response": formatted_ai_response})

        except Exception as e:
            print(f"AI Chat Error: {str(e)}")
            return jsonify({
                "error": "The AI assistant is currently unavailable. Please try again later."
            }), 500

    return render_template('pm_chat.html',
                           logged_in=True,
                           conversation_history=session['conversation_history'])