from flask import Flask, render_template, request, jsonify, send_file, Response
from config import Config
from models import db, Document
from werkzeug.utils import secure_filename
import os
import shutil
from pathlib import Path
import magic
import PyPDF2
import io
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.datastructures import FileStorage
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Remove max content length restriction
app.config['MAX_CONTENT_LENGTH'] = None

# Thread pool for handling concurrent uploads
executor = ThreadPoolExecutor(max_workers=4)
upload_tasks = {}


def compute_file_hash(file_path):
    """Compute MD5 hash of file in chunks"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def stream_file_upload(file: FileStorage, destination_path: str):
    """Stream file upload in chunks"""
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with open(destination_path, 'wb') as f:
        while True:
            chunk = file.read(CHUNK_SIZE)
            if not chunk:
                break
            f.write(chunk)

def process_uploaded_file(file_path: str):
    """Process uploaded file asynchronously"""
    if file_path.lower().endswith('.pdf'):
        try:
            description = extract_pdf_text(file_path)
            filename = os.path.basename(file_path)
            relative_path = os.path.relpath(file_path, app.config['UPLOAD_FOLDER'])
            with app.app_context():
                add_document(filename, relative_path, description)
        except Exception as e:
            print(f"Error processing PDF {file_path}: {str(e)}")
def add_document(name, link, description):
    """Adds a document to the database."""
    if len(description) > 300:
        description = description[:300]
    
    new_doc = Document(name=name, link=link, description=description)
    db.session.add(new_doc)
    db.session.commit()
    return f"Document '{name}' added successfully."

def get_documents():
    """Retrieves all documents from the database."""
    documents = Document.query.all()
    return [{'id': doc.id, 'name': doc.name, 'link': doc.link, 'description': doc.description} for doc in documents]

def delete_document(doc_id):
    """Deletes a document from the database by ID."""
    doc_to_delete = Document.query.get(doc_id)
    if doc_to_delete:
        db.session.delete(doc_to_delete)
        db.session.commit()
        return f"Document with ID {doc_id} deleted successfully."
    return f"Document with ID {doc_id} not found."

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(file_path):
    mime = magic.Magic(mime=True)
    return mime.from_file(file_path)

def extract_pdf_text(file_path, max_chars=300):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text[:max_chars]
    except:
        return ""


@app.route('/')
def index():
    return render_template('index.html')
@app.route('/admin')
def adminPage():
    # delete_all_documents()
    return render_template('admin.html')

@app.route('/documents', methods=['GET'])
def display_documents():
    """Renders an HTML table with the documents."""
    documents = Document.query.all()
    data = [
        {
            'id': doc.id,
            'name': doc.name,
            'link': doc.link,
            'description': doc.description
        } for doc in documents
    ]
    return render_template('data.html', documents=data)

@app.route('/api/content')
def get_content():
    path = request.args.get('path', '')
    
    # Simply use the upload folder as base
    full_path = app.config['UPLOAD_FOLDER']
    if path:  # If path is provided, join it with upload folder
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    folders = []
    files = []

    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        relative_path = os.path.relpath(item_path, app.config['UPLOAD_FOLDER'])
        
        if os.path.isdir(item_path):
            folders.append({
                'id': relative_path,
                'name': item,
                'type': 'folder'
            })
        else:
            files.append({
                'id': relative_path,
                'name': item,
                'type': item.split('.')[-1].lower()
            })

    return jsonify({
        'folders': sorted(folders, key=lambda x: x['name'].lower()),
        'files': sorted(files, key=lambda x: x['name'].lower())
    })

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    path = request.form.get('path', '')
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], path) if path else app.config['UPLOAD_FOLDER']

    if not os.path.exists(upload_path):
        os.makedirs(upload_path)

    uploaded_files = request.files.getlist('files[]')
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_path, filename)
            
            # Stream the file upload
            stream_file_upload(file, file_path)
            
            # Process the file asynchronously
            executor.submit(process_uploaded_file, file_path)

    return jsonify({'message': 'Files uploaded successfully'})

@app.route('/api/upload-folder', methods=['POST'])
def upload_folder():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    base_path = request.form.get('path', '')
    base_upload_path = os.path.join(app.config['UPLOAD_FOLDER'], base_path) if base_path else app.config['UPLOAD_FOLDER']

    files = request.files.getlist('files[]')
    paths = request.form.getlist('paths[]')

    for file, relative_path in zip(files, paths):
        if file and allowed_file(file.filename):
            full_path = os.path.join(base_upload_path, os.path.dirname(relative_path))
            os.makedirs(full_path, exist_ok=True)
            
            file_path = os.path.join(full_path, secure_filename(os.path.basename(relative_path)))
            
            # Stream the file upload
            stream_file_upload(file, file_path)
            
            # Process the file asynchronously
            executor.submit(process_uploaded_file, file_path)

    return jsonify({'message': 'Folder uploaded successfully'})

@app.route('/api/folder', methods=['POST'])
def create_folder():
    data = request.json
    folder_name = secure_filename(data['name'])
    path = data.get('path', '')
    
    new_folder_path = app.config['UPLOAD_FOLDER']
    if path:  # If path provided, append to uploads folder
        new_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    new_folder_path = os.path.join(new_folder_path, folder_name)
    
    if os.path.exists(new_folder_path):
        return jsonify({'error': 'Folder already exists'}), 400
    
    os.makedirs(new_folder_path)
    return jsonify({'message': 'Folder created successfully'})

@app.route('/api/rename', methods=['PUT'])
def rename_item():
    try:
        # Get and validate input data
        data = request.json
        if not all(key in data for key in ['id', 'newName', 'type']):
            return jsonify({'error': 'Missing required fields'}), 400
            
        item_id = data['id'].lstrip('/')  # Remove leading slash if present
        new_name = secure_filename(data['newName'])  # Sanitize the new name
        item_type = data['type']
        
        # Validate new name
        if not new_name:
            return jsonify({'error': 'Invalid new name'}), 400
            
        # Prevent directory traversal attempts
        if '..' in item_id:
            return jsonify({'error': 'Invalid path'}), 400
            
        # Compute the full old and new paths
        old_path = os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], item_id))
        parent_dir = os.path.dirname(old_path)
        new_path = os.path.normpath(os.path.join(parent_dir, new_name))
        
        # Log the paths for debugging
        print(f"Received item_id: {item_id}")
        print(f"Item type: {item_type}")
        print(f"Computed Old Path: {old_path}")
        print(f"Computed New Path: {new_path}")
        
        # Additional validations
        if not os.path.exists(old_path):
            return jsonify({'error': f'Original item not found: {old_path}'}), 404
            
        if os.path.exists(new_path):
            return jsonify({'error': f'An item with name {new_name} already exists in this location'}), 400
            
        # Ensure the parent directory exists and is writable
        if not os.path.exists(parent_dir):
            return jsonify({'error': 'Parent directory not found'}), 404
            
        if not os.access(parent_dir, os.W_OK):
            return jsonify({'error': 'Permission denied'}), 403
            
        # Perform the rename operation
        os.rename(old_path, new_path)
        print(f"Successfully renamed: {old_path} -> {new_path}")
        
        # Update database records if it's a file
        if item_type == 'file':
            # Get the relative path for database storage
            new_relative_path = os.path.relpath(new_path, app.config['UPLOAD_FOLDER'])
            old_relative_path = os.path.relpath(old_path, app.config['UPLOAD_FOLDER'])
            
            # Update the document record
            document = Document.query.filter_by(link=old_relative_path).first()
            if document:
                document.name = new_name
                document.link = new_relative_path
                db.session.commit()
                print(f"Database updated for document: {document.name}, {document.link}")
        
        # If it's a folder, update all affected document records
        elif item_type == 'folder':
            old_relative_path = os.path.relpath(old_path, app.config['UPLOAD_FOLDER'])
            new_relative_path = os.path.relpath(new_path, app.config['UPLOAD_FOLDER'])
            
            # Find all documents that have links starting with the old path
            documents = Document.query.filter(
                Document.link.like(f"{old_relative_path}%")
            ).all()
            
            # Update each document's link to use the new path
            for doc in documents:
                doc.link = doc.link.replace(old_relative_path, new_relative_path, 1)
                print(f"Updated document link: {doc.link}")
            
            if documents:
                db.session.commit()
                print(f"Updated {len(documents)} document records for folder rename")
        
        return jsonify({
            'message': 'Rename successful',
            'new_path': os.path.relpath(new_path, app.config['UPLOAD_FOLDER'])
        })
        
    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Database error during rename: {str(e)}")
        return jsonify({'error': 'Database error occurred'}), 500
        
    except OSError as e:
        print(f"OS error during rename: {str(e)}")
        return jsonify({'error': f'File system error: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Unexpected error during rename: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500



@app.route('/api/delete', methods=['DELETE'])
def delete_item():
    try:
        data = request.json
        item_id = data['id']
        item_type = data['type']
        
        item_path = os.path.join(app.config['UPLOAD_FOLDER'], item_id)
        
        if not os.path.exists(item_path):
            return jsonify({'error': 'Item not found'}), 404
        
        # Handle folder deletion
        if os.path.isdir(item_path):
            # First, get all files in the folder and subfolders before deletion
            files_to_delete = []
            for root, _, files in os.walk(item_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, app.config['UPLOAD_FOLDER'])
                    files_to_delete.append(relative_path)
            
            # Delete the folder and its contents
            shutil.rmtree(item_path)
            
            # Delete all related documents from database
            for file_path in files_to_delete:
                document = Document.query.filter_by(link=file_path).first()
                if document:
                    db.session.delete(document)
        
        # Handle single file deletion
        else:
            # Delete the file first
            os.remove(item_path)
            
            # Then remove from database
            document = Document.query.filter_by(link=item_id).first()
            if document:
                db.session.delete(document)
        
        # Commit all database changes
        db.session.commit()
        return jsonify({'message': 'Item deleted successfully'})
        
    except SQLAlchemyError as e:
        db.session.rollback()
        print(f"Database error during deletion: {str(e)}")
        return jsonify({'error': 'Database error occurred'}), 500
        
    except OSError as e:
        print(f"OS error during deletion: {str(e)}")
        return jsonify({'error': f'File system error: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Unexpected error during deletion: {str(e)}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/api/file/<path:file_id>')
def get_file(file_id):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    def generate():
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk

    return Response(
        generate(),
        mimetype=magic.Magic(mime=True).from_file(file_path),
        headers={
            "Content-Disposition": f"attachment; filename={os.path.basename(file_path)}",
            "Content-Length": str(os.path.getsize(file_path))
        }
    )


# @app.route('/api/delete-all-documents', methods=['DELETE'])
def delete_all_documents():
    """
    Deletes all records from the Documents table in the database.
    """
    try:
        # Delete all records in the Documents table
        num_deleted = Document.query.delete()
        db.session.commit()
        
        return jsonify({
            'message': f"All documents deleted successfully. Total records removed: {num_deleted}"
        }), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({'error': f"Database error occurred: {str(e)}"}), 500
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500







from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
os.environ["GROQ_API_KEY"] = "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"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
sql_db = SQLDatabase.from_uri(f"sqlite:///{os.path.join(BASE_DIR, "metadata.db")}")  # Update to your database path
agent_executor = create_sql_agent(llm, db=sql_db, agent_type="openai-tools", verbose=True)

@app.route('/quick_replies', methods=['POST'])
def quick_replies():
    data = request.get_json()
    session_history = data.get('session_history')

    if not session_history or not isinstance(session_history, list):
        return jsonify({"error": "Invalid session history provided"}), 400

    session_context = "\n".join(session_history)

    prompt_template = """
    Based on the following conversation history, suggest a list of quick replies (questions or actions) that would help continue the conversation effectively. 
    Provide the quick replies as a comma-separated list with no additional explanation.

    Conversation History:
    {session_history}

    Response:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["session_history"])
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    quick_replies = chain.invoke({"session_history": session_context})
    quick_replies_list = [reply.strip() for reply in quick_replies.split(',') if reply.strip()]

    return jsonify({"quick_replies": quick_replies_list})

def get_similar_documents(query):
    print(f"Original search query: '{query}'")
    
    # Break down query and clean terms
    terms = [term.lower() for term in query.split() if term]
    print(f"Search terms: {terms}")
    
    try:
        # Build OR conditions for each term
        conditions = []
        for term in terms:
            conditions.append(
                db.or_(
                    Document.name.ilike(f'%{term}%'),
                    Document.description.ilike(f'%{term}%'),
                    Document.link.ilike(f'%{term}%')
                )
            )
        
        # Combine with OR instead of AND
        documents = Document.query.filter(
            db.or_(*conditions)
        ).limit(20).all()
        
        print(f"\nQuery executed. Found {len(documents)} results")
        
        # Print found documents for debugging
        for doc in documents:
            print(f"\nMatch found:")
            print(f"Name: {doc.name}")
            print(f"Link: {doc.link}")
            print(f"Description preview: {doc.description[:100]}")
        
        results = []
        for doc in documents:
            document = {
                "metadata": doc.link,
                "content_preview": f"Name: {doc.name}\nDescription: {doc.description}"
            }
            results.append(document)
        return results
        
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return []
    
def generate_response(query):
    context = get_similar_documents(query)
    context = json.dumps(context, indent=2)
    prompt_template = """
You are an intelligent assistant designed to assist users in finding past paper links based on their queries. Your task is to:

- Analyze the provided document objects containing all data (metadata and content) for each paper.

documents: {context}

- Use the metadata and the first 200 characters of the content to understand the document’s subject, type, and relevant details.
- Analyze the user query to understand the specific past paper being requested (e.g., subject, year, type).
- Compare the user query against the metadata and content previews within the document objects provided.
- Identify the document that best matches the query.
- Provide the document link and a brief explanation of the document’s relevance based on its metadata and content.

user question:{question}

**Important Instructions**:
1. Prepend all document links with `C:/Users/Abbas/Desktop/pp project test/` before including them in the output.
2. Only include document links and summaries explicitly mentioned in the provided metadata and content.
3. Do not extrapolate or add speculative information, such as "[And so on]."
4. When a user greets (e.g., "hello," "hi"), respond warmly and politely, such as:  
   - "Hello! How can I assist you today?"  
   - "Hi there! Let me know what you're looking for, and I'll help you find it."
5. Ensure the output format follows the example provided below exactly.

**Output Format**:

Here's everything related to [user's query or topic]:

Below are the relevant documents, along with their links and a brief description:

 <div class="container mt-5">
   <div 
    class="card mb-3 w-100" 
    style="border: 2px solid; 
           border-image-slice: 1; 
           border-width: 2px; 
           border-image-source: linear-gradient(135deg, #6a0dad, #ff00ff); 
           border-radius: 50px;"
   >
    <div class="d-flex align-items-center p-4">
     <img alt="PDF icon with text 'PDF' in the center" class="rounded-circle" height="50" src="https://storage.googleapis.com/a1aa/image/IcdeS20aCywfEUzkV2OHvrSYYhR32htvrbDDjfvCaf2q6fBfE.jpg" width="50"/>
     <div class="ml-4">
      <b class="h5 text-dark">
       Document Title 1 (make some title based on pdf name)
      </b>
      <a class="d-block mt-2 text-danger" href="C:/Users/Abbas/Desktop/pp project test/uploads/[here you put relative path given in meta data]" target="_blank">
       <i class="fas fa-file-pdf pr-2">
       </i>
       Link to Document 1
      </a>
      <p class="text-muted mt-2">
       <i>
        Overview:
       </i>
       [Provide document content summary].
      </p>
     </div>
    </div>
   </div>
  </div>

[Repeat as necessary for other documents.]

If no match is found:

"I couldn't find a document matching your query. Please provide more details, such as the subject, year, or type of paper you're looking for."
"""



    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    output_parser = StrOutputParser()

    chain = PROMPT | llm | output_parser

    response = chain.invoke({"context": context, "question": query})
    formatted_response = format_response(response)
    print(response)
    return response

def format_response(response):
    formatted_response = "Here are the results for your query:\n\n"

    if "I couldn't find a document matching your query" in response:
        formatted_response += f"Sorry, no relevant documents were found based on your query.\n"
    else:
        formatted_response += "Below are the relevant documents with brief descriptions:\n\n"
        
        docs = response.split("\n\n")

        for idx, doc in enumerate(docs, 1):
            formatted_response += f"Document {idx}:\n{doc}\n\n"

    return formatted_response

@app.route('/query_paper', methods=['POST'])
def query_paper():
    data = request.get_json()
    query = data.get('query')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    response = generate_response(query)
    return jsonify({"response": response})



if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)