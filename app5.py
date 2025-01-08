from flask import Flask, render_template, request, jsonify, send_file, Response
from config import Config
from models import db, Document, DocumentEmbedding
from werkzeug.utils import secure_filename
import os
import shutil
import magic
import PyPDF2
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.datastructures import FileStorage
import hashlib
from concurrent.futures import ThreadPoolExecutor
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.preprocessing import normalize
import re
from langchain_openai import ChatOpenAI


app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
CHUNK_SIZE = 50 * 1024 * 1024  # 50 MB chunks for streaming
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = None

# Thread pool for handling concurrent uploads
executor = ThreadPoolExecutor(max_workers=4)
upload_tasks = {}

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

os.environ["GOOGLE_API_KEY"] = "AIzaSyCPmn7EE7hY2-hbhrvx3c17tVobQBh5_Gk"
# os.environ["OPENAI_API_KEY"] =""

llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

@app.route('/')
def index():
    # delete_all_documents()
    return render_template('index.html')
@app.route('/admin')
def adminPage():
    # delete_all_documents()
    # generate_all_document_embeddings()
    return render_template('admin.html')

@app.route('/documents', methods=['GET'])
def display_documents():
    documents = Document.query.all()
    embs = DocumentEmbedding.query.all()  
    for emb in embs:
        print(f"ID: {emb.id}, Document ID: {emb.document_id}, Embedding: {emb.embedding}")    
        data = [
        {
            'id': doc.id,
            'name': doc.name,
            'link': doc.link,
            'description': doc.description
        } for doc in documents
    ]
    return render_template('data.html', documents=data)

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
    
    # Update embedding after adding the document
    update_embedding_on_add(
        doc_id=new_doc.id,
        name=new_doc.name,
        description=new_doc.description,
        link=new_doc.link
    )
    print(f"Embedding added for document ID {new_doc.id}")
    
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
        update_embedding_on_delete(doc_id)

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



@app.route('/api/content')
def get_content():
    path = request.args.get('path', '')
    
    full_path = app.config['UPLOAD_FOLDER']
    if path: 
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
    if path:  
        new_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    new_folder_path = os.path.join(new_folder_path, folder_name)
    
    if os.path.exists(new_folder_path):
        return jsonify({'error': 'Folder already exists'}), 400
    
    os.makedirs(new_folder_path)
    return jsonify({'message': 'Folder created successfully'})



def get_absolute_system_path():
    """Get the absolute system path of the upload folder"""
    # In development
    if app.debug:
        return os.path.abspath(os.path.join(os.path.expanduser('~'), 'Desktop', 'pp project test', 'uploads'))
    # In production
    return os.path.abspath(app.config['UPLOAD_FOLDER'])

# Helper function to normalize paths
def normalize_path(path):
    """Normalize path to use forward slashes and remove leading/trailing slashes"""
    return os.path.normpath(path.replace('\\', '/')).strip('/')

@app.route('/api/rename-file', methods=['PUT'])
def rename_file():
    try:
        data = request.json
        app.logger.info(f"Request data: {data}")

        # Validate request data
        if not all(key in data for key in ['fileId', 'newName', 'currentPath']):
            return jsonify({'error': 'Missing required fields'}), 400

        # Get absolute base path
        base_path = get_absolute_system_path()
        app.logger.info(f"Base system path: {base_path}")

        # Normalize frontend-provided paths
        relative_file_path = normalize_path(data['fileId'])
        relative_current_path = normalize_path(data['currentPath'])
        new_name = secure_filename(data['newName'])

        # Construct absolute paths
        absolute_file_path = os.path.join(base_path, relative_file_path)
        absolute_current_path = os.path.join(base_path, relative_current_path)

        app.logger.info(f"Normalized paths - file: {relative_file_path}, current: {relative_current_path}")

        # Query database for the document
        document = Document.query.filter_by(link=normalize_path(relative_file_path)).first()

        if not document:
            constructed_path = f"{relative_current_path}/{os.path.basename(relative_file_path)}"
            app.logger.info(f"Trying constructed path: {constructed_path}")
            document = Document.query.filter_by(link=normalize_path(constructed_path)).first()

        if not document:
            app.logger.error(f"Document not found in database. Tried paths: {relative_file_path}, {constructed_path}")
            return jsonify({'error': 'File not found'}), 404

        # Construct new absolute path
        old_abs_path = os.path.join(base_path, document.link)
        new_abs_path = os.path.join(base_path, relative_current_path, new_name)

        # Normalize paths for operations
        old_abs_path = os.path.normpath(old_abs_path)
        new_abs_path = os.path.normpath(new_abs_path)

        app.logger.info(f"Absolute paths - old: {old_abs_path}, new: {new_abs_path}")

        if not os.path.exists(old_abs_path):
            app.logger.error(f"Source file not found at: {old_abs_path}")
            return jsonify({'error': 'Source file not found'}), 404

        if os.path.exists(new_abs_path):
            app.logger.error(f"A file with this name already exists: {new_abs_path}")
            return jsonify({'error': 'A file with this name already exists'}), 400

        # Perform rename operation
        os.rename(old_abs_path, new_abs_path)

        # Update database
        new_relative_path = normalize_path(os.path.join(relative_current_path, new_name))
        document.name = new_name
        document.link = new_relative_path
        db.session.commit()

        app.logger.info(f"Updated document in DB. New link: {document.link}")

        update_embedding_on_rename(
            doc_id=document.id,
            new_name=document.name,
            new_description=document.description or '',
            new_link=document.link
        )

        return jsonify({'message': 'File renamed successfully'})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in rename_file: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/rename-folder', methods=['PUT'])
def rename_folder():
    try:
        data = request.json
        app.logger.info(f"Request data: {data}")

        if not all(key in data for key in ['folderPath', 'newName', 'currentPath']):
            return jsonify({'error': 'Missing required fields'}), 400

        # Get absolute base path
        base_path = get_absolute_system_path()
        app.logger.info(f"Base system path: {base_path}")

        # Normalize frontend-provided paths
        relative_folder_path = normalize_path(data['folderPath'])
        new_name = secure_filename(data['newName'])

        # Construct absolute paths
        old_abs_path = os.path.join(base_path, relative_folder_path)
        parent_abs_path = os.path.dirname(old_abs_path)
        new_abs_path = os.path.join(parent_abs_path, new_name)

        # Normalize paths for logging and operations
        old_abs_path = os.path.normpath(old_abs_path)
        new_abs_path = os.path.normpath(new_abs_path)

        app.logger.info(f"Absolute paths - old: {old_abs_path}, new: {new_abs_path}")

        if not os.path.exists(old_abs_path):
            app.logger.error(f"Folder not found: {old_abs_path}")
            return jsonify({'error': 'Folder not found'}), 404

        if os.path.exists(new_abs_path):
            app.logger.error(f"A folder with this name already exists: {new_abs_path}")
            return jsonify({'error': 'A folder with this name already exists'}), 400

        # Perform rename operation
        os.rename(old_abs_path, new_abs_path)

        # Update database paths
        old_relative_path = normalize_path(relative_folder_path)
        new_relative_path = normalize_path(os.path.join(os.path.dirname(relative_folder_path), new_name))

        documents = Document.query.filter(
            Document.link.like(f"{old_relative_path}%")
        ).all()

        for doc in documents:
            doc.link = doc.link.replace(old_relative_path, new_relative_path, 1)
            update_embedding_on_rename(
                doc_id=doc.id,
                new_name=doc.name,
                new_description=doc.description or '',
                new_link=doc.link
            )
            app.logger.info(f"Updated document: {doc.link}")

        db.session.commit()

        return jsonify({'message': 'Folder renamed successfully'})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in rename_folder: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    




@app.route('/api/delete-file', methods=['DELETE'])
def delete_file():
    try:
        data = request.json
        app.logger.info(f"Request data: {data}")

        if not all(key in data for key in ['fileId', 'currentPath']):
            return jsonify({'error': 'Missing required fields'}), 400

        # Get absolute base path
        base_path = get_absolute_system_path()
        app.logger.info(f"Base system path: {base_path}")

        # Normalize paths
        relative_file_path = normalize_path(data['fileId'])
        file_path = os.path.join(base_path, relative_file_path)
        file_path = os.path.normpath(file_path)

        app.logger.info(f"Attempting to delete file: {file_path}")

        if not os.path.exists(file_path):
            app.logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404

        if os.path.isdir(file_path):
            app.logger.error(f"Path is a directory, not a file: {file_path}")
            return jsonify({'error': 'Path is a directory, use delete-folder endpoint'}), 400

        # Find the document
        relative_path = normalize_path(os.path.relpath(file_path, base_path))
        document = Document.query.filter_by(link=relative_path).first()
        
        if document:
            # Store the document ID before deletion
            doc_id = document.id
            
            # Delete embedding first
            update_embedding_on_delete(doc_id)
            
            # Then delete the document
            db.session.delete(document)
            
        # Delete the physical file
        os.remove(file_path)
        
        # Commit all changes
        db.session.commit()
        
        return jsonify({'message': 'File deleted successfully'})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in delete_file: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete-folder', methods=['DELETE'])
def delete_folder():
    try:
        data = request.json
        app.logger.info(f"Request data: {data}")

        if not all(key in data for key in ['folderPath', 'currentPath']):
            return jsonify({'error': 'Missing required fields'}), 400

        # Get absolute base path
        base_path = get_absolute_system_path()
        app.logger.info(f"Base system path: {base_path}")

        # Normalize paths
        relative_folder_path = normalize_path(data['folderPath'])
        folder_path = os.path.join(base_path, relative_folder_path)
        folder_path = os.path.normpath(folder_path)

        app.logger.info(f"Attempting to delete folder: {folder_path}")

        if not os.path.exists(folder_path):
            app.logger.error(f"Folder not found: {folder_path}")
            return jsonify({'error': 'Folder not found'}), 404

        if not os.path.isdir(folder_path):
            app.logger.error(f"Path is a file, not a directory: {folder_path}")
            return jsonify({'error': 'Path is a file, use delete-file endpoint'}), 400

        # Get all files in the folder
        files_to_delete = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                relative_path = normalize_path(os.path.relpath(file_path, base_path))
                files_to_delete.append(relative_path)

        # Delete database records first
        for file_path in files_to_delete:
            document = Document.query.filter_by(link=file_path).first()
            if document:
                # Store the document ID before deletion
                doc_id = document.id
                
                # Delete embedding first
                update_embedding_on_delete(doc_id)
                
                # Then delete the document
                db.session.delete(document)

        # Delete the physical folder and its contents
        shutil.rmtree(folder_path)
        
        # Commit all changes
        db.session.commit()

        return jsonify({'message': 'Folder deleted successfully'})

    except Exception as e:
        db.session.rollback()
        app.logger.error(f"Error in delete_folder: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    

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


def delete_all_documents():
    """
    Deletes all records from the Document and DocumentEmbedding tables in the database.
    """
    try:
        embeddings_deleted = DocumentEmbedding.query.delete()        
        documents_deleted = Document.query.delete()        
        db.session.commit()
        
        return jsonify({
            'message': "All documents and their embeddings deleted successfully.",
            'documents_deleted': documents_deleted,
            'embeddings_deleted': embeddings_deleted
        }), 200
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({'error': f"Database error occurred: {str(e)}"}), 500
    except Exception as e:
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500


def preprocess_text(text):
    """
    Preprocess text while preserving academic document structure.
    """
    text = text.lower()
    text = text.replace('_', ' ').replace('-', ' ')
    text = re.sub(r'\.pdf$|\.doc[x]?$', '', text)
    text = re.sub(r'[^a-z0-9\s]+', ' ', text)
    text = ' '.join(text.split())
    return text

def normalize_embedding(embedding):
    """Normalize embeddings to unit vectors."""
    return embedding / np.linalg.norm(embedding)

def update_embedding_on_add(doc_id, name, description, link):
    """Add a new embedding to the database and FAISS index when a document is added."""
    combined_text = f"{name} {link}"
    new_embedding = np.array(embeddings_model.embed_query(combined_text)).astype('float32')
    new_embedding = normalize_embedding(new_embedding)

    doc_embedding = DocumentEmbedding(document_id=doc_id, embedding=new_embedding.tolist())

    try:
        # Save the embedding to the database
        db.session.add(doc_embedding)
        db.session.commit()

        # Add the embedding to the FAISS index
        global index, document_ids
        document_ids.append(doc_id)
        if index is None:
            index = faiss.IndexFlatL2(len(new_embedding)) 
        index.add(new_embedding.reshape(1, -1))
    except Exception as e:
        db.session.rollback()
        raise RuntimeError(f"Error processing embedding on add: {e}")
    finally:
        db.session.close()

def update_embedding_on_rename(doc_id, new_name, new_description, new_link):
    """Update the embedding in the database and FAISS index when a document is renamed."""
    combined_text = f"{new_name} {new_link}"
    updated_embedding = np.array(embeddings_model.embed_query(combined_text)).astype('float32')
    updated_embedding = normalize_embedding(updated_embedding)

    try:
        # Update the embedding in the database
        doc_embedding = db.session.query(DocumentEmbedding).filter_by(document_id=doc_id).first()
        if doc_embedding:
            doc_embedding.embedding = updated_embedding.tolist()
            db.session.commit()

        # Update the embedding in the FAISS index
        global index, document_ids
        if doc_id in document_ids:
            idx = document_ids.index(doc_id)
            index.reconstruct(idx)[:] = updated_embedding
    except Exception as e:
        db.session.rollback()
        raise RuntimeError(f"Error processing embedding on rename: {e}")
    finally:
        db.session.close()





def update_embedding_on_delete(doc_id):
    try:
        # Use a session with no autoflush to prevent premature database operations
        with db.session.no_autoflush:
            # Get the embedding record
            doc_embedding = db.session.query(DocumentEmbedding).filter_by(document_id=doc_id).first()
            if doc_embedding:
                # Delete the embedding first
                db.session.delete(doc_embedding)
                db.session.flush()  # Flush to ensure embedding is deleted
                
    except Exception as e:
        db.session.rollback()
        raise RuntimeError(f"Error processing embedding on delete: {e}")

# def update_embedding_on_delete(doc_id):
#     """Delete the embedding from the database and FAISS index when a document is deleted."""
#     global index, document_ids
#     try:
#         if doc_id in document_ids:
#             idx = document_ids.index(doc_id)
#             document_ids.pop(idx)

#             index.remove_ids(np.array([idx]))

#             doc_embedding = db.session.query(DocumentEmbedding).filter_by(document_id=doc_id).first()
#             if doc_embedding:
#                 db.session.delete(doc_embedding)
#                 db.session.commit()
#     except Exception as e:
#         db.session.rollback()
#         raise RuntimeError(f"Error processing embedding on delete: {e}")
#     finally:
#         db.session.close()



embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
index = None
document_ids = []
def initialize_faiss_index():
    """Initialize FAISS index for document embeddings."""
    global index, document_ids
    document_ids = []

    embeddings = DocumentEmbedding.query.all()
    if not embeddings:
        return

    embedding_dim = len(embeddings[0].embedding)
    if len(embeddings) > 10000:
        quantizer = faiss.IndexFlatL2(embedding_dim)
        nlist = max(10, len(embeddings) // 100)
        index = faiss.IndexIVFPQ(quantizer, embedding_dim, nlist, 8, 8)
        index.train(np.vstack([np.array(emb.embedding).astype('float32') for emb in embeddings]))
    else:
        index = faiss.IndexFlatIP(embedding_dim)  

    for emb in embeddings:
        document_ids.append(emb.document_id)
        index.add(np.array(emb.embedding).astype('float32').reshape(1, -1))

def get_similar_documents(query, top_k=20):
    """Fetch similar documents using semantic similarity."""
    global index, document_ids
    if index is None or index.ntotal == 0:
        print("FAISS index is empty. Please initialize it first.")
        return []

    processed_query = preprocess_text(query)
    query_embedding = np.array(embeddings_model.embed_query(processed_query)).astype('float32')
    query_embedding = normalize_embedding(query_embedding)

    distances, indices = index.search(query_embedding.reshape(1, -1), top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(document_ids):
            doc_id = document_ids[idx]
            doc = db.session.query(Document).filter_by(id=doc_id).first()
            if doc:
                score = 1.0 - distances[0][i]  # Convert distance to similarity score
                results.append({
                    "metadata": doc.link,
                    "content_preview": f"Name: {doc.name}\nDescription: {doc.description}",
                    "score": score
                })
    print(results)
    db.session.close()
    print(results)
    return sorted(results, key=lambda x: x['score'], reverse=True)
        



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

    # chain = prompt | llm | output_parser

    # quick_replies = chain.invoke({"session_history": session_context})
    # quick_replies_list = [reply.strip() for reply in quick_replies.split(',') if reply.strip()]

    # return jsonify({"quick_replies": quick_replies_list})




def generate_response(query,history):
    context = get_similar_documents(query)
    context = json.dumps(context, indent=2)
    prompt_template = """
You are an intelligent assistant designed to assist users in finding past paper links based on their queries. Your task is to:

- Analyze the provided document objects containing all data (metadata and content) for each paper.

documents: {context}

- Use the metadata (including document name, link, path) to understand the document's subject, type, and details.
- Analyze the user query to understand the specific past paper being requested.
- Compare the user query against the metadata and content previews.
- Identify the best matching document(s). If no exact match exists, find the most relevant alternatives.
- Provide document links and brief explanations of relevance.

user question: {question}

**Important Instructions**:
1. When constructing document links, use forward slashes (/) in paths:
   `file:///C:/Users/Abbas/Desktop/pp project test/uploads/[subject]/[year]/[paper type]/[filename].pdf`
2. Only include documents explicitly mentioned in the provided metadata/content.
3. Avoid speculative information or placeholders.
4. For greetings, respond naturally but briefly.
5. Follow the output format exactly.
6. Prioritize exact matches to user queries, then show relevant alternatives.
7. Maximum 5 results per response.

**Output Format Example**:

Here's everything related to Physics HL Paper 2:

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
      <img alt="PDF icon" class="rounded-circle" height="50" src="https://storage.googleapis.com/a1aa/image/IcdeS20aCywfEUzkV2OHvrSYYhR32htvrbDDjfvCaf2q6fBfE.jpg" width="50"/>
      <div class="ml-4">
        <b class="h5 text-dark">
          Physics HL Paper 2 - November 2015 (Spanish)
        </b>
        <a class="d-block mt-2 text-danger" href="file:///C:/Users/Abbas/Desktop/pp project test/uploads/Physics HL Papers/2015 November Examination Session/Spanish Papers/Physics_paper_2_HL_Spanish.pdf" target="_blank">
          <i class="fas fa-file-pdf pr-2"></i>
          Download Document
        </a>
        <p class="text-muted mt-2">
          <i>Overview:</i>
          Higher Level Physics Paper 2 examination from November 2015 session in Spanish.
        </p>
      </div>
    </div>
  </div>
</div>

see this history: {history} (last 3 conversation exchanges between assistant and user) Response: [User's next message]
Assistant Response: [Format response according to query and instructions above]
"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question","history"])
    output_parser = StrOutputParser()

    chain = PROMPT | llm | output_parser

    response = chain.invoke({"context": context, "question": query,"history":history})
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
    history=data.get('history')

    if not query:
        return jsonify({"error": "No query provided"}), 400

    response = generate_response(query,history)
    return jsonify({"response": response})


if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        initialize_faiss_index()
    app.run(debug=True)