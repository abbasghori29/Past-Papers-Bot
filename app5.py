from flask import Flask, render_template, request, jsonify, send_file, Response
from config import Config
from models import db, Document, DocumentEmbedding
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
from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import json
import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.preprocessing import normalize
from datetime import datetime
import re
from sqlalchemy.orm import sessionmaker


app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
CHUNK_SIZE = 50 * 1024 * 1024  # 1MB chunks for streaming
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


@app.route('/')
def index():
    delete_all_documents()

    return render_template('index.html')
@app.route('/admin')
def adminPage():
    # delete_all_documents()
    # generate_all_document_embeddings()
    return render_template('admin.html')

@app.route('/documents', methods=['GET'])
def display_documents():
    """Renders an HTML table with the documents."""
    documents = Document.query.all()
    embs = DocumentEmbedding.query.all()  # Retrieve all embeddings
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
            # Try using constructed path
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

        # Check file existence
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

        # Update embeddings or other dependent systems
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
                    update_embedding_on_delete(document.id)  # Call to update embeddings

        # Handle single file deletion
        else:
            # Delete the file first
            os.remove(item_path)
            
            # Then remove from database
            document = Document.query.filter_by(link=item_id).first()
            if document:
                db.session.delete(document)
                update_embedding_on_delete(document.id)  # Call to update embeddings
        
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
    Deletes all records from the Document and DocumentEmbedding tables in the database.
    """
    try:
        # Delete all records in the DocumentEmbedding table first
        embeddings_deleted = DocumentEmbedding.query.delete()
        
        # Delete all records in the Document table
        documents_deleted = Document.query.delete()
        
        # Commit the transaction
        db.session.commit()
        
        return jsonify({
            'message': "All documents and their embeddings deleted successfully.",
            'documents_deleted': documents_deleted,
            'embeddings_deleted': embeddings_deleted
        }), 200
    except SQLAlchemyError as e:
        # Rollback in case of database error
        db.session.rollback()
        return jsonify({'error': f"Database error occurred: {str(e)}"}), 500
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'error': f"An unexpected error occurred: {str(e)}"}), 500





os.environ["GOOGLE_API_KEY"]="AIzaSyCPmn7EE7hY2-hbhrvx3c17tVobQBh5_Gk"

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
index = None
document_ids = []



def initialize_faiss_index():
    global index, document_ids
    document_ids = []
    
    # Get all embeddings from database
    embeddings = DocumentEmbedding.query.all()
    if not embeddings:
        return
    
    # Initialize index with first embedding's dimension
    first_embedding = np.array(embeddings[0].embedding).astype('float32')
    index = faiss.IndexFlatL2(len(first_embedding))
    
    # Add all embeddings to index
    all_embeddings = []
    for emb in embeddings:
        document_ids.append(emb.document_id)
        all_embeddings.append(np.array(emb.embedding).astype('float32'))
    
    if all_embeddings:
        index.add(np.vstack(all_embeddings))

def update_embedding_on_add(doc_id, name, description, link):
    """Add a new embedding to the database and FAISS index when a document is added."""
    combined_text = f"{link} {name} {description} "
    new_embedding = embeddings_model.embed_query(combined_text)
    new_embedding = np.array(new_embedding).astype('float32')

    doc_embedding = DocumentEmbedding(document_id=doc_id, embedding=new_embedding.tolist())

    try:
        # Save the embedding to the database
        db.session.add(doc_embedding)
        db.session.commit()

        # Add the embedding to the FAISS index
        global index, document_ids
        document_ids.append(doc_id)
        if index is None:
            index = faiss.IndexFlatL2(len(new_embedding))  # Initialize index with the correct size
        index.add(new_embedding.reshape(1, -1))
    except Exception as e:
        db.session.rollback()
        raise RuntimeError(f"Error processing embedding on add: {e}")
    finally:
        db.session.close()


def update_embedding_on_rename(doc_id, new_name, new_description, new_link):
    """Update the embedding in the database and FAISS index when a document is renamed."""
    combined_text = f"{new_name} {new_description} {new_link}"
    updated_embedding = embeddings_model.embed_query(combined_text)
    updated_embedding = np.array(updated_embedding).astype('float32')

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
    """Delete the embedding from the database and FAISS index when a document is deleted."""
    global index, document_ids
    try:
        if doc_id in document_ids:
            idx = document_ids.index(doc_id)
            document_ids.pop(idx)

            # Remove the embedding from the FAISS index
            index.remove_ids(np.array([idx]))

            # Remove the embedding from the database
            doc_embedding = db.session.query(DocumentEmbedding).filter_by(document_id=doc_id).first()
            if doc_embedding:
                db.session.delete(doc_embedding)
                db.session.commit()
    except Exception as e:
        db.session.rollback()
        raise RuntimeError(f"Error processing embedding on delete: {e}")
    finally:
        db.session.close()




os.environ["GOOGLE_API_KEY"]="AIzaSyCPmn7EE7hY2-hbhrvx3c17tVobQBh5_Gk"

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
index = None
document_ids = []

def preprocess_text(text):
    """Preprocess text to improve embedding quality"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def extract_features(text):
    """Extract structured features from text"""
    features = {
        'year': None,
        'subject': None,
        'paper_type': None,
        'level': None
    }
    
    # Extract year (2000-2024)
    year_match = re.search(r'20[0-2][0-9]', text)
    if year_match:
        features['year'] = year_match.group()
    
    # Extract subject
    subjects = ['physics', 'chemistry', 'biology', 'math', 'mathematics']
    for subject in subjects:
        if subject in text.lower():
            features['subject'] = subject
            break
    
    # Extract level (HL/SL)
    if 'hl' in text.lower() or 'higher level' in text.lower():
        features['level'] = 'hl'
    elif 'sl' in text.lower() or 'standard level' in text.lower():
        features['level'] = 'sl'
    
    return features

def create_enhanced_embedding(text, features):
    """Create enhanced embedding with feature boosting"""
    base_embedding = np.array(embeddings_model.embed_query(text))
    
    # Create feature embedding
    feature_text = f"{features['year'] or ''} {features['subject'] or ''} {features['level'] or ''}"
    feature_embedding = np.array(embeddings_model.embed_query(feature_text))
    
    # Combine embeddings with weights
    combined_embedding = 0.7 * base_embedding + 0.3 * feature_embedding
    
    # Normalize the combined embedding
    return normalize(combined_embedding.reshape(1, -1))[0]

def initialize_faiss_index():
    global index, document_ids
    document_ids = []
    
    embeddings = DocumentEmbedding.query.all()
    if not embeddings:
        return
    
    first_embedding = np.array(embeddings[0].embedding).astype('float32')
    
    # Use IVFFlat index for better performance with exact search
    nlist = min(64, max(4, len(embeddings) // 10))  # number of clusters
    quantizer = faiss.IndexFlatL2(len(first_embedding))
    index = faiss.IndexIVFFlat(quantizer, len(first_embedding), nlist, faiss.METRIC_L2)
    
    # Train the index
    all_embeddings = []
    for emb in embeddings:
        document_ids.append(emb.document_id)
        all_embeddings.append(np.array(emb.embedding).astype('float32'))
    
    if all_embeddings:
        training_data = np.vstack(all_embeddings)
        index.train(training_data)
        index.add(training_data)

def get_similar_documents(query, top_k=20):
    """Enhanced similarity search"""
    global index, document_ids
    if index is None or index.ntotal == 0:
        print("FAISS index is empty. Please initialize it first.")
        return []

    # Preprocess query and extract features
    processed_query = preprocess_text(query)
    features = extract_features(query)
    
    # Create enhanced query embedding
    query_embedding = create_enhanced_embedding(processed_query, features)
    query_embedding = query_embedding.astype('float32').reshape(1, -1)

    # Perform search with exact KNN
    index.nprobe = min(64, index.nlist)  # Increase number of clusters to search
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    try:
        for i, idx in enumerate(indices[0]):
            if idx < len(document_ids) and idx >= 0:
                doc_id = document_ids[idx]
                doc = db.session.query(Document).filter_by(id=doc_id).first()
                
                if doc:
                    # Score boost based on feature matching
                    doc_features = extract_features(f"{doc.name} {doc.description}")
                    score = 1.0 - distances[0][i]  # Convert distance to similarity score
                    
                    # Boost score based on feature matches
                    if features['year'] and features['year'] == doc_features['year']:
                        score *= 1.2
                    if features['subject'] and features['subject'] == doc_features['subject']:
                        score *= 1.3
                    if features['level'] and features['level'] == doc_features['level']:
                        score *= 1.4
                    
                    results.append({
                        "metadata": doc.link,
                        "content_preview": f"Name: {doc.name}\nDescription: {doc.description}",
                        "score": score
                    })
        
        # Sort results by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
    except Exception as e:
        raise RuntimeError(f"Error fetching similar documents: {e}")
    finally:
        db.session.close()

    return results



os.environ["GROQ_API_KEY"] = "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"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
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




def generate_response(query,history):
    context = get_similar_documents(query)
    context = json.dumps(context, indent=2)
    prompt_template = """
You are an intelligent assistant designed to assist users in finding past paper links based on their queries. Your task is to:

- Analyze the provided document objects containing all data (metadata and content) for each paper.

documents: {context}

- Use the metadata (including document name, link, path, and any other relevant information) to understand the document’s subject, type, and relevant details.
- Analyze the user query to understand the specific past paper being requested (e.g., subject, year, type). If the query mentions a "marking scheme" or specific paper, prioritize both types in your results (i.e., past paper and marking scheme).
- Compare the user query against the full metadata (including the document name, link, and path) and the content previews within the document objects provided.
- Identify the document that best matches the query. If no exact match is found, prioritize the next most relevant document based on its subject, year, type, and metadata details (such as name, link, and path).
- Provide the document link and a brief explanation of the document’s relevance based on its metadata and content.

user question: {question}

**Important Instructions**:
1. Prepend all document links with `uploads/` before including them in the output.
2. Only include document links and summaries explicitly mentioned in the provided metadata and content.
3. Do not extrapolate or add speculative information, such as "[And so on]."
4. Only when a user greets (e.g., "hello," "hi"), respond warmly and politely, such as:  
   - "Hello! How can I assist you today?"  
   - "Hi there! Let me know what you're looking for, and I'll help you find it."
5. Ensure the output format follows the example provided below exactly.
6. Ensure that if the query specifies a specific paper (e.g., "Physics HL English 2020"), the results must prioritize that specific paper, even if no marking scheme is available. If no paper for that year is found, the assistant should try to return the closest match (e.g., other Physics HL English papers, or marking schemes).
7. Prepend all document links with uploads/ before including them in the output. If the metadata includes file:///C:/Users/Abbas/Desktop/pp project test, ensure it is appended correctly for all links.
8.Ensure all links are structured as downloadable links using the format: file:///C:/Users/Abbas/Desktop/pp project test/uploads/[relative path from metadata].


**Output Format**:

Here's everything related to [user's query or topic]:

Below are the relevant documents, along with their links and a brief description: [ensure that you give at least 4 and a maximum of 5 documents that best match with the user query]

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
      <a class="d-block mt-2 text-danger" href="file:///C:/Users/Abbas/Desktop/pp project test/uploads/[here you put relative path given in meta data] download" target="_blank">
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

see this history: {history} they are 10 latest conversationl exchange between you and user please tailor your response accordingly
If no exact match is found, the following documents are the closest matches based on the query's subject, year, type, and metadata details (e.g., name, link, path):

"I couldn't find an exact match for your query. However, I found these documents which might be relevant."

If no match is found:

"I couldn't find a document matching your query. Please provide more details, such as the subject, year, or type of paper you're looking for."
"""



    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question","history"])
    output_parser = StrOutputParser()

    chain = PROMPT | llm | output_parser

    response = chain.invoke({"context": context, "question": query,"history":history})
    formatted_response = format_response(response)
    # print(response)
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