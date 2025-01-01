from models import db, Document

def add_document(name, link, description):
    """Adds a document to the database."""
    if len(description) > 200:
        raise ValueError("Description must be 200 characters or less.")
    
    new_doc = Document(name=name, link=link, description=description)
    db.session.add(new_doc)
    db.session.commit()
    return f"Document '{name}' added successfully."

def get_documents():
    """Retrieves all documents from the database."""
    documents = Document.query.all()
    return [{'id': doc.id, 'name': doc.name, 'link': doc.link, 'description': doc.description} for doc in documents]


from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import shutil
from pathlib import Path
import magic
import PyPDF2
import io

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_file_type(file_path):
    mime = magic.Magic(mime=True)
    return mime.from_file(file_path)

def extract_pdf_text(file_path, max_chars=150):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text[:max_chars]
    except:
        return ""

def add_document(name, link, description):
    """Adds a document to the database."""
    if len(description) > 200:
        raise ValueError("Description must be 200 characters or less.")
    
    # Your database logic here
    pass

def delete_document(doc_id):
    """Deletes a document from the database by ID."""
    # Your database logic here
    pass

# Routes
@app.route('/api/content')
def get_content():
    path = request.args.get('path', 'root')
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], path)
    
    if not os.path.exists(full_path):
        return jsonify({'error': 'Path not found'}), 404

    folders = []
    files = []

    for item in os.listdir(full_path):
        item_path = os.path.join(full_path, item)
        if os.path.isdir(item_path):
            folders.append({
                'id': item,
                'name': item,
                'type': 'folder'
            })
        else:
            files.append({
                'id': item,
                'name': item,
                'type': item.split('.')[-1].lower()
            })

    return jsonify({
        'folders': folders,
        'files': files
    })

@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    path = request.form.get('path', 'root')
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], path)

    if not os.path.exists(upload_path):
        os.makedirs(upload_path)

    uploaded_files = request.files.getlist('files[]')
    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_path, filename)
            file.save(file_path)

            # For PDFs, extract content and add to database
            if filename.lower().endswith('.pdf'):
                description = extract_pdf_text(file_path)
                add_document(filename, file_path, description)

    return jsonify({'message': 'Files uploaded successfully'})

@app.route('/api/folder', methods=['POST'])
def create_folder():
    data = request.json
    folder_name = secure_filename(data['name'])
    path = data['path']
    
    new_folder_path = os.path.join(app.config['UPLOAD_FOLDER'], path, folder_name)
    
    if os.path.exists(new_folder_path):
        return jsonify({'error': 'Folder already exists'}), 400
    
    os.makedirs(new_folder_path)
    return jsonify({'message': 'Folder created successfully'})

@app.route('/api/rename', methods=['PUT'])
def rename_item():
    data = request.json
    item_id = data['id']
    new_name = secure_filename(data['newName'])
    item_type = data['type']
    
    old_path = os.path.join(app.config['UPLOAD_FOLDER'], item_id)
    new_path = os.path.join(app.config['UPLOAD_FOLDER'], new_name)
    
    if os.path.exists(new_path):
        return jsonify({'error': 'Item with this name already exists'}), 400
    
    os.rename(old_path, new_path)
    return jsonify({'message': 'Item renamed successfully'})

@app.route('/api/delete', methods=['DELETE'])
def delete_item():
    data = request.json
    item_id = data['id']
    item_type = data['type']
    
    item_path = os.path.join(app.config['UPLOAD_FOLDER'], item_id)
    
    if not os.path.exists(item_path):
        return jsonify({'error': 'Item not found'}), 404
    
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)
    else:
        os.remove(item_path)
        # If it's a document, remove from database
        delete_document(item_id)
    
    return jsonify({'message': 'Item deleted successfully'})

@app.route('/api/file/<file_id>')
def get_file(file_id):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    return send_file(file_path)

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)