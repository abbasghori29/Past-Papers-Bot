# AI-Powered Document Management System

## Overview
An intelligent document management system built with Flask that combines secure file handling with AI-powered search and chat capabilities. The system features semantic document search, user authentication, and an AI assistant for document queries.

## Features

### Core Functionality
- Document upload and management
- Semantic search using AI embeddings
- Intelligent chat interface for document queries
- User authentication and authorization
- File organization system

### Document Management
- Single and bulk file uploads
- Folder structure management
- File renaming and deletion
- Support for multiple file formats (PDF, DOC, DOCX, TXT, images)
- Secure file handling

### AI Capabilities
- Semantic document search using FAISS
- Google AI embeddings for document representation
- LLM-powered chat interface using Groq
- Context-aware responses
- Quick reply suggestions

## Technical Stack

### Backend
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **Flask-Login**: Authentication management
- **Flask-Migrate**: Database migrations

### AI & ML
- **FAISS**: Vector similarity search
- **LangChain**: LLM integration
- **Google Generative AI**: Document embeddings
- **Groq**: LLM provider (llama-3.3-70b-versatile model)

### File Processing
- **PyPDF2**: PDF text extraction
- **python-magic**: File type detection
- **Werkzeug**: File handling utilities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

5. Initialize the database:
```bash
flask db upgrade
```

## Configuration

Create a `config.py` file with the following settings:
```python
class Config:
    SECRET_KEY = 'your-secret-key'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///your-database.db'
    UPLOAD_FOLDER = 'uploads'
    MAX_CONTENT_LENGTH = None  # Set max file size limit if needed
```

## Usage

1. Start the development server: python app.py

2. Access the application:
- Open `http://localhost:5000` in your browser
- Register a new account or login
- Start uploading and managing documents
- Use the search functionality to find documents
- Interact with the AI assistant for document queries

## API Endpoints

### Authentication
- `POST /login`: User login
- `POST /register`: User registration
- `GET /logout`: User logout

### Document Management
- `GET /api/content`: List files and folders
- `POST /api/upload`: Upload files
- `POST /api/upload-folder`: Upload folders
- `POST /api/folder`: Create new folder
- `PUT /api/rename-file`: Rename file
- `PUT /api/rename-folder`: Rename folder
- `DELETE /api/delete-file`: Delete file
- `DELETE /api/delete-folder`: Delete folder

### AI Features
- `POST /query_paper`: Query documents using AI
- `POST /quick_replies`: Get AI-suggested quick replies

## Security Features
- Secure password hashing
- Session token management
- File type validation
- Secure filename handling
- Admin role authorization

## Database Schema

### Users
- id: Primary key
- username: Username
- email: Email address
- password_hash: Hashed password
- is_admin: Admin status
- session_token: Session management
- session_expiry: Token expiration

### Documents
- id: Primary key
- name: Document name
- link: File path
- description: Document description

### DocumentEmbeddings
- id: Primary key
- document_id: Foreign key to Documents
- embedding: Vector representation

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

