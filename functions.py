# functions.py

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
