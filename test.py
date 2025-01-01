# test.py

from app4 import app
from functions import add_document, get_documents

with app.app_context():
    # Add a document
    try:
        print(add_document("Sample Document", "/path/to/document", "This is a sample description."))
    except ValueError as e:
        print(f"Error: {e}")

    # Retrieve documents
    documents = get_documents()
    print("Documents in the database:")
    for doc in documents:
        print(doc)
