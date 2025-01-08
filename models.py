# models.py

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False,index=True)
    link = db.Column(db.String(200), nullable=False)
    description = db.Column(db.String(200), nullable=False,index=True)

    def __repr__(self):
        return f'<Document {self.name}>'

class DocumentEmbedding(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('document.id'), nullable=False)
    embedding = db.Column(db.PickleType, nullable=False)  # Use PickleType for storing arrays or lists
    
    document = db.relationship('Document', backref=db.backref('embedding', uselist=False))  # One-to-one relationship

    def __repr__(self):
        return f'<DocumentEmbedding for Document ID {self.document_id}>'
