# models.py

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash

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


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128))
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    last_login = db.Column(db.DateTime)
    session_token = db.Column(db.String(64), unique=True, nullable=True)
    session_expiry = db.Column(db.DateTime, nullable=True)
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'