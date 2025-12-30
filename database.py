from flask_sqlalchemy import SQLAlchemy
from datetime import datetime


db = SQLAlchemy()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    security_question = db.Column(db.String(255), nullable=False)
    security_answer = db.Column(db.String(255), nullable=False)

class ImageModel(db.Model):
    id = db.Column(db.String(50), primary_key=True)
    image_path = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class RecommendationResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    event = db.Column(db.String(50), nullable=False)
    outfit = db.Column(db.Text, nullable=False)
    scores = db.Column(db.Text, nullable=False)
    match_score = db.Column(db.Float, nullable=False)
    heatmap_paths = db.Column(db.Text, nullable=False)

class Saved(db.Model):
    __tablename__ = 'saved'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), nullable=False)
    event = db.Column(db.String(100), nullable=False)
    outfit = db.Column(db.JSON, nullable=False)
    clothes_ids = db.Column(db.JSON, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'event': self.event,
            'outfit': self.outfit
        }

class UploadedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())

class GeneratedOutfit(db.Model):
    __tablename__ = 'generated_outfits'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    outfit = db.Column(db.Text, nullable=False)  # store JSON array of image filenames
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, user_id, outfit):
        self.user_id = user_id
        self.outfit = outfit
