from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    full_name = db.Column(db.String(100))
    bio = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    preferences = db.relationship('StudyPreference', backref='user', uselist=False)
    sent_requests = db.relationship('BuddyRequest', 
                                  foreign_keys='BuddyRequest.sender_id',
                                  backref='sender', lazy='dynamic')
    received_requests = db.relationship('BuddyRequest',
                                      foreign_keys='BuddyRequest.receiver_id',
                                      backref='receiver', lazy='dynamic')
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class StudyPreference(db.Model):
    __tablename__ = 'study_preferences'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    course_name = db.Column(db.String(100), nullable=False)
    preferred_study_times = db.Column(db.String(200))
    study_style = db.Column(db.String(50))  # visual, auditory, hands-on, etc.
    goals = db.Column(db.Text)
    availability = db.Column(db.String(100))  # days and times available
    
    def to_dict(self):
        return {
            'course_name': self.course_name,
            'preferred_study_times': self.preferred_study_times,
            'study_style': self.study_style,
            'goals': self.goals,
            'availability': self.availability
        }

class BuddyRequest(db.Model):
    __tablename__ = 'buddy_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    receiver_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, accepted, rejected
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'sender': self.sender.username,
            'receiver': self.receiver.username,
            'status': self.status,
            'message': self.message,
            'created_at': self.created_at.isoformat()
        }
