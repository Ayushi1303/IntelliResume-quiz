from datetime import datetime
import re
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Index, func

db = SQLAlchemy()

class User(db.Model):
    """User model representing a registered user in the system.
    
    Attributes:
        username: Unique username for the user
        email: User's email address (must be unique)
        password_hash: Hashed password for security
        full_name: User's full name
        bio: Short biography or description
        created_at: Timestamp of account creation
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128), nullable=False)
    full_name = db.Column(db.String(100), nullable=True)
    bio = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)
    last_login = db.Column(db.DateTime, nullable=True)
    
    # Relationships
    preferences = db.relationship('StudyPreference', backref='user', uselist=False)
    sent_requests = db.relationship('BuddyRequest', 
                                  foreign_keys='BuddyRequest.sender_id',
                                  backref='sender', lazy='dynamic')
    received_requests = db.relationship('BuddyRequest',
                                      foreign_keys='BuddyRequest.receiver_id',
                                      backref='receiver', lazy='dynamic')
    
    def set_password(self, password: str) -> None:
        """Set user's password.
        
        Args:
            password: Plain text password to hash and store
        """
        if not password or len(password) < 8:
            raise ValueError("Password must be at least 8 characters long")
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password: str) -> bool:
        """Verify the provided password against the stored hash.
        
        Args:
            password: Plain text password to verify
            
        Returns:
            bool: True if password matches, False otherwise
        """
        return check_password_hash(self.password_hash, password)
        
    def update_last_login(self) -> None:
        """Update the last login timestamp to current time."""
        self.last_login = datetime.utcnow()
        db.session.commit()
        
    def to_dict(self) -> dict:
        """Convert user object to dictionary.
        
        Returns:
            dict: User data as a dictionary
        """
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None
        }
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
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, 
                         onupdate=datetime.utcnow)
    
    # Add a composite index on user_id and course_name
    __table_args__ = (
        Index('idx_user_course', 'user_id', 'course_name', unique=True),
    )
    
    def update(self, **kwargs) -> None:
        """Update study preferences with provided fields.
        
        Args:
            **kwargs: Fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert study preferences to dictionary.
        
        Returns:
            dict: Study preferences as a dictionary
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'course_name': self.course_name,
            'preferred_study_times': self.preferred_study_times,
            'study_style': self.study_style,
            'goals': self.goals,
            'availability': self.availability,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class BuddyRequest(db.Model):
    """Model representing a study buddy request between users.
    
    Attributes:
        sender_id: User ID of the requester
        receiver_id: User ID of the recipient
        status: Status of the request (pending/accepted/rejected)
        message: Optional message with the request
    """
    __tablename__ = 'buddy_requests'
    
    STATUS_CHOICES = ('pending', 'accepted', 'rejected')
    
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, 
                         db.ForeignKey('users.id', ondelete='CASCADE'), 
                         nullable=False, 
                         index=True)
    receiver_id = db.Column(db.Integer, 
                          db.ForeignKey('users.id', ondelete='CASCADE'), 
                          nullable=False,
                          index=True)
    status = db.Column(db.String(20), default='pending', nullable=False, index=True)
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, 
                          onupdate=datetime.utcnow)
    
    # Add a composite index for common query patterns
    __table_args__ = (
        Index('idx_sender_receiver', 'sender_id', 'receiver_id', unique=True),
        Index('idx_receiver_status', 'receiver_id', 'status'),
    )
    
    def accept(self) -> None:
        """Mark the request as accepted."""
        self.status = 'accepted'
        self.updated_at = datetime.utcnow()
        
    def reject(self) -> None:
        """Mark the request as rejected."""
        self.status = 'rejected'
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        """Convert buddy request to dictionary.
        
        Returns:
            dict: Request data as a dictionary
        """
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'status': self.status,
            'message': self.message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
