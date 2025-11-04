from flask_sqlalchemy import SQLAlchemy
from flask import current_app
from .models import db
import os

def init_db(app):
    """Initialize the database with the Flask app."""
    db.init_app(app)
    
    with app.app_context():
        # Create tables if they don't exist
        db.create_all()
        
        # Create uploads directory if it doesn't exist
        upload_folder = os.path.join(app.root_path, '..', app.config['UPLOAD_FOLDER'])
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

def get_db_session():
    """Get a database session."""
    return db.session

def add_to_db(instance):
    """Add an instance to the database session and commit."""
    try:
        db.session.add(instance)
        db.session.commit()
        return True, "Success"
    except Exception as e:
        db.session.rollback()
        return False, str(e)

def delete_from_db(instance):
    """Delete an instance from the database."""
    try:
        db.session.delete(instance)
        db.session.commit()
        return True, "Success"
    except Exception as e:
        db.session.rollback()
        return False, str(e)

def get_user_by_id(user_id):
    """Get a user by their ID."""
    from .models import User
    return User.query.get(user_id)

def get_user_by_username(username):
    """Get a user by their username."""
    from .models import User
    return User.query.filter_by(username=username).first()

def get_user_by_email(email):
    """Get a user by their email."""
    from .models import User
    return User.query.filter_by(email=email).first()

def get_user_preferences(user_id):
    """Get study preferences for a user."""
    from .models import StudyPreference
    return StudyPreference.query.filter_by(user_id=user_id).first()

def update_user_preferences(user_id, **kwargs):
    """Update user's study preferences."""
    from .models import StudyPreference
    prefs = StudyPreference.query.filter_by(user_id=user_id).first()
    
    if not prefs:
        prefs = StudyPreference(user_id=user_id)
        db.session.add(prefs)
    
    for key, value in kwargs.items():
        if hasattr(prefs, key):
            setattr(prefs, key, value)
    
    try:
        db.session.commit()
        return True, "Preferences updated successfully"
    except Exception as e:
        db.session.rollback()
        return False, str(e)

def get_pending_requests(user_id):
    """Get all pending buddy requests for a user."""
    from .models import BuddyRequest, User
    return BuddyRequest.query.filter(
        BuddyRequest.receiver_id == user_id,
        BuddyRequest.status == 'pending'
    ).join(User, BuddyRequest.sender_id == User.id).all()

def get_sent_requests(user_id):
    """Get all buddy requests sent by a user."""
    from .models import BuddyRequest, User
    return BuddyRequest.query.filter(
        BuddyRequest.sender_id == user_id
    ).join(User, BuddyRequest.receiver_id == User.id).all()

def get_buddies(user_id):
    """Get all accepted buddy connections for a user."""
    from .models import BuddyRequest, User
    # Get users who have accepted requests from this user
    sent = db.session.query(User).join(
        BuddyRequest,
        and_(
            BuddyRequest.receiver_id == User.id,
            BuddyRequest.sender_id == user_id,
            BuddyRequest.status == 'accepted'
        )
    ).all()
    
    # Get users who have sent accepted requests to this user
    received = db.session.query(User).join(
        BuddyRequest,
        and_(
            BuddyRequest.sender_id == User.id,
            BuddyRequest.receiver_id == user_id,
            BuddyRequest.status == 'accepted'
        )
    ).all()
    
    # Combine and remove duplicates
    return list({user.id: user for user in sent + received}.values())

def search_users(query, current_user_id):
    """Search for users by username, email, or name."""
    from .models import User
    search = f"%{query}%"
    return User.query.filter(
        and_(
            User.id != current_user_id,
            or_(
                User.username.ilike(search),
                User.email.ilike(search),
                User.full_name.ilike(search)
            )
        )
    ).all()
