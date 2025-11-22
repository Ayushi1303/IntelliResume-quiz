"""Test database setup and basic operations."""
import pytest
from study_buddy_project import create_app, db as _db
from study_buddy_project.models import User

def test_db_connection(app, db):
    """Test that the database connection works and tables are created."""
    with app.app_context():
        # Verify tables exist
        inspector = _db.inspect(_db.engine)
        assert 'users' in inspector.get_table_names()
        assert 'study_preferences' in inspector.get_table_names()
        assert 'buddy_requests' in inspector.get_table_names()

def test_create_user(app, db):
    """Test creating a user in the database."""
    with app.app_context():
        # Create a test user
        user = User(username='testuser', email='test@example.com')
        user.set_password('testpass123')
        
        # Add to database
        _db.session.add(user)
        _db.session.commit()
        
        # Verify user was created
        assert user.id is not None
        
        # Query the user
        queried_user = User.query.filter_by(username='testuser').first()
        assert queried_user is not None
        assert queried_user.email == 'test@example.com'
