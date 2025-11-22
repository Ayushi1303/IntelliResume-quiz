"""Test configuration and fixtures for the application."""
import os
import sys
import tempfile
import pytest
from datetime import datetime, timedelta

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# Ensure project root is on sys.path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import the app and models after adding to path
from study_buddy_project import create_app, db as _db
from study_buddy_project.models import User, StudyPreference, BuddyRequest

# Test configuration
TEST_CONFIG = {
    'TESTING': True,
    'WTF_CSRF_ENABLED': False,
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'SECRET_KEY': 'test-secret-key',
    'PRESERVE_CONTEXT_ON_EXCEPTION': False
}

# Disable SQLAlchemy event system for tests
SQLALCHEMY_ENGINE_OPTIONS = {
    'poolclass': 'StaticPool',
    'connect_args': {'check_same_thread': False}
}

@pytest.fixture(scope='session')
def app():
    """Create and configure a new app instance for testing."""
    # Create a test app with the test configuration
    app = create_app(TEST_CONFIG)
    
    # Set up the database
    with app.app_context():
        _db.create_all()
        _db.session.commit()  # Ensure tables are created
    
    yield app
    
    # Clean up the database after tests
    with app.app_context():
        _db.session.remove()
        _db.drop_all()
        _db.get_engine(app).dispose()

@pytest.fixture(scope='function')
def client(app):
    """A test client for the app."""
    with app.test_client() as client:
        with app.app_context():
            # No need to create_all here as it's handled by the db fixture
            pass
        yield client

@pytest.fixture(scope='function')
def db(app):
    """A database session for testing."""
    with app.app_context():
        # Drop all tables and recreate them
        _db.drop_all()
        _db.create_all()
        _db.session.commit()  # Ensure schema is applied
    
    # Yield the db instance
    yield _db
    
    # Clean up after each test
    with app.app_context():
        _db.session.remove()
        _db.drop_all()
        _db.session.commit()

@pytest.fixture
def test_user(db):
    """Create a test user."""
    user = User(
        username='testuser',
        email='test@example.com',
        full_name='Test User'
    )
    user.set_password('testpass123')
    db.session.add(user)
    db.session.commit()
    return user

@pytest.fixture
def test_study_preference(db, test_user):
    """Create a test study preference."""
    preference = StudyPreference(
        user_id=test_user.id,
        course_name='Python Programming',
        study_style='visual',
        goals='Learn advanced Python',
        preferred_study_times='Evenings',
        availability='Mon, Wed, Fri 6-9pm'
    )
    db.session.add(preference)
    db.session.commit()
    return preference

@pytest.fixture
def test_buddy_request(db, test_user):
    """Create a test buddy request."""
    # Create a second user
    user2 = User(
        username='testuser2',
        email='test2@example.com',
        full_name='Test User 2'
    )
    user2.set_password('testpass123')
    db.session.add(user2)
    
    # Create request
    request = BuddyRequest(
        sender_id=test_user.id,
        receiver_id=user2.id,
        message='Let\'s study together!'
    )
    db.session.add(request)
    db.session.commit()
    return request
