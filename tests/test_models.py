"""Unit tests for database models."""
import pytest
from datetime import datetime, timedelta
from study_buddy_project.models import User, StudyPreference, BuddyRequest, db as _db
from werkzeug.security import check_password_hash

def test_user_creation(app, db, client):
    with app.app_context():
        user = User(username='testuser', email='test@example.com', full_name='Test User')
        user.set_password('testpass123')
        db.session.add(user)
        db.session.commit()
        
        assert user.id is not None
        assert user.username == 'testuser'
        assert user.email == 'test@example.com'
        assert user.full_name == 'Test User'
        assert check_password_hash(user.password_hash, 'testpass123')
        assert user.is_active is True
        assert isinstance(user.created_at, datetime)

def test_user_to_dict(app, db, client):
    with app.app_context():
        user = User(username='testuser', email='test@example.com', full_name='Test User')
        db.session.add(user)
        db.session.commit()
        
        user_dict = user.to_dict()
        assert user_dict['username'] == 'testuser'
        assert user_dict['email'] == 'test@example.com'
        assert 'password_hash' not in user_dict
        assert 'last_login' in user_dict

def test_study_preference_creation(app, db, client):
    with app.app_context():
        user = User(username='testuser', email='test@example.com')
        db.session.add(user)
        preference = StudyPreference(
            user=user,
            course_name='Computer Science',
            preferred_study_times='Evenings',
            study_style='Visual',
            goals='Learn Python'
        )
        db.session.add(preference)
        db.session.commit()
        
        assert preference.id is not None
        assert preference.user == user
        assert preference.course_name == 'Computer Science'
        assert preference.study_style == 'Visual'
        assert preference.goals == 'Learn Python'
        
        # Test update method
        preference.update(study_style='Auditory', goals='Master Python')
        assert preference.study_style == 'Auditory'
        assert preference.goals == 'Master Python'

def test_buddy_request_workflow(app, db, client):
    with app.app_context():
        sender = User(username='user1', email='user1@example.com')
        receiver = User(username='user2', email='user2@example.com')
        db.session.add_all([sender, receiver])
        
        request = BuddyRequest(sender=sender, receiver=receiver, message='Study together?')
        db.session.add(request)
        db.session.commit()
        
        assert request.status == 'pending'
        assert request.sender == sender
        assert request.receiver == receiver
        
        # Test accepting the request
        request.accept()
        assert request.status == 'accepted'
        assert request.updated_at is not None
        
        # Test rejecting the request
        request.reject()
        assert request.status == 'rejected'

def test_user_relationships(app, db, client):
    with app.app_context():
        # Test user-study preference relationship
        user = User(username='testuser', email='test@example.com')
        preference = StudyPreference(
            user=user, 
            course_name='Math',
            preferred_study_times='Afternoons',
            study_style='Visual',
            goals='Learn Math'
        )
        db.session.add_all([user, preference])
        db.session.commit()
        
        assert user.preferences == preference
        
        # Test buddy request relationships
        user1 = User(username='user1', email='user1@example.com')
        user2 = User(username='user2', email='user2@example.com')
        request = BuddyRequest(sender=user1, receiver=user2, message='Hi')
        db.session.add_all([user1, user2, request])
        db.session.commit()
        
        assert len(user1.sent_requests.all()) == 1
        assert len(user2.received_requests.all()) == 1
        assert request.updated_at is not None
        
        # Test rejecting the request
        request.reject()
        assert request.status == 'rejected'


def test_user_relationships(app, db, client):
    """Test relationships between users, preferences, and requests."""
    with app.app_context():
        # Create test users
        user1 = User(username='user1', email='user1@example.com')
        user2 = User(username='user2', email='user2@example.com')
        
        # Create study preferences
        pref1 = StudyPreference(
            user=user1,
            course_name='Math',
            study_style='visual',
            preferred_study_times='Afternoons',
            goals='Learn Math'
        )
        
        # Create buddy request
        request = BuddyRequest(
            sender=user1,
            receiver=user2,
            message='Math study partner?'
        )
        
        # Add to session and commit
        db.session.add_all([user1, user2, pref1, request])
        db.session.commit()
        
        # Test relationships
        assert user1.preferences == [pref1]
        assert len(user1.sent_requests.all()) == 1
        assert user1.sent_requests.first() == request
        assert len(user2.received_requests.all()) == 1
        assert user2.received_requests.first() == request
        
        # Test cascade delete
        db.session.delete(user1)
        db.session.commit()
        
        # User1 and their preferences should be deleted
        assert User.query.get(user1.id) is None
        assert StudyPreference.query.filter_by(user_id=user1.id).first() is None
        
        # But user2 should still exist
        assert User.query.get(user2.id) is not None
        # The request should also be deleted due to cascade delete
        assert BuddyRequest.query.get(request.id) is None
