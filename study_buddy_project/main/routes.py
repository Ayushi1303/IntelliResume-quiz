from flask import render_template, redirect, url_for, flash, request, jsonify
from flask_login import login_required, current_user
from study_buddy_project import db
from study_buddy_project.models import User, StudyPreference, BuddyRequest
from study_buddy_project.matching_algorithm import StudyBuddyMatcher
from . import bp

@bp.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    return render_template('index.html')

@bp.route('/dashboard')
@login_required
def dashboard():
    # Get pending requests
    pending_requests = BuddyRequest.query.filter_by(
        receiver_id=current_user.id,
        status='pending'
    ).all()
    
    # Get study buddies
    matcher = StudyBuddyMatcher(current_user.id)
    buddies = matcher.get_buddies()
    
    # Get user preferences
    preferences = StudyPreference.query.filter_by(
        user_id=current_user.id
    ).first()
    
    return render_template('dashboard.html',
                         pending_requests=pending_requests,
                         buddies=buddies,
                         preferences=preferences)

@bp.route('/find-buddies')
@login_required
def find_buddies():
    matcher = StudyBuddyMatcher(current_user.id)
    potential_matches = matcher.find_potential_matches(limit=10)
    return render_template('find_buddy.html', matches=potential_matches)

@bp.route('/send-request/<int:receiver_id>', methods=['POST'])
@login_required
def send_request(receiver_id):
    message = request.form.get('message', '')
    matcher = StudyBuddyMatcher(current_user.id)
    success, msg = matcher.send_buddy_request(receiver_id, message)
    
    if success:
        flash('Buddy request sent!', 'success')
    else:
        flash(f'Error: {msg}', 'error')
        
    return redirect(url_for('main.find_buddies'))

@bp.route('/respond-request/<int:request_id>/<action>', methods=['POST'])
@login_required
def respond_request(request_id, action):
    matcher = StudyBuddyMatcher(current_user.id)
    success, msg = matcher.respond_to_request(request_id, action == 'accept')
    
    if success:
        status = 'accepted' if action == 'accept' else 'rejected'
        flash(f'Request {status} successfully!', 'success')
    else:
        flash(f'Error: {msg}', 'error')
        
    return redirect(url_for('main.dashboard'))

@bp.route('/update-preferences', methods=['GET', 'POST'])
@login_required
def update_preferences():
    if request.method == 'POST':
        course_name = request.form.get('course_name')
        study_style = request.form.get('study_style')
        preferred_times = request.form.get('preferred_times')
        availability = request.form.get('availability')
        goals = request.form.get('goals')
        
        prefs = StudyPreference.query.filter_by(user_id=current_user.id).first()
        if not prefs:
            prefs = StudyPreference(user_id=current_user.id)
            db.session.add(prefs)
        
        prefs.course_name = course_name
        prefs.study_style = study_style
        prefs.preferred_study_times = preferred_times
        prefs.availability = availability
        prefs.goals = goals
        
        try:
            db.session.commit()
            flash('Preferences updated successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash('Error updating preferences. Please try again.', 'error')
        
        return redirect(url_for('main.dashboard'))
    
    preferences = StudyPreference.query.filter_by(user_id=current_user.id).first()
    return render_template('preferences.html', preferences=preferences)
