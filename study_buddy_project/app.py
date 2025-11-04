from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from .models import db, User, StudyPreference, BuddyRequest
from .matching_algorithm import StudyBuddyMatcher
from .database import init_db, get_user_by_username, get_user_by_email, add_to_db, get_user_preferences, update_user_preferences, get_pending_requests, get_sent_requests, get_buddies, search_users
from .config import Config
import os

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # Initialize extensions
    db.init_app(app)
    
    # Initialize login manager
    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)
    
    @login_manager.user_loader
    def load_user(user_id):
        return User.query.get(int(user_id))
    
    # Initialize database
    with app.app_context():
        db.create_all()
    
    # Blueprint registration would go here
    # app.register_blueprint(auth_bp)
    # app.register_blueprint(main_bp)
    
    # Routes
    @app.route('/')
    def index():
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        return render_template('index.html')
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        # Get pending requests
        pending_requests = get_pending_requests(current_user.id)
        
        # Get study buddies
        buddies = get_buddies(current_user.id)
        
        # Get user preferences
        preferences = get_user_preferences(current_user.id)
        
        return render_template('dashboard.html', 
                             pending_requests=pending_requests,
                             buddies=buddies,
                             preferences=preferences)
    
    @app.route('/find-buddies')
    @login_required
    def find_buddies():
        matcher = StudyBuddyMatcher(current_user.id)
        potential_matches = matcher.find_potential_matches(limit=10)
        return render_template('find_buddy.html', matches=potential_matches)
    
    @app.route('/send-request/<int:receiver_id>', methods=['POST'])
    @login_required
    def send_request(receiver_id):
        message = request.form.get('message', '')
        matcher = StudyBuddyMatcher(current_user.id)
        success, msg = matcher.send_buddy_request(receiver_id, message)
        
        if success:
            flash('Buddy request sent!', 'success')
        else:
            flash(f'Error: {msg}', 'error')
            
        return redirect(url_for('find_buddies'))
    
    @app.route('/respond-request/<int:request_id>/<action>', methods=['POST'])
    @login_required
    def respond_request(request_id, action):
        matcher = StudyBuddyMatcher(current_user.id)
        success, msg = matcher.respond_to_request(request_id, action == 'accept')
        
        if success:
            status = 'accepted' if action == 'accept' else 'rejected'
            flash(f'Request {status} successfully!', 'success')
        else:
            flash(f'Error: {msg}', 'error')
            
        return redirect(url_for('dashboard'))
    
    @app.route('/update-preferences', methods=['GET', 'POST'])
    @login_required
    def update_preferences():
        if request.method == 'POST':
            course_name = request.form.get('course_name')
            study_style = request.form.get('study_style')
            preferred_times = request.form.get('preferred_times')
            availability = request.form.get('availability')
            goals = request.form.get('goals')
            
            success, msg = update_user_preferences(
                current_user.id,
                course_name=course_name,
                study_style=study_style,
                preferred_study_times=preferred_times,
                availability=availability,
                goals=goals
            )
            
            if success:
                flash('Preferences updated successfully!', 'success')
            else:
                flash(f'Error: {msg}', 'error')
                
            return redirect(url_for('dashboard'))
            
        return render_template('preferences.html')
    
    @app.route('/search')
    @login_required
    def search():
        query = request.args.get('q', '')
        results = []
        if query:
            results = search_users(query, current_user.id)
        return render_template('search.html', results=results, query=query)
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(e):
        db.session.rollback()
        return render_template('500.html'), 500
    
    return app

# For development
if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
