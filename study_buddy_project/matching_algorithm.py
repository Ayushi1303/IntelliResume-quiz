from sqlalchemy import and_
from .models import db, StudyPreference, User, BuddyRequest
from datetime import datetime, timedelta

class StudyBuddyMatcher:
    def __init__(self, user_id):
        self.user_id = user_id
    
    def find_potential_matches(self, limit=5):
        """
        Find potential study buddies based on course preferences and availability.
        Returns a list of potential matches with compatibility scores.
        """
        # Get current user's preferences
        user_prefs = StudyPreference.query.filter_by(user_id=self.user_id).first()
        if not user_prefs:
            return []
            
        # Find users with matching course preferences
        potential_matches = StudyPreference.query.filter(
            and_(
                StudyPreference.user_id != self.user_id,
                StudyPreference.course_name == user_prefs.course_name
            )
        ).limit(limit * 3).all()  # Get extra to filter further
        
        # Score matches based on compatibility
        scored_matches = []
        for match in potential_matches:
            score = self._calculate_match_score(user_prefs, match)
            if score > 0:  # Only include matches with some compatibility
                user = User.query.get(match.user_id)
                if user:
                    # Check if there's already a pending/accepted request
                    existing_request = BuddyRequest.query.filter(
                        ((BuddyRequest.sender_id == self.user_id) & 
                         (BuddyRequest.receiver_id == user.id)) |
                        ((BuddyRequest.sender_id == user.id) & 
                         (BuddyRequest.receiver_id == self.user_id))
                    ).first()
                    
                    if not existing_request or existing_request.status == 'rejected':
                        scored_matches.append({
                            'user': user,
                            'preferences': match,
                            'score': score
                        })
        
        # Sort by score and return top matches
        scored_matches.sort(key=lambda x: x['score'], reverse=True)
        return scored_matches[:limit]
    
    def _calculate_match_score(self, user_prefs, other_prefs):
        """Calculate compatibility score between two users' preferences.
        Higher score means better match."""
        score = 0
        
        # Course match (already filtered in query)
        score += 30
        
        # Study style compatibility
        if user_prefs.study_style == other_prefs.study_style:
            score += 20
            
        # Time availability overlap (simplified example)
        if self._check_time_overlap(user_prefs.preferred_study_times, 
                                  other_prefs.preferred_study_times):
            score += 25
            
        # Availability match (days)
        if user_prefs.availability and other_prefs.availability:
            user_days = set(day.strip().lower() for day in user_prefs.availability.split(','))
            other_days = set(day.strip().lower() for day in other_prefs.availability.split(','))
            if user_days.intersection(other_days):
                score += 25
                
        return min(score, 100)  # Cap score at 100
    
    def _check_time_overlap(self, time1, time2):
        """Check if two time ranges overlap (simplified example)."""
        if not time1 or not time2:
            return False
            
        # This is a simplified example - would need to be more sophisticated
        # for real time range comparisons
        time1_parts = time1.lower().split('-')
        time2_parts = time2.lower().split('-')
        
        if len(time1_parts) == 2 and len(time2_parts) == 2:
            try:
                # Very basic time range comparison
                time1_start = self._parse_time(time1_parts[0])
                time1_end = self._parse_time(time1_parts[1])
                time2_start = self._parse_time(time2_parts[0])
                time2_end = self._parse_time(time2_parts[1])
                
                return (time1_start <= time2_end and time1_end >= time2_start)
            except:
                return False
        return False
    
    def _parse_time(self, time_str):
        """Convert time string to minutes since midnight for comparison."""
        time_str = time_str.strip()
        try:
            if 'am' in time_str or 'pm' in time_str:
                # Handle 12-hour format
                time_obj = datetime.strptime(time_str, '%I:%M%p')
            else:
                # Handle 24-hour format
                time_obj = datetime.strptime(time_str, '%H:%M')
            return time_obj.hour * 60 + time_obj.minute
        except ValueError:
            # If parsing fails, return a default value
            return 0

    def send_buddy_request(self, receiver_id, message=None):
        """Send a buddy request to another user."""
        # Check if request already exists
        existing = BuddyRequest.query.filter_by(
            sender_id=self.user_id,
            receiver_id=receiver_id
        ).first()
        
        if existing:
            return False, "Request already sent"
            
        request = BuddyRequest(
            sender_id=self.user_id,
            receiver_id=receiver_id,
            message=message,
            status='pending'
        )
        
        try:
            db.session.add(request)
            db.session.commit()
            return True, "Request sent successfully"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
    
    def respond_to_request(self, request_id, accept):
        """Respond to a buddy request (accept or reject)."""
        request = BuddyRequest.query.get(request_id)
        if not request:
            return False, "Request not found"
            
        if request.receiver_id != self.user_id:
            return False, "Not authorized to respond to this request"
            
        if request.status != 'pending':
            return False, "Request has already been processed"
            
        request.status = 'accepted' if accept else 'rejected'
        
        try:
            db.session.commit()
            return True, "Request updated successfully"
        except Exception as e:
            db.session.rollback()
            return False, str(e)
