from flask import Blueprint

bp = Blueprint('main', __name__)

from study_buddy_project.main import routes
