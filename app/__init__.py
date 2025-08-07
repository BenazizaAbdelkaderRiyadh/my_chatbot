from flask import Flask
from .auth import auth_bp
from .main import main_bp

def create_app():
    """
    Application factory function to create and configure the Flask app.
    """
    app = Flask(__name__)
    
    app.secret_key = 'your_secret_key_needs_to_be_changed'

    # Register Blueprints for different parts of the application
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    return app