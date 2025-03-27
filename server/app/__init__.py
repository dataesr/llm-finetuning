from flask import Flask
from flask_cors import CORS
from app.home.routes import home_routes
from app.models.routes import models_routes


def create_app():
    # Instantiate the app
    app = Flask(
        __name__,
    )

    # Set CORS
    CORS(app, origins="*")

    # Set config
    # app.config.from_object("app_settings_module")

    # Blueprints registration
    app.register_blueprint(home_routes)
    app.register_blueprint(models_routes)

    # Shell context for flask cli
    app.shell_context_processor({"app": app})

    return app
