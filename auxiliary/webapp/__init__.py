import logging
import sys

from flask import Flask
from werkzeug.wsgi import DispatcherMiddleware


def app_root_404(env, resp):
    resp("404", [("Content-Type", "text/plain")])
    return [b"404 The application root has been reconfigured."]


# Logging config
logging.basicConfig(level=logging.INFO,
                    handlers=[logging.StreamHandler(sys.stderr),
                              logging.FileHandler("second-dream-webapp.log")])

# Create app
app = Flask(__name__)
app.config["WTF_CSRF_ENABLED"] = False

# Load config from disk
app.config.from_pyfile("settings.cfg", silent=True)
app.config.from_envvar("SD_SETTINGS", silent=True)

# Change the application root if configured
if "APPLICATION_ROOT" in app.config and app.config["APPLICATION_ROOT"] != "/":
    app.wsgi_app = DispatcherMiddleware(app_root_404, {app.config["APPLICATION_ROOT"]: app.wsgi_app})

# Load data
from . import data
# Initialize routes
from . import routes
