import logging
import sys

from flask import Flask

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

# Load data
from . import data
# Initialize routes
from . import routes
