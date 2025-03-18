"""
Main entry point for Emotion Sound System
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root directory to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

from src.ui.app import create_app

def main():
    """Run the Emotion Sound System application."""
    try:
        logger.info("Starting Emotion Sound System application...")
        
        # Create the Flask application
        app = create_app()
        if app is None:
            raise RuntimeError("Failed to create Flask application - check the logs for details")
        
        # Configure the application
        logger.debug("Configuring Flask application...")
        app.config['DEBUG'] = True
        app.config['ENV'] = 'development'
        app.config['TEMPLATES_AUTO_RELOAD'] = True
        
        # Run the application
        logger.info("Starting Flask server...")
        app.run(
            host='127.0.0.1',  # localhost
            port=5000,         # default Flask port
            debug=True         # enable debug mode
        )
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        if hasattr(e, '__traceback__'):
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()