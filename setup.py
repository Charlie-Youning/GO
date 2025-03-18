import nltk
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup():
    """Download required NLTK data."""
    try:
        logger.info("Downloading NLTK vader_lexicon...")
        nltk.download('vader_lexicon', quiet=True)
        logger.info("Successfully downloaded NLTK data")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {str(e)}")
        raise

if __name__ == "__main__":
    setup() 