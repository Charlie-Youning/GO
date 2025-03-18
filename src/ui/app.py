from flask import Flask, render_template, request, jsonify, send_file
import os
import sys
import tempfile
import logging
import traceback
import json
from pathlib import Path
import time
import numpy as np
import cv2

# Add the project root to Python path for absolute imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Use absolute imports
from src.music.generator import MusicGenerator
from src.sentiment.analyzer import MultimodalSentimentAnalyzer
from src.sentiment.enhanced_analyzer import EnhancedSentimentAnalyzer
from src.music.enhanced_generator import EnhancedMusicGenerator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_app():
    """Create and configure the Flask application."""
    try:
        # Create Flask app
        app = Flask(__name__)
        app.debug = True  # Enable debug mode
        logger.debug("Created Flask application")

        # Configure upload folder
        app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

        # Initialize analyzers as global variables
        global sentiment_analyzer, music_generator
        try:
            # Try to use enhanced analyzers first
            try:
                sentiment_analyzer = EnhancedSentimentAnalyzer()
                music_generator = EnhancedMusicGenerator()
                logger.info("Successfully initialized enhanced analyzers")
            except Exception as e:
                logger.warning(f"Could not initialize enhanced analyzers: {e}. Falling back to basic versions.")
                sentiment_analyzer = MultimodalSentimentAnalyzer()
                music_generator = MusicGenerator()
                logger.info("Successfully initialized basic analyzers")
        except Exception as e:
            logger.error(f"Failed to initialize analyzers: {str(e)}")
            logger.error(traceback.format_exc())
            raise

        @app.before_request
        def log_request_info():
            """Log details about the incoming request."""
            logger.debug('Headers: %s', dict(request.headers))
            logger.debug('Body: %s', request.get_data(as_text=True))
            logger.debug('Args: %s', dict(request.args))
            logger.debug('Form: %s', dict(request.form))

        @app.route('/')
        def index():
            """Render the main page."""
            try:
                return render_template('index.html')
            except Exception as e:
                logger.error(f"Error rendering index page: {str(e)}")
                logger.error(traceback.format_exc())
                return jsonify({'error': str(e)}), 500

        @app.route('/analyze', methods=['POST'])
        def analyze():
            """Analyze sentiment using available modalities."""
            try:
                # Log request details
                logger.debug("Received analyze request")
                logger.debug(f"Request headers: {dict(request.headers)}")
                logger.debug(f"Request Content-Type: {request.content_type}")

                # Initialize variables for multimodal analysis
                text = None
                audio_path = None
                face_image = None

                # Handle different content types
                if request.content_type == 'application/json':
                    try:
                        data = request.get_json()
                        logger.debug(f"Parsed JSON data: {data}")
                        text = data.get('text')
                    except Exception as e:
                        logger.error(f"Error parsing JSON: {str(e)}")
                        return jsonify({'error': f'Invalid JSON format: {str(e)}'}), 400
                elif request.content_type and request.content_type.startswith('multipart/form-data'):
                    # Get text input
                    text = request.form.get('text')

                    # Handle audio file
                    if 'audio' in request.files:
                        audio_file = request.files['audio']
                        if audio_file.filename:
                            try:
                                # Save audio file
                                audio_path = os.path.join(
                                    app.config['UPLOAD_FOLDER'],
                                    'temp_audio_' + str(int(time.time())) + '.wav'
                                )
                                audio_file.save(audio_path)
                            except Exception as e:
                                logger.error(f"Error saving audio file: {str(e)}")
                                return jsonify({'error': f'Error processing audio: {str(e)}'}), 400

                    # Handle image file
                    if 'image' in request.files:
                        image_file = request.files['image']
                        if image_file.filename:
                            try:
                                # Read and process image
                                image_data = image_file.read()
                                nparr = np.frombuffer(image_data, np.uint8)
                                face_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            except Exception as e:
                                logger.error(f"Error processing image: {str(e)}")
                                return jsonify({'error': f'Error processing image: {str(e)}'}), 400
                else:
                    # Handle x-www-form-urlencoded or missing content type
                    try:
                        text = request.form.get('text')
                        if not text and request.data:
                            # Try to parse as JSON if form data is empty
                            try:
                                data = json.loads(request.data)
                                text = data.get('text')
                            except:
                                # Last resort: try to get raw data as text
                                text = request.get_data(as_text=True)
                    except Exception as e:
                        logger.error(f"Error extracting text from request: {str(e)}")
                        return jsonify({'error': 'Could not extract text from request'}), 400

                # Validate that at least one input is provided
                if not any([text, audio_path, face_image]):
                    return jsonify({'error': 'No input provided'}), 400

                # Perform multimodal analysis
                try:
                    analysis_results = sentiment_analyzer.analyze_multimodal(
                        text=text,
                        audio_path=audio_path,
                        face_image=face_image
                    )
                    logger.debug(f"Analysis results: {analysis_results}")
                except Exception as e:
                    logger.error(f"Error in sentiment analysis: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({'error': f'Error analyzing input: {str(e)}'}), 500

                # Clean up temporary files
                if audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception as e:
                        logger.warning(f"Error removing temporary audio file: {str(e)}")

                # Prepare word-level analysis if text was provided
                if text:
                    try:
                        word_sentiments = sentiment_analyzer.analyze_words(text)
                        analysis_results['word_sentiments'] = word_sentiments
                    except Exception as e:
                        logger.error(f"Error in word sentiment analysis: {str(e)}")
                        logger.error(traceback.format_exc())
                        return jsonify({'error': f'Error analyzing word sentiments: {str(e)}'}), 500

                return jsonify(analysis_results)

            except Exception as e:
                error_msg = f"Error in analyze endpoint: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return jsonify({'error': error_msg}), 500

        @app.route('/generate-music', methods=['POST'])
        def generate_music():
            """Generate music based on the provided text."""
            try:
                # Log request details
                logger.debug("Received generate-music request")
                logger.debug(f"Request headers: {dict(request.headers)}")
                logger.debug(f"Request Content-Type: {request.content_type}")

                # Get request data
                try:
                    if request.is_json:
                        data = request.get_json()
                        logger.debug(f"Parsed JSON data: {data}")
                    else:
                        raw_data = request.get_data(as_text=True)
                        logger.debug(f"Raw request data: {raw_data}")
                        if not raw_data:
                            return jsonify({'error': 'Empty request body'}), 400
                        try:
                            data = json.loads(raw_data)
                            logger.debug(f"Parsed JSON from raw data: {data}")
                        except json.JSONDecodeError as e:
                            logger.error(f"JSON decode error: {str(e)}")
                            # Try to parse as form data
                            data = dict(request.form)
                            logger.debug(f"Parsed form data: {data}")
                            if not data:
                                return jsonify({'error': f'Invalid request format: {str(e)}'}), 400
                except Exception as e:
                    logger.error(f"Error reading request data: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({'error': f'Error reading request data: {str(e)}'}), 400

                # Validate data
                if not isinstance(data, dict):
                    error_msg = f"Invalid data format: expected dict, got {type(data)}"
                    logger.error(error_msg)
                    return jsonify({'error': error_msg}), 400

                text = data.get('text')
                if not text:
                    logger.warning("No text provided in request")
                    return jsonify({'error': 'No text provided'}), 400

                if not isinstance(text, str):
                    error_msg = f"Invalid text format: expected string, got {type(text)}"
                    logger.error(error_msg)
                    return jsonify({'error': error_msg}), 400

                # Analyze word-level sentiments
                logger.debug("Analyzing word-level sentiments")
                try:
                    word_sentiments = sentiment_analyzer.analyze_words(text)
                    if not word_sentiments or len(word_sentiments) == 0:
                        logger.warning(f"No word sentiments generated for text: {text}")
                        return jsonify({'error': 'Could not analyze sentiment from the provided text'}), 400

                    logger.debug(f"Word sentiments result: {word_sentiments}")
                except Exception as e:
                    logger.error(f"Error analyzing word sentiments: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({'error': f'Error analyzing word sentiments: {str(e)}'}), 500

                # Generate melody
                logger.debug("Generating melody")
                try:
                    # Format word sentiments for the generator based on analyzer type
                    formatted_word_sentiments = []

                    # Check the format of word_sentiments
                    if word_sentiments and isinstance(word_sentiments, list):
                        for item in word_sentiments:
                            if isinstance(item, dict) and 'word' in item and 'score' in item:
                                # Original analyzer format
                                formatted_word_sentiments.append((item['word'], item['score']))
                            elif isinstance(item, tuple):
                                # Check tuple length
                                if len(item) >= 2:
                                    # Already in the right format
                                    formatted_word_sentiments.append(item)
                                else:
                                    logger.warning(f"Invalid tuple format in word sentiments: {item}")
                            else:
                                logger.warning(f"Unknown word sentiment format: {item}")

                    # If no valid sentiments were extracted, return an error
                    if not formatted_word_sentiments:
                        logger.warning("No valid word sentiments extracted")
                        return jsonify({'error': 'Could not extract valid sentiments from text'}), 400

                    # Generate the melody
                    melody = music_generator.generate_melody_from_word_sentiments(formatted_word_sentiments)
                except Exception as e:
                    logger.error(f"Error generating melody: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({'error': f'Error generating melody: {str(e)}'}), 500

                # Save to temporary file
                logger.debug("Saving melody to temporary file")
                try:
                    with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                        midi_file = tmp.name
                        melody.write('midi', fp=midi_file)

                    # Get just the filename without the path
                    filename = os.path.basename(midi_file)
                    logger.debug(f"Generated MIDI file: {filename}")

                    response_data = {
                        'midi_file': filename,
                        'message': 'Music generated successfully'
                    }

                    return jsonify(response_data)
                except Exception as e:
                    logger.error(f"Error saving melody: {str(e)}")
                    logger.error(traceback.format_exc())
                    return jsonify({'error': f'Error saving melody: {str(e)}'}), 500

            except Exception as e:
                error_msg = f"Error in generate_music endpoint: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return jsonify({'error': error_msg}), 500

        @app.route('/download/<filename>')
        def download_midi(filename):
            """Download the generated MIDI file."""
            try:
                logger.debug(f"Download request for file: {filename}")
                # Get the full path to the temporary file
                temp_dir = tempfile.gettempdir()
                file_path = os.path.join(temp_dir, filename)
                logger.debug(f"Full file path: {file_path}")

                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    return jsonify({'error': 'File not found'}), 404

                return send_file(
                    file_path,
                    mimetype='audio/midi',
                    as_attachment=True,
                    download_name=filename
                )

            except Exception as e:
                error_msg = f"Error in download_midi endpoint: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                return jsonify({'error': error_msg}), 500

        @app.route('/visualize/<filename>')
        def visualize_midi(filename):
            """Render the visualization page for the MIDI file."""
            return render_template('visualize.html', filename=filename)

        # Add error handlers
        @app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal Server Error: {error}")
            return jsonify({'error': 'Internal Server Error', 'details': str(error)}), 500

        @app.errorhandler(404)
        def not_found_error(error):
            logger.error(f"Not Found Error: {error}")
            return jsonify({'error': 'Not Found', 'details': str(error)}), 404

        @app.errorhandler(Exception)
        def handle_exception(error):
            logger.error(f"Unhandled Exception: {error}")
            logger.error(traceback.format_exc())
            return jsonify({'error': 'Internal Server Error', 'details': str(error)}), 500

        logger.debug("Successfully configured all routes")
        return app

    except Exception as e:
        logger.error(f"Error creating Flask application: {str(e)}")
        logger.error(traceback.format_exc())
        return None