# Emotion Sound System

An interactive system that helps people with alexithymia express their emotions through music. The system analyzes the sentiment of text and converts it to musical output, mapping emotional content to consonance/dissonance and harmonic qualities.

## Features

- **Text Sentiment Analysis**: Analyzes the emotional content of text using VADER sentiment analysis.
- **Music Generation**: Converts sentiment scores to musical elements, mapping positive emotions to consonant intervals and negative emotions to dissonant intervals.
- **Web Interface**: Provides a user-friendly web interface for interacting with the system.
- **Facial Expression Analysis**: (Optional) Analyzes facial expressions to provide multimodal emotion recognition.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/EmotionSoundSystem.git
   cd EmotionSoundSystem
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Download NLTK data:
   ```python
   python -c "import nltk; nltk.download('vader_lexicon')"
   ```

## Usage

The system can be run in three different modes:

### Web Interface Mode (Default)

```
python main.py --mode web --port 5000
```

This will start a web server at http://localhost:5000 where you can interact with the system through a user-friendly interface.

### Text Analysis Mode

```
python main.py --mode text --input "Your text here" --output "output.mid"
```

This will analyze the sentiment of the provided text and generate a MIDI file with the corresponding musical output.

### Facial Expression Analysis Mode

```
python main.py --mode face --output "expression.mid"
```

This will open your webcam and analyze your facial expressions in real-time. Press 'c' to capture your current expression and generate music based on it, or 'q' to quit.

## Project Structure

- `src/sentiment/`: Sentiment analysis module
- `src/music/`: Music generation module
- `src/ui/`: Web interface module
- `src/facial_analysis/`: Facial expression analysis module
- `main.py`: Main application entry point

## Dependencies

- NLTK and VADER for sentiment analysis
- music21 for music generation and notation
- Flask for the web interface
- OpenCV for facial expression analysis
- pygame for playing generated music

## Future Enhancements

- Improved sentiment analysis with more nuanced emotional categories
- More sophisticated music generation with harmony, rhythm, and timbre variations
- Integration with speech recognition for real-time analysis of spoken language
- Support for different musical styles and cultural preferences
- Mobile application for increased accessibility

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VADER sentiment analysis: Hutto, C.J. & Gilbert, E.E. (2014)
- music21 toolkit: MIT
- OpenCV: Intel Corporation 