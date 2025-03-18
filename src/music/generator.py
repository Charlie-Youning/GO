import music21
import numpy as np
import os
import tempfile
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Handle imports for both package and direct execution
if __name__ == '__main__':
    # Add the project root to the Python path for direct execution
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    from src.sentiment.analyzer import MultimodalSentimentAnalyzer
else:
    # Use relative import when used as a package
    from ..sentiment.analyzer import MultimodalSentimentAnalyzer

class MusicGenerator:
    def __init__(self):
        """Initialize the music generator with default settings."""
        # Define the mapping between sentiment scores and musical intervals
        # Positive sentiment -> consonant intervals (perfect 5th, major 3rd)
        # Negative sentiment -> dissonant intervals (minor 2nd, tritone)
        self.interval_mapping = {
            # Very negative: tritone (augmented 4th)
            -1.0: music21.interval.Interval('A4'),
            # Negative: minor 2nd
            -0.75: music21.interval.Interval('m2'),
            # Slightly negative: minor 7th
            -0.5: music21.interval.Interval('m7'),
            # Neutral: perfect 4th
            0.0: music21.interval.Interval('P4'),
            # Slightly positive: major 2nd
            0.25: music21.interval.Interval('M2'),
            # Positive: major 3rd
            0.5: music21.interval.Interval('M3'),
            # Very positive: perfect 5th
            0.75: music21.interval.Interval('P5'),
            # Extremely positive: octave
            1.0: music21.interval.Interval('P8')
        }
        
        # Base note (middle C)
        self.base_note = music21.note.Note('C4')
        
        # Tempo mapping (sentiment to tempo)
        # Positive -> faster, Negative -> slower
        self.min_tempo = 60  # beats per minute
        self.max_tempo = 120  # beats per minute
        
    def _get_closest_interval(self, sentiment_score):
        """Get the closest predefined interval for a given sentiment score."""
        # Find the closest sentiment value in our mapping
        sentiment_values = list(self.interval_mapping.keys())
        closest_idx = np.argmin(np.abs(np.array(sentiment_values) - sentiment_score))
        closest_sentiment = sentiment_values[closest_idx]
        
        return self.interval_mapping[closest_sentiment]
    
    def _calculate_tempo(self, sentiment_score):
        """Calculate tempo based on sentiment score."""
        # Map sentiment from [-1, 1] to [min_tempo, max_tempo]
        normalized_score = (sentiment_score + 1) / 2  # Map from [-1, 1] to [0, 1]
        tempo = self.min_tempo + normalized_score * (self.max_tempo - self.min_tempo)
        return int(tempo)
    
    def generate_note_from_sentiment(self, sentiment_score):
        """Generate a musical note based on sentiment score."""
        # Get the interval based on sentiment
        interval = self._get_closest_interval(sentiment_score)
        
        # Create a new note by adding the interval to the base note
        new_note = interval.transposeNote(self.base_note)
        
        # Set duration based on sentiment (more positive = shorter/happier)
        if sentiment_score > 0:
            new_note.duration.quarterLength = 0.5  # Eighth note for positive
        else:
            new_note.duration.quarterLength = 1.0  # Quarter note for negative
            
        return new_note
    
    def generate_melody_from_word_sentiments(self, word_sentiments):
        """
        Generate a melody from a list of word sentiment scores.
        
        Args:
            word_sentiments: List of tuples (word, sentiment_score)
            
        Returns:
            A music21 Stream object containing the melody
        """
        # Create a music21 stream
        melody = music21.stream.Stream()
        
        # Set tempo based on overall sentiment
        if word_sentiments:
            overall_sentiment = sum(score for _, score in word_sentiments) / len(word_sentiments)
            tempo = self._calculate_tempo(overall_sentiment)
            melody.append(music21.tempo.MetronomeMark(number=tempo))
        
        # Add notes for each word
        for word, sentiment in word_sentiments:
            note = self.generate_note_from_sentiment(sentiment)
            # Add the word as lyric
            note.lyric = word
            melody.append(note)
        
        return melody
    
    def save_melody(self, melody, output_format='midi', filename=None):
        """
        Save the melody to a file.
        
        Args:
            melody: A music21 Stream object
            output_format: Format to save (midi, xml, etc.)
            filename: Output filename (if None, a temporary file is created)
            
        Returns:
            The path to the saved file
        """
        if filename is None:
            # Create a temporary file
            fd, filename = tempfile.mkstemp(suffix=f'.{output_format}')
            os.close(fd)
        
        # Save the melody
        melody.write(output_format, fp=filename)
        
        return filename
    
    def play_melody(self, melody):
        """
        Play the melody using music21's built-in player.
        
        Args:
            melody: A music21 Stream object
        """
        try:
            melody.show('midi')
        except Exception as e:
            print(f"Error playing melody: {e}")
            print("You may need to configure music21 to use a specific MIDI player.")
            print("See: https://web.mit.edu/music21/doc/usersGuide/usersGuide_23_environment.html")

# Example usage
if __name__ == "__main__":
    # Create a sentiment analyzer
    analyzer = MultimodalSentimentAnalyzer()
    
    # Analyze a sample text
    sample_text = "I love this beautiful day. The weather is nice and I feel happy."
    word_sentiments = analyzer.analyze_words(sample_text)
    
    # Create a music generator
    generator = MusicGenerator()
    
    # Generate a melody from the word sentiments
    melody = generator.generate_melody_from_word_sentiments(word_sentiments)
    
    # Save the melody to a MIDI file
    midi_file = generator.save_melody(melody)
    print(f"Melody saved to: {midi_file}")
    
    # Play the melody
    generator.play_melody(melody) 