"""
Test script for the music generator
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.sentiment.analyzer import MultimodalSentimentAnalyzer
from src.music.generator import MusicGenerator

def test_music_generation():
    """Test basic music generation functionality."""
    # Create analyzers
    sentiment_analyzer = MultimodalSentimentAnalyzer()
    music_generator = MusicGenerator()
    
    # Test with a sample text
    sample_text = "I love this beautiful day. The weather is nice and I feel happy."
    print(f"Analyzing text: {sample_text}")
    
    # Get word sentiments
    word_sentiments = sentiment_analyzer.analyze_words(sample_text)
    print("\nWord-level sentiments:")
    for word_info in word_sentiments:
        print(f"{word_info['word']}: {word_info['score']:.4f} ({word_info['emotion']})")
    
    # Generate melody
    print("\nGenerating melody...")
    melody = music_generator.generate_melody_from_word_sentiments(word_sentiments)
    
    # Save the melody
    output_file = "test_melody.mid"
    melody.write('midi', fp=output_file)
    print(f"\nMelody saved to: {output_file}")
    
    # Try to play the melody
    try:
        print("\nAttempting to play melody...")
        music_generator.play_melody(melody)
    except Exception as e:
        print(f"Could not play melody: {e}")
        print("You may need to configure a MIDI player.")

if __name__ == "__main__":
    test_music_generation() 