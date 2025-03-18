"""
Enhanced music generator with more expressive capabilities based on emotions.
"""

import music21
import numpy as np
import os
import tempfile
import logging
import sys
from pathlib import Path
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EnhancedMusicGenerator:
    """Enhanced music generator with more expressive capabilities based on emotions."""

    def __init__(self):
        """Initialize the enhanced music generator."""
        # Define the mapping between sentiment scores and musical intervals
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

        # Extended interval mapping for more emotional expressiveness
        self.extended_interval_mapping = {
            # Joy spectrum - consonant intervals
            'ecstatic': music21.interval.Interval('M3'),  # Major third - very bright and happy
            'elated': music21.interval.Interval('P5'),  # Perfect fifth - clear and bright
            'happy': music21.interval.Interval('M6'),  # Major sixth - bright and cheerful
            'content': music21.interval.Interval('M3'),  # Major third - warm and pleasant
            'pleased': music21.interval.Interval('M2'),  # Major second - gentle movement

            # Neutral spectrum
            'neutral': music21.interval.Interval('P4'),  # Perfect fourth - stable, neutral
            'calm': music21.interval.Interval('P1'),  # Unison - serene, minimal movement
            'indifferent': music21.interval.Interval('P5'),  # Perfect fifth - detached clarity

            # Sadness spectrum - more minor and downward intervals
            'disappointed': music21.interval.Interval('m3'),  # Minor third - somber
            'sad': music21.interval.Interval('m6'),  # Minor sixth - melancholic
            'sorrowful': music21.interval.Interval('m7'),  # Minor seventh - deeply emotional
            'miserable': music21.interval.Interval('m2'),  # Minor second - dissonant, tense
            'despairing': music21.interval.Interval('d5'),  # Diminished fifth - despair

            # Anger spectrum - dissonant intervals
            'annoyed': music21.interval.Interval('M7'),  # Major seventh - sharp, edgy
            'angry': music21.interval.Interval('A4'),  # Augmented fourth - tense, unstable
            'furious': music21.interval.Interval('A5'),  # Augmented fifth - extreme tension

            # Fear spectrum - unsettling intervals
            'nervous': music21.interval.Interval('m7'),  # Minor seventh - uneasy
            'anxious': music21.interval.Interval('d5'),  # Diminished fifth - anxiety
            'scared': music21.interval.Interval('d7'),  # Diminished seventh - fear
            'terrified': music21.interval.Interval('A2')  # Augmented second - disorienting
        }

        # Base note (middle C)
        self.base_note = music21.note.Note('C4')

        # Tempo mapping (sentiment to tempo)
        # Positive -> faster, Negative -> slower
        self.min_tempo = 60  # beats per minute
        self.max_tempo = 120  # beats per minute

        # Emotion-based tempo mapping
        self.emotion_tempo_mapping = {
            # Joy spectrum - faster, energetic tempos
            'ecstatic': 160,  # Very fast, excited
            'elated': 140,  # Fast, lively
            'happy': 120,  # Moderate-fast, upbeat
            'content': 100,  # Moderate, comfortable
            'pleased': 100,  # Moderate, pleasant

            # Neutral spectrum - moderate tempos
            'neutral': 90,  # Moderate, steady
            'calm': 75,  # Slower moderate, relaxed
            'indifferent': 85,  # Moderate, unaffected

            # Sadness spectrum - slower tempos
            'disappointed': 80,  # Moderate-slow, slightly dragging
            'sad': 70,  # Slow, heavy
            'sorrowful': 60,  # Very slow, weighty
            'miserable': 50,  # Extremely slow, lethargic
            'despairing': 45,  # Slowest, barely moving

            # Anger spectrum - variable tempos, usually fast
            'annoyed': 110,  # Slightly fast, agitated
            'angry': 130,  # Fast, intense
            'furious': 150,  # Very fast, aggressive

            # Fear spectrum - erratic tempos
            'nervous': 100,  # Moderate but unstable
            'anxious': 115,  # Fast, fluttery
            'scared': 140,  # Fast, urgent
            'terrified': 160  # Very fast, frantic
        }

        # Rhythm patterns for different emotions
        self.emotion_rhythm_patterns = {
            # Joy spectrum - flowing, regular rhythms
            'ecstatic': [0.25, 0.25, 0.25, 0.25],  # Continuous eighth notes, energetic
            'elated': [0.5, 0.25, 0.25],  # Dotted rhythm, bouncy
            'happy': [0.5, 0.5],  # Regular quarter notes, steady
            'content': [1.0],  # Whole notes, flowing
            'pleased': [0.75, 0.25],  # Gentle dotted rhythm

            # Neutral spectrum - regular, even rhythms
            'neutral': [1.0],  # Whole notes, even
            'calm': [2.0],  # Half notes, spacious
            'indifferent': [1.0, 1.0],  # Even whole notes

            # Sadness spectrum - irregular, drawn-out rhythms
            'disappointed': [1.5, 0.5],  # Slightly uneven, hesitant
            'sad': [2.0, 1.0],  # Long, drawn out notes
            'sorrowful': [2.0, 1.5, 0.5],  # Very uneven, sighing quality
            'miserable': [3.0, 1.0],  # Extremely drawn out, heavy
            'despairing': [4.0],  # Longest notes, minimal movement

            # Anger spectrum - sharp, accented rhythms
            'annoyed': [0.5, 0.25, 0.25, 0.5],  # Agitated, uneven rhythm
            'angry': [0.25, 0.75, 0.25, 0.75],  # Sharp, accented rhythm
            'furious': [0.25, 0.25, 0.5, 0.25, 0.25, 0.5],  # Intense, driving rhythm

            # Fear spectrum - erratic, unpredictable rhythms
            'nervous': [0.25, 0.5, 0.25, 0.25, 0.75],  # Irregular, jumpy
            'anxious': [0.25, 0.75, 0.25, 0.25],  # Uneven, unsettled
            'scared': [0.125, 0.125, 0.25, 0.5],  # Rapid, urgent
            'terrified': [0.125, 0.125, 0.125, 0.125, 0.5]  # Most erratic, frantic
        }

        # Mode mapping for different emotions (major/minor/other scales)
        self.emotion_mode_mapping = {
            # Joy spectrum - major modes
            'ecstatic': 'major',  # Bright, joyful
            'elated': 'major',  # Cheerful
            'happy': 'major',  # Positive
            'content': 'major',  # Warm
            'pleased': 'major',  # Pleasant

            # Neutral spectrum - mixed modes
            'neutral': 'major',  # Balanced
            'calm': 'major',  # Peaceful
            'indifferent': 'dorian',  # Ambiguous

            # Sadness spectrum - minor modes
            'disappointed': 'minor',  # Slightly dark
            'sad': 'minor',  # Melancholic
            'sorrowful': 'minor',  # Deeply emotional
            'miserable': 'phrygian',  # Dark, brooding
            'despairing': 'locrian',  # Most dissonant and unstable

            # Anger spectrum - sharper minor modes
            'annoyed': 'minor',  # Tense
            'angry': 'phrygian',  # Aggressive
            'furious': 'locrian',  # Harsh, unstable

            # Fear spectrum - unstable modes
            'nervous': 'minor',  # Uneasy
            'anxious': 'phrygian',  # Unsettled
            'scared': 'locrian',  # Dissonant
            'terrified': 'diminished'  # Most unstable, dissonant
        }

        # Volume (dynamic) mapping for different emotions
        self.emotion_dynamic_mapping = {
            # Joy spectrum - louder, more energetic dynamics
            'ecstatic': 'ff',  # Fortissimo - very loud
            'elated': 'f',  # Forte - loud
            'happy': 'mf',  # Mezzo-forte - moderately loud
            'content': 'mp',  # Mezzo-piano - moderately soft
            'pleased': 'mp',  # Mezzo-piano - moderately soft

            # Neutral spectrum - moderate dynamics
            'neutral': 'mp',  # Mezzo-piano - moderate
            'calm': 'p',  # Piano - soft
            'indifferent': 'mp',  # Mezzo-piano - moderate

            # Sadness spectrum - quieter dynamics, some variation
            'disappointed': 'p',  # Piano - soft
            'sad': 'p',  # Piano - soft
            'sorrowful': 'pp',  # Pianissimo - very soft
            'miserable': 'pp',  # Pianissimo - very soft
            'despairing': 'ppp',  # Pianississimo - extremely soft

            # Anger spectrum - louder, more intense dynamics
            'annoyed': 'mf',  # Mezzo-forte - moderately loud
            'angry': 'f',  # Forte - loud
            'furious': 'ff',  # Fortissimo - very loud

            # Fear spectrum - varied dynamics, often contrasting
            'nervous': 'mp',  # Mezzo-piano with variations
            'anxious': 'mf',  # Mezzo-forte with variations
            'scared': 'f',  # Forte with sudden contrasts
            'terrified': 'ff'  # Fortissimo with extreme contrasts
        }

        # Dynamic mapping to velocity (MIDI velocity)
        self.dynamic_velocity_mapping = {
            'ppp': 30,  # Pianississimo
            'pp': 45,  # Pianissimo
            'p': 60,  # Piano
            'mp': 75,  # Mezzo-piano
            'mf': 90,  # Mezzo-forte
            'f': 105,  # Forte
            'ff': 120  # Fortissimo
        }

        logger.info("Enhanced music generator initialized successfully")

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

    def _get_emotion_interval(self, emotion_name, sentiment_score):
        """Get the musical interval based on specific emotion."""
        # First try to use the extended mapping if emotion is recognized
        if emotion_name in self.extended_interval_mapping:
            return self.extended_interval_mapping[emotion_name]

        # Fallback to the base sentiment-based interval
        return self._get_closest_interval(sentiment_score)

    def _get_emotion_tempo(self, emotion_name, sentiment_score):
        """Get the tempo based on specific emotion."""
        # First try to use the emotion tempo mapping
        if emotion_name in self.emotion_tempo_mapping:
            return self.emotion_tempo_mapping[emotion_name]

        # Fallback to the base sentiment-score tempo calculation
        return self._calculate_tempo(sentiment_score)

    def _get_emotion_rhythm(self, emotion_name):
        """Get rhythm pattern based on emotion."""
        if emotion_name in self.emotion_rhythm_patterns:
            return self.emotion_rhythm_patterns[emotion_name]

        # Default rhythm pattern
        return [1.0]  # Quarter note

    def _get_emotion_mode(self, emotion_name, sentiment_score):
        """Get musical mode based on emotion."""
        if emotion_name in self.emotion_mode_mapping:
            return self.emotion_mode_mapping[emotion_name]

        # Default based on sentiment score
        if sentiment_score >= 0:
            return 'major'
        else:
            return 'minor'

    def _get_emotion_dynamic(self, emotion_name, sentiment_score):
        """Get dynamic (volume) marking based on emotion."""
        if emotion_name in self.emotion_dynamic_mapping:
            return self.emotion_dynamic_mapping[emotion_name]

        # Default based on sentiment score
        if sentiment_score >= 0.5:
            return 'f'  # Forte - loud
        elif sentiment_score >= 0:
            return 'mf'  # Mezzo-forte - moderately loud
        elif sentiment_score >= -0.5:
            return 'mp'  # Mezzo-piano - moderately soft
        else:
            return 'p'  # Piano - soft

    def generate_note_from_emotion(self, word, sentiment_score, emotion_info=None):
        """
        Generate a musical note based on word, sentiment score, and detailed emotion.

        Args:
            word: The word being analyzed
            sentiment_score: Basic sentiment score (-1 to 1)
            emotion_info: Optional dict with detailed emotion information

        Returns:
            A music21 Note object with appropriate pitch and duration
        """
        # Extract emotion name if available, otherwise determine from score
        emotion_name = 'neutral'
        if emotion_info and isinstance(emotion_info, dict) and 'specific_emotion' in emotion_info:
            emotion_name = emotion_info['specific_emotion']
        elif sentiment_score >= 0.75:
            emotion_name = 'elated'
        elif sentiment_score >= 0.5:
            emotion_name = 'happy'
        elif sentiment_score >= 0.25:
            emotion_name = 'pleased'
        elif sentiment_score > -0.25:
            emotion_name = 'neutral'
        elif sentiment_score >= -0.5:
            emotion_name = 'disappointed'
        elif sentiment_score >= -0.75:
            emotion_name = 'sad'
        else:
            emotion_name = 'miserable'

        # Get the interval based on emotion
        interval = self._get_emotion_interval(emotion_name, sentiment_score)

        # Create a new note by adding the interval to the base note
        new_note = interval.transposeNote(self.base_note)

        # Set duration based on emotion
        rhythm_pattern = self._get_emotion_rhythm(emotion_name)
        if rhythm_pattern:
            # Use the first duration in the pattern
            new_note.duration.quarterLength = rhythm_pattern[0]
        else:
            # Default duration based on sentiment
            if sentiment_score > 0:
                new_note.duration.quarterLength = 0.5  # Eighth note for positive
            else:
                new_note.duration.quarterLength = 1.0  # Quarter note for negative

        # Set dynamic (volume) based on emotion
        dynamic = self._get_emotion_dynamic(emotion_name, sentiment_score)
        velocity = self.dynamic_velocity_mapping.get(dynamic, 90)  # Default to mezzo-forte if not found
        new_note.volume.velocity = velocity

        # Add articulation based on emotion
        if emotion_name in ['ecstatic', 'elated', 'happy']:
            # Add staccato for happy emotions
            staccato = music21.articulations.Staccato()
            new_note.articulations.append(staccato)
        elif emotion_name in ['sorrowful', 'miserable', 'despairing']:
            # Add tenuto for sad emotions
            tenuto = music21.articulations.Tenuto()
            new_note.articulations.append(tenuto)
        elif emotion_name in ['angry', 'furious']:
            # Add accent for angry emotions
            accent = music21.articulations.Accent()
            new_note.articulations.append(accent)

        return new_note

    def generate_note_from_sentiment(self, sentiment_score):
        """Generate a musical note based on sentiment score (backward compatibility)."""
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
            word_sentiments: List of tuples (word, sentiment_score) or
                            (word, sentiment_score, emotion_info)

        Returns:
            A music21 Stream object containing the melody
        """
        # Create a music21 stream
        melody = music21.stream.Stream()

        # Check if we have any word sentiments
        if not word_sentiments:
            logger.warning("No word sentiments provided")
            return melody

        # Determine overall emotion
        overall_sentiment = 0.0
        overall_emotion = 'neutral'

        # Calculate overall sentiment
        if word_sentiments:
            # Extract sentiment scores from different possible formats
            scores = []
            emotions = []

            for item in word_sentiments:
                if isinstance(item, tuple):
                    if len(item) >= 2:
                        # (word, score) or (word, score, emotion_info)
                        scores.append(item[1])

                        # Try to extract emotion if available
                        if len(item) >= 3 and isinstance(item[2], dict) and 'specific_emotion' in item[2]:
                            emotions.append(item[2]['specific_emotion'])
                elif isinstance(item, dict) and 'score' in item:
                    # dict format
                    scores.append(item['score'])
                    if 'specific_emotion' in item:
                        emotions.append(item['specific_emotion'])

            # Calculate average sentiment
            if scores:
                overall_sentiment = sum(scores) / len(scores)

            # Determine dominant emotion (if available)
            if emotions:
                from collections import Counter
                emotion_counter = Counter(emotions)
                overall_emotion = emotion_counter.most_common(1)[0][0]
            else:
                # Derive emotion from overall sentiment
                if overall_sentiment >= 0.75:
                    overall_emotion = 'elated'
                elif overall_sentiment >= 0.5:
                    overall_emotion = 'happy'
                elif overall_sentiment >= 0.25:
                    overall_emotion = 'pleased'
                elif overall_sentiment > -0.25:
                    overall_emotion = 'neutral'
                elif overall_sentiment >= -0.5:
                    overall_emotion = 'disappointed'
                elif overall_sentiment >= -0.75:
                    overall_emotion = 'sad'
                else:
                    overall_emotion = 'miserable'

        # Set tempo based on overall emotion
        tempo = self._get_emotion_tempo(overall_emotion, overall_sentiment)
        melody.append(music21.tempo.MetronomeMark(number=tempo))

        # Set time signature based on emotion
        if overall_emotion in ['ecstatic', 'elated', 'happy', 'pleased']:
            # Use 4/4 for positive emotions
            time_sig = music21.meter.TimeSignature('4/4')
        elif overall_emotion in ['neutral', 'calm', 'indifferent']:
            # Use 4/4 for neutral emotions
            time_sig = music21.meter.TimeSignature('4/4')
        elif overall_emotion in ['disappointed', 'sad', 'sorrowful']:
            # Use 3/4 for sad emotions
            time_sig = music21.meter.TimeSignature('3/4')
        elif overall_emotion in ['angry', 'furious', 'annoyed']:
            # Use faster 2/4 for angry emotions
            time_sig = music21.meter.TimeSignature('2/4')
        elif overall_emotion in ['nervous', 'anxious', 'scared', 'terrified']:
            # Use irregular 5/8 for fear emotions
            time_sig = music21.meter.TimeSignature('5/8')
        else:
            # Default to 4/4
            time_sig = music21.meter.TimeSignature('4/4')
        melody.append(time_sig)

        # Set key based on emotion mode
        mode = self._get_emotion_mode(overall_emotion, overall_sentiment)
        if mode == 'major':
            key = music21.key.Key('C')
        elif mode == 'minor':
            key = music21.key.Key('a')
        elif mode == 'dorian':
            key = music21.key.Key('d', 'dorian')
        elif mode == 'phrygian':
            key = music21.key.Key('e', 'phrygian')
        elif mode == 'locrian':
            key = music21.key.Key('b', 'locrian')
        elif mode == 'diminished':
            # Approximation of diminished sound using locrian
            key = music21.key.Key('b', 'locrian')
        else:
            key = music21.key.Key('C')
        melody.append(key)

        # Add notes for each word
        for item in word_sentiments:
            # Extract word and sentiment from different possible formats
            if isinstance(item, tuple):
                if len(item) == 2:
                    # Format: (word, score)
                    word, sentiment = item
                    emotion_info = None
                elif len(item) >= 3:
                    # Format: (word, score, emotion_info)
                    word, sentiment, emotion_info = item
                else:
                    # Invalid format
                    logger.warning(f"Invalid word sentiment format: {item}")
                    continue
            elif isinstance(item, dict) and 'word' in item and 'score' in item:
                # Dictionary format
                word = item['word']
                sentiment = item['score']
                emotion_info = item
            else:
                # Unknown format
                logger.warning(f"Unknown word sentiment format: {item}")
                continue

            # Generate note with enhanced emotion information
            note = self.generate_note_from_emotion(word, sentiment, emotion_info)

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
            logger.error(f"Error playing melody: {e}")
            logger.info("You may need to configure music21 to use a specific MIDI player.")
            logger.info("See: https://web.mit.edu/music21/doc/usersGuide/usersGuide_23_environment.html")

    def create_chord_progression(self, overall_emotion, overall_sentiment):
        """
        Create a chord progression based on overall emotion.

        Args:
            overall_emotion: The dominant emotion
            overall_sentiment: Overall sentiment score (-1 to 1)

        Returns:
            List of music21 chord objects
        """
        # Determine mode based on emotion
        mode = self._get_emotion_mode(overall_emotion, overall_sentiment)

        # Define chord progressions based on mode and emotion
        if mode == 'major':
            if overall_emotion in ['ecstatic', 'elated']:
                # Uplifting major progression
                chord_symbols = ['I', 'IV', 'V', 'I']
            elif overall_emotion in ['happy', 'pleased']:
                # Classic happy progression
                chord_symbols = ['I', 'vi', 'IV', 'V']
            elif overall_emotion in ['content', 'calm']:
                # Warm, relaxed progression
                chord_symbols = ['I', 'iii', 'vi', 'IV']
            else:
                # Default major progression
                chord_symbols = ['I', 'IV', 'V', 'I']
        elif mode == 'minor':
            if overall_emotion in ['disappointed', 'sad']:
                # Melancholic minor progression
                chord_symbols = ['i', 'VI', 'III', 'VII']
            elif overall_emotion in ['sorrowful', 'miserable']:
                # Deeply sad progression
                chord_symbols = ['i', 'iv', 'v', 'i']
            elif overall_emotion in ['angry', 'annoyed']:
                # Tense minor progression
                chord_symbols = ['i', 'VII', 'III', 'v']
            else:
                # Default minor progression
                chord_symbols = ['i', 'iv', 'v', 'i']
        elif mode == 'dorian':
            # Dorian is minor with raised 6th - good for ambiguous emotions
            chord_symbols = ['i', 'IV', 'VII', 'III']
        elif mode == 'phrygian':
            # Phrygian is minor with lowered 2nd - exotic, mysterious
            chord_symbols = ['i', 'II', 'VII', 'i']
        elif mode == 'locrian' or mode == 'diminished':
            # Locrian is diminished - unstable, tense
            chord_symbols = ['i(dim)', 'VII', 'VI', 'iv']
        else:
            # Default to major
            chord_symbols = ['I', 'IV', 'V', 'I']

        # Create chord objects (placeholder)
        # In a complete implementation, this would convert roman numerals to actual chords
        # For now, we'll return the symbols
        return chord_symbols


# Example usage
if __name__ == "__main__":
    # Test the enhanced music generator
    generator = EnhancedMusicGenerator()

    # Example test cases with different emotions
    test_cases = [
        ('I am feeling very happy today!', 'happy'),
        ('I am so excited about the party!', 'elated'),
        ('I feel completely devastated by the news.', 'despairing'),
        ('This makes me somewhat anxious.', 'anxious'),
        ('I am furious about what happened!', 'furious'),
        ('It is a calm, peaceful day.', 'calm')
    ]

    for text, expected_emotion in test_cases:
        print(f"\nTesting with: '{text}' (expected emotion: {expected_emotion})")

        # Create dummy word sentiments
        # In practice these would come from a sentiment analyzer
        word_sentiments = []
        for word in text.split():
            clean_word = ''.join(c for c in word if c.isalnum())
            if clean_word.lower() == expected_emotion:
                score = 0.8 if expected_emotion in ['happy', 'elated', 'excited', 'calm'] else -0.8
            else:
                score = 0.0  # Neutral for other words

            emotion_info = {'specific_emotion': expected_emotion, 'intensity': 0.8}
            word_sentiments.append((clean_word, score, emotion_info))

        # Generate melody
        melody = generator.generate_melody_from_word_sentiments(word_sentiments)

        # Save to file
        filename = f"{expected_emotion}_melody.mid"
        midi_file = generator.save_melody(melody, filename=filename)
        print(f"Melody saved to: {midi_file}")

        # Display chord progression
        chord_progression = generator.create_chord_progression(expected_emotion,
                                                               0.8 if expected_emotion in ['happy', 'elated', 'excited',
                                                                                           'calm'] else -0.8)
        print(f"Chord progression: {chord_progression}")