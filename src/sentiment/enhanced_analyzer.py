"""
Enhanced sentiment analyzer with more granular emotion detection.
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import logging
import numpy as np
from pathlib import Path
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmotionCNN(nn.Module):
    """CNN model for speech emotion recognition."""

    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 7)  # 7 emotions
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)


class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer with more granular emotion detection."""

    def __init__(self):
        """Initialize the enhanced sentiment analyzer."""
        try:
            # Initialize text analyzer
            nltk.download('vader_lexicon', quiet=True)
            self.text_analyzer = SentimentIntensityAnalyzer()

            # Initialize speech emotion model if available
            try:
                self.speech_model = EmotionCNN()
                model_path = Path(__file__).parent / 'models' / 'speech_emotion.pth'

                # Create models directory if it doesn't exist
                model_path.parent.mkdir(parents=True, exist_ok=True)

                if model_path.exists():
                    self.speech_model.load_state_dict(torch.load(model_path))
                    logger.info("Loaded speech emotion model from file")
                else:
                    logger.warning(f"Speech emotion model not found at {model_path}. Using default initialization.")

                self.speech_model.eval()
            except Exception as e:
                logger.warning(f"Speech emotion model initialization failed: {e}")
                self.speech_model = None

            # Basic emotion mapping
            self.emotions = {
                0: 'angry',
                1: 'disgust',
                2: 'fear',
                3: 'happy',
                4: 'neutral',
                5: 'sad',
                6: 'surprise'
            }

            # Extended emotion mappings for more granular analysis
            self.emotion_intensity_map = {
                # Joy spectrum
                'ecstatic': 1.0,  # Highest joy (e.g., "thrilled", "overjoyed")
                'elated': 0.85,  # Very high joy (e.g., "delighted", "excited")
                'happy': 0.7,  # Standard happiness (e.g., "happy", "glad")
                'content': 0.5,  # Mild happiness (e.g., "satisfied", "pleased")
                'pleased': 0.35,  # Slight happiness (e.g., "nice", "good")

                # Neutral spectrum
                'neutral': 0.0,  # Neutral (e.g., "normal", "okay")
                'calm': 0.1,  # Slightly positive neutral (e.g., "peaceful")
                'indifferent': -0.1,  # Slightly negative neutral (e.g., "meh")

                # Sadness spectrum
                'disappointed': -0.35,  # Slight sadness (e.g., "letdown", "unhappy")
                'sad': -0.5,  # Standard sadness (e.g., "sad", "down")
                'sorrowful': -0.7,  # Deep sadness (e.g., "grieving", "distressed")
                'miserable': -0.85,  # Very high sadness (e.g., "depressed", "devastated")
                'despairing': -1.0,  # Highest sadness (e.g., "hopeless", "heartbroken")

                # Anger spectrum
                'annoyed': -0.4,  # Slight anger (e.g., "irritated", "bothered")
                'angry': -0.6,  # Standard anger (e.g., "mad", "angry")
                'furious': -0.8,  # High anger (e.g., "outraged", "livid")

                # Fear spectrum
                'nervous': -0.3,  # Slight fear (e.g., "uneasy", "concerned")
                'anxious': -0.5,  # Moderate fear (e.g., "worried", "stressed")
                'scared': -0.7,  # High fear (e.g., "afraid", "frightened")
                'terrified': -0.9  # Extreme fear (e.g., "petrified", "horrified")
            }

            # Dictionary of emotion intensifiers
            self.intensifiers = {
                # Positive intensifiers
                'very': 0.2,
                'really': 0.2,
                'extremely': 0.3,
                'incredibly': 0.3,
                'absolutely': 0.3,
                'so': 0.2,
                'totally': 0.25,
                'completely': 0.25,
                'exceedingly': 0.3,
                'exceptionally': 0.3,
                'immensely': 0.3,
                'particularly': 0.15,
                'especially': 0.2,
                'remarkably': 0.2,
                'truly': 0.2,
                'highly': 0.2,
                'super': 0.25,
                'greatly': 0.2,
                'deeply': 0.25,

                # Negative intensifiers
                'barely': -0.2,
                'hardly': -0.2,
                'slightly': -0.15,
                'somewhat': -0.1,
                'a bit': -0.1,
                'a little': -0.1,
                'rather': -0.05,
                'quite': 0.1  # "quite" can be positive or somewhat negative depending on context
            }

            # Keywords for specific emotions
            self.emotion_keywords = {
                # Joy keywords
                'happy': 'happy',
                'glad': 'happy',
                'joy': 'happy',
                'delighted': 'elated',
                'thrilled': 'ecstatic',
                'excited': 'elated',
                'overjoyed': 'ecstatic',
                'ecstatic': 'ecstatic',
                'content': 'content',
                'satisfied': 'content',
                'pleased': 'pleased',
                'cheerful': 'happy',
                'wonderful': 'elated',
                'great': 'happy',
                'excellent': 'elated',
                'fantastic': 'elated',
                'amazing': 'elated',
                'love': 'elated',
                'loving': 'happy',
                'enjoy': 'happy',

                # Neutral keywords
                'neutral': 'neutral',
                'okay': 'neutral',
                'fine': 'neutral',
                'alright': 'neutral',
                'average': 'neutral',
                'calm': 'calm',
                'peaceful': 'calm',
                'indifferent': 'indifferent',
                'meh': 'indifferent',
                'ordinary': 'neutral',

                # Sadness keywords
                'sad': 'sad',
                'unhappy': 'disappointed',
                'disappointed': 'disappointed',
                'upset': 'sad',
                'depressed': 'miserable',
                'miserable': 'miserable',
                'devastated': 'miserable',
                'heartbroken': 'despairing',
                'gloomy': 'sad',
                'hopeless': 'despairing',
                'despairing': 'despairing',
                'grief': 'sorrowful',
                'grieving': 'sorrowful',
                'sorrow': 'sorrowful',
                'sorrowful': 'sorrowful',
                'melancholy': 'sad',

                # Anger keywords
                'angry': 'angry',
                'annoyed': 'annoyed',
                'irritated': 'annoyed',
                'bothered': 'annoyed',
                'mad': 'angry',
                'furious': 'furious',
                'outraged': 'furious',
                'livid': 'furious',

                # Fear keywords
                'nervous': 'nervous',
                'anxious': 'anxious',
                'worried': 'anxious',
                'scared': 'scared',
                'afraid': 'scared',
                'frightened': 'scared',
                'terrified': 'terrified',
                'horrified': 'terrified',
                'petrified': 'terrified'
            }

            logger.info("Enhanced sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing enhanced sentiment analyzer: {e}")
            raise

    def analyze_text(self, text):
        """
        Analyze the sentiment of the given text with more granular emotion recognition.

        Args:
            text (str): The text to analyze

        Returns:
            dict: A dictionary containing enhanced sentiment scores and emotion details
        """
        try:
            if not text:
                return {
                    'compound': 0.0,
                    'pos': 0.0,
                    'neu': 1.0,
                    'neg': 0.0,
                    'primary_emotion': 'neutral',
                    'emotion_intensity': 0.0,
                    'emotion_spectrum': [],
                    'confidence': 1.0
                }

            # Get VADER sentiment scores
            scores = self.text_analyzer.polarity_scores(text)

            # Enhanced analysis with more granular emotion detection
            words = text.lower().split()

            # Initialize enhanced fields
            results = {
                'compound': float(scores.get('compound', 0.0)),
                'pos': float(scores.get('pos', 0.0)),
                'neu': float(scores.get('neu', 1.0)),
                'neg': float(scores.get('neg', 0.0)),
                'primary_emotion': 'neutral',
                'emotion_intensity': 0.0,
                'emotion_spectrum': [],
                'confidence': 1.0
            }

            # Look for emotional keywords in the text
            detected_emotions = {}

            # First pass: find emotion keywords
            for i, word in enumerate(words):
                # Clean word (remove punctuation)
                clean_word = ''.join(c for c in word if c.isalnum())

                # Skip empty words after cleaning
                if not clean_word:
                    continue

                # Check if this word is an emotion keyword
                if clean_word in self.emotion_keywords:
                    emotion_name = self.emotion_keywords[clean_word]
                    emotion_value = self.emotion_intensity_map.get(emotion_name, 0.0)

                    # Check for intensifiers before this emotion word
                    if i > 0:
                        prev_word = words[i - 1].lower().strip('.,!?;:')
                        if prev_word in self.intensifiers:
                            intensifier = self.intensifiers[prev_word]
                            # For negative emotions, a positive intensifier makes it more negative
                            if emotion_value < 0:
                                emotion_value -= intensifier
                            else:
                                emotion_value += intensifier

                            # Clamp value to valid range
                            emotion_value = max(-1.0, min(1.0, emotion_value))

                    # Add or update emotion
                    if emotion_name in detected_emotions:
                        detected_emotions[emotion_name] = max(detected_emotions[emotion_name], emotion_value)
                    else:
                        detected_emotions[emotion_name] = emotion_value

            # If no specific emotions were detected, use the compound score to assign a general emotion
            if not detected_emotions:
                compound = scores['compound']

                if compound >= 0.75:
                    results['primary_emotion'] = 'elated'
                    results['emotion_intensity'] = abs(compound)
                elif compound >= 0.5:
                    results['primary_emotion'] = 'happy'
                    results['emotion_intensity'] = abs(compound)
                elif compound >= 0.25:
                    results['primary_emotion'] = 'pleased'
                    results['emotion_intensity'] = abs(compound)
                elif compound > -0.25:
                    if compound >= 0:
                        results['primary_emotion'] = 'neutral'
                        results['emotion_intensity'] = 0.1
                    else:
                        results['primary_emotion'] = 'indifferent'
                        results['emotion_intensity'] = 0.1
                elif compound >= -0.5:
                    results['primary_emotion'] = 'disappointed'
                    results['emotion_intensity'] = abs(compound)
                elif compound >= -0.75:
                    results['primary_emotion'] = 'sad'
                    results['emotion_intensity'] = abs(compound)
                else:
                    results['primary_emotion'] = 'miserable'
                    results['emotion_intensity'] = abs(compound)
            else:
                # Find the dominant emotion (highest absolute intensity)
                dominant_emotion = max(detected_emotions.items(), key=lambda x: abs(x[1]))
                results['primary_emotion'] = dominant_emotion[0]
                results['emotion_intensity'] = abs(dominant_emotion[1])

                # Create the emotion spectrum - all detected emotions sorted by intensity
                emotion_spectrum = [
                    {'emotion': emotion, 'intensity': abs(value), 'valence': value}
                    for emotion, value in sorted(detected_emotions.items(), key=lambda x: abs(x[1]), reverse=True)
                ]
                results['emotion_spectrum'] = emotion_spectrum

            # Set confidence based on emotion intensity
            results['confidence'] = min(results['emotion_intensity'] * 1.5, 1.0)

            return results
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
                'primary_emotion': 'neutral',
                'emotion_intensity': 0.0,
                'emotion_spectrum': [],
                'confidence': 1.0
            }

    def analyze_speech(self, audio_path):
        """
        Analyze emotion from speech audio.

        Args:
            audio_path (str): Path to the audio file

        Returns:
            dict: Emotion prediction results
        """
        try:
            if self.speech_model is None:
                logger.warning("Speech model not available for analysis")
                return {
                    'emotion': 'neutral',
                    'confidence': 1.0,
                    'probabilities': {emotion: 0.0 for emotion in self.emotions.values()}
                }

            # Load and preprocess audio
            y, sr = librosa.load(audio_path, duration=3)
            mel_spect = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spect = librosa.power_to_db(mel_spect, ref=np.max)

            # Prepare input tensor
            mel_spect = torch.FloatTensor(mel_spect).unsqueeze(0).unsqueeze(0)

            # Get prediction
            with torch.no_grad():
                output = self.speech_model(mel_spect)
                pred_idx = output.argmax(dim=1).item()
                confidence = output[0][pred_idx].item()
                emotion = self.emotions[pred_idx]

            # Map to more specific emotion based on confidence
            specific_emotion = self._map_basic_to_specific_emotion(emotion, confidence)

            return {
                'basic_emotion': emotion,
                'specific_emotion': specific_emotion,
                'confidence': confidence,
                'probabilities': {
                    self.emotions[i]: float(prob)
                    for i, prob in enumerate(output[0])
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing speech: {e}")
            return {
                'basic_emotion': 'neutral',
                'specific_emotion': 'neutral',
                'confidence': 1.0,
                'probabilities': {emotion: 0.0 for emotion in self.emotions.values()}
            }

    def _map_basic_to_specific_emotion(self, basic_emotion, confidence):
        """Map a basic emotion to a more specific one based on confidence level."""
        if basic_emotion == 'happy':
            if confidence > 0.9:
                return 'ecstatic'
            elif confidence > 0.7:
                return 'elated'
            elif confidence > 0.5:
                return 'happy'
            else:
                return 'pleased'
        elif basic_emotion == 'sad':
            if confidence > 0.9:
                return 'despairing'
            elif confidence > 0.7:
                return 'sorrowful'
            elif confidence > 0.5:
                return 'sad'
            else:
                return 'disappointed'
        elif basic_emotion == 'angry':
            if confidence > 0.8:
                return 'furious'
            elif confidence > 0.5:
                return 'angry'
            else:
                return 'annoyed'
        elif basic_emotion == 'fear':
            if confidence > 0.8:
                return 'terrified'
            elif confidence > 0.5:
                return 'scared'
            else:
                return 'anxious'
        elif basic_emotion == 'neutral':
            return 'neutral'
        elif basic_emotion == 'surprise':
            return 'elated' if confidence > 0.5 else 'pleased'
        elif basic_emotion == 'disgust':
            return 'annoyed'
        else:
            return 'neutral'

    def analyze_multimodal(self, text=None, audio_path=None, face_image=None):
        """
        Perform multimodal sentiment analysis.

        Args:
            text (str, optional): Text to analyze
            audio_path (str, optional): Path to audio file
            face_image (numpy.ndarray, optional): Face image array

        Returns:
            dict: Combined analysis results
        """
        results = {
            'text_sentiment': None,
            'speech_emotion': None,
            'face_emotion': None,
            'combined_emotion': 'neutral',
            'primary_emotion': 'neutral',
            'emotion_intensity': 0.0,
            'emotion_spectrum': [],
            'confidence': 1.0
        }

        try:
            # Analyze text if provided
            if text:
                results['text_sentiment'] = self.analyze_text(text)
                results['primary_emotion'] = results['text_sentiment']['primary_emotion']
                results['emotion_intensity'] = results['text_sentiment']['emotion_intensity']
                if 'emotion_spectrum' in results['text_sentiment']:
                    results['emotion_spectrum'] = results['text_sentiment']['emotion_spectrum']

            # Analyze speech if provided
            if audio_path:
                results['speech_emotion'] = self.analyze_speech(audio_path)

            # Combine results (weighted average)
            emotions = []
            weights = []

            if results['text_sentiment']:
                emotions.append(results['text_sentiment']['primary_emotion'])
                weights.append(results['text_sentiment']['confidence'])

            if results['speech_emotion']:
                emotions.append(results['speech_emotion']['specific_emotion'])
                weights.append(results['speech_emotion']['confidence'])

            if emotions:
                # Simple majority voting for now
                from collections import Counter
                emotion_counts = Counter(emotions)
                results['combined_emotion'] = emotion_counts.most_common(1)[0][0]
                results['confidence'] = sum(weights) / len(weights)

            return results

        except Exception as e:
            logger.error(f"Error in multimodal analysis: {e}")
            return results

    def analyze_words(self, text):
        """
        Analyze the sentiment of individual words in the text with enhanced emotion detection.

        Args:
            text (str): The text to analyze.

        Returns:
            list: A list of tuples (word, sentiment_score) for each word with enhanced emotion info
        """
        try:
            if not text:
                return []

            # Split text into words
            words = text.split()

            # Analyze each word
            word_sentiments = []

            for i, word in enumerate(words):
                # Clean the word
                clean_word = ''.join(c for c in word if c.isalnum() or c.isspace())
                if not clean_word:
                    continue

                # Get sentiment scores
                scores = self.text_analyzer.polarity_scores(clean_word)
                compound = scores['compound']

                # Determine more specific emotion based on word and context
                specific_emotion = self._determine_specific_emotion(clean_word, compound, i, words)

                # Adjust intensity based on context
                adjusted_intensity = self._adjust_intensity_from_context(i, words, compound)

                # Create word sentiment result
                word_sentiments.append((
                    clean_word,
                    compound
                ))

            return word_sentiments
        except Exception as e:
            logger.error(f"Error analyzing words: {e}")
            return []

    def _determine_specific_emotion(self, word, score, position, all_words):
        """Determine a specific emotion for a word based on score, keyword matching, and context."""
        # First check if the word itself is an emotion keyword
        word_lower = word.lower()
        if word_lower in self.emotion_keywords:
            return self.emotion_keywords[word_lower]

        # Check for intensifiers before this word
        intensifier_value = 0.0
        if position > 0:
            prev_word = all_words[position - 1].lower().strip('.,!?;:')
            if prev_word in self.intensifiers:
                intensifier_value = self.intensifiers[prev_word]

        # Adjust score with intensifier
        adjusted_score = score
        if score < 0:
            adjusted_score -= intensifier_value  # More negative for negative emotions
        else:
            adjusted_score += intensifier_value  # More positive for positive emotions

        # Clamp to valid range
        adjusted_score = max(-1.0, min(1.0, adjusted_score))

        # Map to specific emotion based on adjusted score
        if adjusted_score >= 0.75:
            return 'elated'
        elif adjusted_score >= 0.5:
            return 'happy'
        elif adjusted_score >= 0.25:
            return 'pleased'
        elif adjusted_score > -0.25:
            return 'neutral'
        elif adjusted_score >= -0.5:
            return 'disappointed'
        elif adjusted_score >= -0.75:
            return 'sad'
        else:
            return 'miserable'

    def _adjust_intensity_from_context(self, position, all_words, base_score):
        """Adjust the emotional intensity based on surrounding context."""
        intensity = abs(base_score)

        # Check for intensifiers before this word
        if position > 0:
            prev_word = all_words[position - 1].lower().strip('.,!?;:')
            if prev_word in self.intensifiers:
                intensity += abs(self.intensifiers[prev_word])

        # Check for additional modifiers in surrounding context (up to 2 words before)
        for i in range(max(0, position - 2), position):
            if i == position - 1:  # Already checked the immediate previous word
                continue

            context_word = all_words[i].lower().strip('.,!?;:')
            if context_word in self.intensifiers:
                intensity += abs(self.intensifiers[context_word]) * 0.5  # Half effect for words further away

        # Clamp to valid range
        return min(intensity, 1.0)


# Example usage
if __name__ == "__main__":
    analyzer = EnhancedSentimentAnalyzer()

    # Test with various sample texts
    sample_texts = [
        "I am happy today.",
        "I am very happy today.",
        "I am extremely happy today.",
        "I am so incredibly happy today!",
        "I feel sad.",
        "I feel very sad.",
        "I am completely devastated.",
        "I'm a bit disappointed.",
        "This is wonderful! I'm thrilled with the results.",
        "This is okay, nothing special."
    ]

    print("Testing enhanced sentiment analysis:")
    for text in sample_texts:
        result = analyzer.analyze_text(text)
        print(f"\nText: '{text}'")
        print(f"Primary emotion: {result['primary_emotion']} (intensity: {result['emotion_intensity']:.2f})")
        print(f"Compound score: {result['compound']:.2f}")

        if 'emotion_spectrum' in result and result['emotion_spectrum']:
            print("Emotion spectrum:")
            for emotion in result['emotion_spectrum']:
                print(f"  {emotion['emotion']}: {emotion['intensity']:.2f}")