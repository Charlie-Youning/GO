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

class MultimodalSentimentAnalyzer:
    """Enhanced sentiment analyzer with support for text, speech, and facial expressions."""
    
    def __init__(self):
        """Initialize the multimodal sentiment analyzer."""
        try:
            # Initialize text analyzer
            nltk.download('vader_lexicon', quiet=True)
            self.text_analyzer = SentimentIntensityAnalyzer()
            
            # Initialize speech emotion model
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
            
            # Emotion mapping
            self.emotions = {
                0: 'angry',
                1: 'disgust',
                2: 'fear',
                3: 'happy',
                4: 'neutral',
                5: 'sad',
                6: 'surprise'
            }
            
            logger.info("Multimodal sentiment analyzer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing multimodal sentiment analyzer: {e}")
            raise
    
    def analyze_text(self, text):
        """
        Analyze the sentiment of the given text.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            dict: A dictionary containing sentiment scores and enhanced features
        """
        try:
            if not text:
                return {
                    'compound': 0.0,
                    'pos': 0.0,
                    'neu': 1.0,
                    'neg': 0.0,
                    'emotion': 'neutral',
                    'confidence': 1.0
                }
            
            # Get VADER sentiment scores
            scores = self.text_analyzer.polarity_scores(text)
            
            # Determine dominant emotion
            compound = scores['compound']
            if compound >= 0.5:
                emotion = 'happy'
                confidence = min(abs(compound), 1.0)
            elif compound <= -0.5:
                emotion = 'sad'
                confidence = min(abs(compound), 1.0)
            else:
                emotion = 'neutral'
                confidence = 1.0 - abs(compound)
            
            return {
                'compound': float(scores.get('compound', 0.0)),
                'pos': float(scores.get('pos', 0.0)),
                'neu': float(scores.get('neu', 1.0)),
                'neg': float(scores.get('neg', 0.0)),
                'emotion': emotion,
                'confidence': confidence
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {e}")
            return {
                'compound': 0.0,
                'pos': 0.0,
                'neu': 1.0,
                'neg': 0.0,
                'emotion': 'neutral',
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
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': {
                    self.emotions[i]: float(prob)
                    for i, prob in enumerate(output[0])
                }
            }
        except Exception as e:
            logger.error(f"Error analyzing speech: {e}")
            return {
                'emotion': 'neutral',
                'confidence': 1.0,
                'probabilities': {emotion: 0.0 for emotion in self.emotions.values()}
            }
    
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
            'confidence': 1.0
        }
        
        try:
            # Analyze text if provided
            if text:
                results['text_sentiment'] = self.analyze_text(text)
            
            # Analyze speech if provided
            if audio_path:
                results['speech_emotion'] = self.analyze_speech(audio_path)
            
            # Combine results (weighted average)
            emotions = []
            weights = []
            
            if results['text_sentiment']:
                emotions.append(results['text_sentiment']['emotion'])
                weights.append(results['text_sentiment']['confidence'])
            
            if results['speech_emotion']:
                emotions.append(results['speech_emotion']['emotion'])
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
        Analyze the sentiment of individual words in the text.
        
        Args:
            text (str): The text to analyze.
            
        Returns:
            list: A list of tuples (word, sentiment_score) for each word.
        """
        try:
            if not text:
                return []
            
            # Split text into words
            words = text.split()
            
            # Analyze each word
            word_sentiments = []
            for word in words:
                # Clean the word
                clean_word = ''.join(c for c in word if c.isalnum() or c.isspace())
                if clean_word:
                    # Get sentiment scores
                    scores = self.text_analyzer.polarity_scores(clean_word)
                    
                    # Get emotion
                    compound = scores['compound']
                    if compound >= 0.5:
                        emotion = 'happy'
                    elif compound <= -0.5:
                        emotion = 'sad'
                    else:
                        emotion = 'neutral'
                    
                    word_sentiments.append({
                        'word': clean_word,
                        'score': float(compound),
                        'emotion': emotion
                    })
            
            return word_sentiments
        except Exception as e:
            logger.error(f"Error analyzing words: {e}")
            return []

# Example usage
if __name__ == "__main__":
    analyzer = MultimodalSentimentAnalyzer()
    
    # Test with a sample text
    sample_text = "I love this beautiful day. The weather is nice and I feel happy."
    result = analyzer.analyze_text(sample_text)
    print(f"Overall sentiment: {result}")
    
    # Test word-level analysis
    word_sentiments = analyzer.analyze_words(sample_text)
    print("\nWord-level sentiments:")
    for word_info in word_sentiments:
        print(f"{word_info['word']}: {word_info['score']:.4f} ({word_info['emotion']})") 