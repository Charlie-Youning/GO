import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FacialExpressionDetector:
    """Basic facial expression detector using OpenCV."""
    
    def __init__(self):
        """Initialize the facial expression detector."""
        try:
            # Load the pre-trained face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Check if the cascade classifier was loaded successfully
            if self.face_cascade.empty():
                logger.error("Error loading face cascade classifier.")
                raise ValueError("Failed to load face cascade classifier.")
            
            logger.info("Facial expression detector initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing facial expression detector: {e}")
            raise
    
    def detect_faces(self, frame):
        """
        Detect faces in the given frame.
        
        Args:
            frame: OpenCV image frame (numpy array)
            
        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        if frame is None:
            logger.warning("Empty frame provided for face detection.")
            return []
        
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            logger.debug(f"Detected {len(faces)} faces.")
            return faces
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def analyze_expression(self, frame, face_rect=None):
        """
        Analyze facial expression in the given frame.
        
        Args:
            frame: OpenCV image frame (numpy array)
            face_rect: Optional (x, y, w, h) tuple for a detected face
            
        Returns:
            dict: A dictionary containing expression scores:
                - happy: Score for happiness (0 to 1)
                - sad: Score for sadness (0 to 1)
                - angry: Score for anger (0 to 1)
                - neutral: Score for neutral expression (0 to 1)
        """
        if frame is None:
            logger.warning("Empty frame provided for expression analysis.")
            return {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 1}
        
        try:
            # If no face_rect is provided, detect faces
            if face_rect is None:
                faces = self.detect_faces(frame)
                if len(faces) == 0:
                    logger.warning("No faces detected for expression analysis.")
                    return {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 1}
                
                # Use the first detected face
                face_rect = faces[0]
            
            # Extract the face region
            x, y, w, h = face_rect
            face_roi = frame[y:y+h, x:x+w]
            
            # In a real application, you would use a pre-trained model for expression recognition
            # For this example, we'll return random values
            # This is a placeholder for actual expression analysis
            
            # Simulate expression analysis with random values
            import random
            happy = random.random() * 0.5
            sad = random.random() * 0.3
            angry = random.random() * 0.2
            
            # Ensure the sum is 1.0
            total = happy + sad + angry
            neutral = 1.0 - total
            
            return {
                'happy': happy,
                'sad': sad,
                'angry': angry,
                'neutral': neutral
            }
        except Exception as e:
            logger.error(f"Error analyzing facial expression: {e}")
            return {'happy': 0, 'sad': 0, 'angry': 0, 'neutral': 1}
    
    def map_expression_to_sentiment(self, expression):
        """
        Map facial expression scores to a sentiment score.
        
        Args:
            expression: Dictionary of expression scores
            
        Returns:
            float: Sentiment score (-1 to 1)
        """
        try:
            # Simple mapping:
            # happy -> positive
            # sad, angry -> negative
            # neutral -> neutral
            
            positive = expression['happy']
            negative = expression['sad'] + expression['angry']
            
            # Calculate sentiment score in range [-1, 1]
            sentiment = positive - negative
            
            return sentiment
        except Exception as e:
            logger.error(f"Error mapping expression to sentiment: {e}")
            return 0.0

# Example usage
if __name__ == "__main__":
    detector = FacialExpressionDetector()
    
    # Open a video capture (0 for webcam)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()
    
    print("Press 'q' to quit.")
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Detect faces
        faces = detector.detect_faces(frame)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Analyze expression
            expression = detector.analyze_expression(frame, (x, y, w, h))
            sentiment = detector.map_expression_to_sentiment(expression)
            
            # Display sentiment score
            sentiment_text = f"Sentiment: {sentiment:.2f}"
            cv2.putText(frame, sentiment_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Facial Expression Analysis', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows() 