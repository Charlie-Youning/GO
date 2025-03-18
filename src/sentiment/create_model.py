import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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

def create_dummy_model():
    """Create a dummy speech emotion recognition model."""
    try:
        # Initialize the model
        model = EmotionCNN()
        
        # Save the model
        model_dir = Path(__file__).parent / 'models'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / 'speech_emotion.pth'
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"Dummy speech emotion model saved to {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error creating dummy model: {e}")
        return False

if __name__ == "__main__":
    create_dummy_model() 