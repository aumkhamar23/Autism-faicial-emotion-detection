from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

# Paths
DATA_RAW       = BASE_DIR / 'data' / 'raw'
DATA_PROCESSED = BASE_DIR / 'data' / 'processed'
MODEL_DIR      = BASE_DIR / 'models'
MODEL_PATH     = MODEL_DIR / 'custom_emotion_model.keras'

# Image settings
IMG_SIZE     = 48        # pixels, square — FER2013 native size
IMG_CHANNELS = 1         # grayscale — FER2013 native format, 3x fewer input params

# Model backend: "custom_cnn" (no download needed, fast CPU training)
#                "efficientnet" (requires internet to download ImageNet weights)
MODEL_TYPE  = "custom_cnn"

# Class definitions — order is the label index used in training
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
NUM_CLASSES = len(EMOTIONS)

# Balancing thresholds
MAX_PER_CLASS = 3500    # cap dominant classes (happy, neutral, sad) here
MIN_PER_CLASS = 1500    # augment minority classes (disgust, fear) up to here

# Train / val / test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# test gets the remainder (0.15)

RANDOM_SEED = 42

# Raw dataset folder names and their class-name maps → unified EMOTIONS labels
# Value None = skip that class
DATASET_SOURCES = {
    'fer2013': {
        'root': DATA_RAW / 'fer2013',
        'splits': ['train', 'test'],   # sub-folders to pool
        'label_map': {
            'angry':    'angry',
            'disgust':  'disgust',
            'fear':     'fear',
            'happy':    'happy',
            'neutral':  'neutral',
            'sad':      'sad',
            'surprise': 'surprise',
        },
    },
    'ckplus': {
        'root': DATA_RAW / 'ckplus' / 'CK+48',
        'splits': None,                # images sit directly in emotion sub-folders
        'label_map': {
            'anger':    'angry',
            'disgust':  'disgust',
            'fear':     'fear',
            'happy':    'happy',
            'sadness':  'sad',
            'surprise': 'surprise',
            'contempt': None,          # no contempt class in our model
        },
    },
}
