# utils.py
import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess an image for model prediction.
    
    Args:
        image: PIL Image object
        target_size: Tuple of (height, width) to resize the image
        
    Returns:
        Preprocessed image as numpy array
    """
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Handle images with alpha channel
    if image_array.shape[-1] == 4:
        image_array = image_array[..., :3]
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Define skin disease classes
SKIN_CLASSES = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis',
    'Dermatofibroma',
    'Melanoma',
    'Melanocytic nevi',
    'Vascular lesions'
]

# Information about each skin condition
CONDITION_INFO = {
    'Actinic keratoses': "Rough, scaly patches caused by years of sun exposure. Early form of skin cancer.",
    'Basal cell carcinoma': "Most common type of skin cancer. Usually appears as a slightly transparent bump.",
    'Benign keratosis': "Non-cancerous skin growths that appear as waxy, stuck-on-the-skin growths.",
    'Dermatofibroma': "Common, harmless skin growths that often appear as small, firm bumps.",
    'Melanoma': "The most serious type of skin cancer. Develops in cells that produce melanin.",
    'Melanocytic nevi': "Common moles. Usually harmless growths on the skin.",
    'Vascular lesions': "Abnormalities of blood vessels, including hemangiomas and port-wine stains."
}