import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Disease class names for the HAM10000 dataset
DISEASE_CLASSES = {
    'akiec': 'Actinic Keratoses and Intraepithelial Carcinoma',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model input
    
    Args:
        image: PIL Image or file bytes
        target_size: Size to resize the image to
        
    Returns:
        Preprocessed image as numpy array
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image))
    
    # Resize image
    image = image.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def plot_prediction_probabilities(probabilities, class_names):
    """
    Create a bar chart of prediction probabilities
    
    Args:
        probabilities: Array of prediction probabilities
        class_names: List of class names
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(class_names))
    
    # Sort probabilities and class names by probability (descending)
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    bars = ax.barh(y_pos, sorted_probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_classes)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel('Probability')
    ax.set_title('Prediction Probabilities')
    
    # Add probability values as text
    for i, bar in enumerate(bars):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{sorted_probs[i]:.4f}', va='center')
    
    plt.tight_layout()
    return fig

def get_disease_info(disease_code):
    """
    Get detailed information about a skin disease
    
    Args:
        disease_code: Short code for the disease
        
    Returns:
        Dictionary with disease information
    """
    info = {
        'akiec': {
            'name': DISEASE_CLASSES['akiec'],
            'description': 'Actinic Keratoses are pre-cancerous lesions caused by sun damage. Intraepithelial Carcinoma is an early stage of squamous cell carcinoma.',
            'risk_factors': 'Sun exposure, fair skin, age over 40',
            'treatment': 'Cryotherapy, topical medications, photodynamic therapy, or surgical removal'
        },
        'bcc': {
            'name': DISEASE_CLASSES['bcc'],
            'description': 'Basal Cell Carcinoma is the most common form of skin cancer. It rarely metastasizes but can cause significant local damage if left untreated.',
            'risk_factors': 'Sun exposure, fair skin, radiation therapy, chronic skin inflammation',
            'treatment': 'Surgical excision, Mohs surgery, radiation therapy, or topical medications'
        },
        'bkl': {
            'name': DISEASE_CLASSES['bkl'],
            'description': 'Benign Keratosis-like Lesions include seborrheic keratoses and solar lentigo. These are non-cancerous growths that appear with age.',
            'risk_factors': 'Age, sun exposure, genetic factors',
            'treatment': 'Usually no treatment needed; can be removed for cosmetic reasons'
        },
        'df': {
            'name': DISEASE_CLASSES['df'],
            'description': 'Dermatofibroma is a common benign skin tumor that often appears as a hard lump under the skin.',
            'risk_factors': 'Minor trauma, insect bites, may be more common in women',
            'treatment': 'Usually no treatment needed; surgical excision if problematic'
        },
        'mel': {
            'name': DISEASE_CLASSES['mel'],
            'description': 'Melanoma is the most dangerous form of skin cancer. It can spread to other parts of the body if not detected early.',
            'risk_factors': 'Sun exposure, fair skin, family history, multiple moles',
            'treatment': 'Surgical excision, immunotherapy, targeted therapy, radiation, or chemotherapy depending on stage'
        },
        'nv': {
            'name': DISEASE_CLASSES['nv'],
            'description': 'Melanocytic Nevi are common moles. Most are benign, but they should be monitored for changes that might indicate melanoma.',
            'risk_factors': 'Genetic factors, sun exposure',
            'treatment': 'Usually no treatment needed; removal if suspicious'
        },
        'vasc': {
            'name': DISEASE_CLASSES['vasc'],
            'description': 'Vascular Lesions include hemangiomas, angiomas, and pyogenic granulomas. These are benign growths made up of blood vessels.',
            'risk_factors': 'Some are congenital, others may develop after injury',
            'treatment': 'Laser therapy, surgical removal, or observation depending on type and location'
        }
    }
    
    return info.get(disease_code, {})