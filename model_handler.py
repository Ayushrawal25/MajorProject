import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

class SkinDiseaseModel:
    def __init__(self):
        """Initialize the skin disease detection model"""
        self.model = None
        self.processor = None
        self.class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        self.loaded = False
        
    def load_model(self):
        """Load the pre-trained model from Hugging Face"""
        try:
            print("Loading model...")
            self.processor = AutoImageProcessor.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
            self.model = AutoModelForImageClassification.from_pretrained("Anwarkh1/Skin_Cancer-Image_Classification")
            self.loaded = True
            print("Model loaded successfully!")
            
            # Update class names from the model's config if available
            if hasattr(self.model.config, 'id2label'):
                self.class_names = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
                
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def predict(self, image):
        """
        Make a prediction on the input image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with prediction results
        """
        if not self.loaded:
            success = self.load_model()
            if not success:
                return {"error": "Failed to load model"}
        
        try:
            # Process the image using the model's processor
            inputs = self.processor(images=image, return_tensors="pt")
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convert to probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1).numpy()[0]
            
            # Get the predicted class
            predicted_class_idx = np.argmax(probabilities)
            predicted_class = self.class_names[predicted_class_idx]
            
            return {
                "predicted_class": predicted_class,
                "probabilities": probabilities,
                "class_names": self.class_names
            }
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": str(e)}