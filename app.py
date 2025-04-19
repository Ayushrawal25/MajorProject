import streamlit as st
from PIL import Image
import io
import numpy as np
import time

from model_handler import SkinDiseaseModel
from utils import plot_prediction_probabilities, get_disease_info

# Set page configuration
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🔬",
    layout="wide"
)

# Initialize the model
@st.cache_resource
def load_model():
    model = SkinDiseaseModel()
    model.load_model()
    return model

# Main function
def main():
    # Header
    st.title("Skin Disease Detection System")
    st.markdown("""
    This application uses a machine learning model to detect and classify skin diseases from images.
    Upload a clear image of the skin condition to get a prediction.
    
    **Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.
    """)
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application demonstrates the use of a Vision Transformer (ViT) model "
        "for skin disease classification. The model was trained on the HAM10000 dataset "
        "and can identify 7 different types of skin conditions."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.markdown("""
    1. Upload an image of the skin condition
    2. Wait for the model to process the image
    3. View the prediction results and information
    """)
    
    # Load model
    with st.spinner("Loading model... This may take a moment."):
        model = load_model()
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Process uploaded image
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            # Add a small delay to show the spinner
            time.sleep(1)
            
            # Get prediction
            results = model.predict(image)
            
            if "error" in results:
                st.error(f"Error during prediction: {results['error']}")
            else:
                predicted_class = results["predicted_class"]
                probabilities = results["probabilities"]
                class_names = results["class_names"]
                
                # Display results
                with col2:
                    st.subheader("Prediction Results")
                    st.markdown(f"**Predicted condition:** {predicted_class}")
                    
                    # Get disease information
                    disease_info = get_disease_info(predicted_class)
                    if disease_info:
                        st.markdown(f"**Full name:** {disease_info.get('name', 'N/A')}")
                        st.markdown("### Description")
                        st.write(disease_info.get('description', 'No description available.'))
                        st.markdown("### Risk Factors")
                        st.write(disease_info.get('risk_factors', 'No information available.'))
                        st.markdown("### Common Treatments")
                        st.write(disease_info.get('treatment', 'No information available.'))
                
                # Display probability chart
                st.subheader("Prediction Probabilities")
                fig = plot_prediction_probabilities(probabilities, class_names)
                st.pyplot(fig)
                
                # Disclaimer
                st.warning(
                    "**Medical Disclaimer:** This tool is for educational purposes only. "
                    "The predictions should not be used for diagnosis or treatment decisions. "
                    "Please consult a healthcare professional for medical advice."
                )

if __name__ == "__main__":
    main()