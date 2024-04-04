import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.convert('RGB')  # Convert image to RGB mode
    img = img.resize((128, 128))  # Resize image to match the input size of the model
    img = np.array(img) / 255.0  # Normalize pixel values
    img = img[np.newaxis, ...]  # Add batch dimension
    return img

def main():

    st.title('GRADED CHALLENGE 7 - CV ANN')
    st.write('In recent years, there has been an increasing emphasis on workplace safety and personal protective equipment (PPE) compliance across various industries. One crucial aspect of PPE compliance is the proper use of masks, particularly in environments where respiratory hazards are present. However, ensuring consistent adherence to mask-wearing protocols can be challenging, leading to potential safety risks and regulatory non-compliance. In response, this project aims to develop an Artificial Neural Network (ANN) based computer vision system for classifying images of individuals based on their mask-wearing behavior. The system will be trained to categorize images into three classes: wearing a mask correctly, not wearing a mask, and wearing a mask improperly. The objective is to provide a robust tool for monitoring mask compliance in industrial and occupational settings, ultimately enhancing workplace safety standards and regulatory compliance efforts.')

    st.subheader('Data Example')

    # Load the image from file
    with_mask_path = 'with_mask.jpg'
    without_mask_path = 'without_mask.png'
    with_mask = Image.open(with_mask_path)
    without_mask = Image.open(without_mask_path)
    
    # Display the image
    st.image(with_mask, caption='with_mask', use_column_width=True)
    st.image(without_mask, caption='without_mask', use_column_width=True)

    st.subheader('Inference')

     # Allow user to upload an image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    model = load_model('my_model.keras')

    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Display prediction result
        if prediction[0] > 0.5:
            st.write("Prediction: without_mask")
        else:
            st.write("Prediction: with_mask")


if __name__ == "__main__":
    main()