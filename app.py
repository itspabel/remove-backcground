# File path: app.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

def remove_background(image):
    """
    Removes the background from an image using OpenCV's GrabCut algorithm.

    Args:
        image (PIL.Image.Image): The uploaded image as a PIL Image object.

    Returns:
        PIL.Image.Image: Processed image with the background removed.
    """
    # Convert PIL Image to OpenCV format (numpy array)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Create a mask initialized to "probably background"
    mask = np.zeros(image.shape[:2], np.uint8)

    # Define the initial rectangular region of interest (ROI)
    height, width = image.shape[:2]
    rect = (10, 10, width - 20, height - 20)  # Rectangle within the image

    # Allocate space for models used by GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Apply GrabCut with the rectangle
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)

    # Modify the mask to classify sure background and sure foreground
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    result = image * mask[:, :, np.newaxis]

    # Convert the result back to PIL Image format
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result)

# Streamlit App
st.title("Background Removal Tool")
st.write("Upload an image, and we'll remove its background!")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Remove the background
    st.write("Processing the image...")
    processed_image = remove_background(image)

    # Display the processed image
    st.image(processed_image, caption="Processed Image", use_column_width=True)

    # Download button
    st.download_button(
        label="Download Processed Image",
        data=processed_image.tobytes(),
        file_name="processed_image.png",
        mime="image/png"
    )
