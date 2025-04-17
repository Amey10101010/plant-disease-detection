import streamlit as st
import tensorflow as tf
import numpy as np
import os
import tempfile
from PIL import Image


# ========== PREDICTION FUNCTION ==========
def model_prediction(test_image):
    try:
         current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "training", "trained_plant_disease_model.keras")
        # =====================================

        # Debugging checks (optional but recommended)
        st.write("Current directory:", current_dir)
        st.write("Model path:", model_path)
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None, None

        model = tf.keras.models.load_model(model_path)

        # Process image from either source
        image = Image.open(test_image)
        image = image.resize((256, 256))
        input_arr = tf.keras.utils.img_to_array(image)
        input_arr = np.array([input_arr])  # Convert to batch format
        predictions = model.predict(input_arr)

        # Get confidence percentage and predicted class
        confidence = round(100 * np.max(predictions), 2)
        predicted_class = np.argmax(predictions)
        return predicted_class, confidence

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None


# ========== SIDEBAR & APP MODE ==========
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# ========== PAGE ROUTING ==========
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "assets/home_page.jpeg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
     This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 3 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                
    """)

elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    # Input method selection
    input_method = st.radio("Select Input Method:", ["Upload Image", "Take Photo"])

    test_image = None

    if input_method == "Upload Image":
        test_image = st.file_uploader("Choose an Image:", type=["jpg", "jpeg", "png"])
    else:
        test_image = st.camera_input("Take a picture of plant:")

    if test_image:
        # Show image preview
        st.image(test_image, use_container_width=True)

        if st.button("Predict"):
            st.snow()

            # Process image using tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                tmp.write(test_image.getbuffer())
                tmp_path = tmp.name

            # Get prediction and confidence
            result_index, confidence = model_prediction(tmp_path)

            # Clean up temporary file
            try:
                os.remove(tmp_path)
            except Exception as e:
                st.warning(f"Couldn't delete temp file: {str(e)}")

            if result_index is not None and confidence is not None:
                class_names = [
                    "Potato___Early_blight",
                    "Potato___Late_blight",
                    "Potato___healthy",
                ]

                # Get predicted class name
                predicted_class = class_names[result_index]

                # Disease explanations dictionary
                disease_info = {
                    "Potato___Early_blight": "Brown spots with concentric rings on lower leaves",
                    "Potato___Late_blight": "Water-soaked lesions that spread quickly",
                    "Potato___healthy": "No visible disease symptoms",
                }

                # Display results
                st.success(f"**Prediction:** {predicted_class}")
                st.info(f"**Confidence Level:** {confidence}%")

                # Show explanation and recommendations
                with st.expander(f"What does {predicted_class} mean?"):
                    st.markdown(f"""
                    **Characteristics:**
                    - {disease_info[predicted_class]}
                    """)

                    # Add specific recommendations
                    if predicted_class == "Potato___Early_blight":
                        st.warning("""
                        **Recommended Action:** 
                        - Remove affected leaves immediately
                        - Apply copper-based fungicide
                        - Avoid overhead watering
                        """)

                    elif predicted_class == "Potato___Late_blight":
                        st.warning("""
                        **Recommended Action:** 
                        - Destroy infected plants
                        - Apply chlorothalonil fungicide
                        - Ensure good air circulation
                        """)

                    elif predicted_class == "Potato___healthy":
                        st.success("""
                        **Recommendation:** 
                        - Continue regular monitoring
                        - Maintain proper spacing between plants
                        - Water at soil level only
                        """)


# Add footer
st.markdown("---")
st.caption(
    "Plant Disease Recognition System © 2025 | Made by VIT Students | Hosted on Streamlit Cloud"
)
