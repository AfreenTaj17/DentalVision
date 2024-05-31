import streamlit as st, numpy as np, os, shutil, json, subprocess; from PIL import Image

st.title("Dental Vision YOLOv7 Annotation Tool")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img = Image.open(uploaded_file); img.save(uploaded_file.name); st.image(img, caption='Uploaded Image', use_column_width=True)
    
    try:
        subprocess.run(["python", "detect.py", "--weights", "best.pt", "--source", uploaded_file.name], check=True)
    except subprocess.CalledProcessError as e:
        st.error(f"Error occurred during YOLOv7 detection: {e}")
    else:
        img_path0 = os.path.join(os.getcwd(), "runs/detect/exp", uploaded_file.name)
        processed_image_path = img_path0 if os.path.exists(img_path0) else None
        
        if processed_image_path is None:
            for i in range(2, 100):  
                img_pathx = os.path.join(os.getcwd(), f"runs/detect/exp{i}", uploaded_file.name)
                if os.path.exists(img_pathx):
                    processed_image_path = img_pathx
                    break

        if processed_image_path is not None:
            st.image(processed_image_path, caption='Processed Image', use_column_width=True)
        else:
            st.error("Processed image not found. YOLOv7 detection might have failed.")

st.sidebar.header("About"); st.sidebar.text("This app allows users to upload dental images, process them through a YOLOv7 model, and display the results.")