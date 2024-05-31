import streamlit as st
from PIL import Image
import pandas as pd
from ultralytics import YOLO
import cv2

supported_logos = '''
    Adidas, Corona, Google, Ritter Sport,
    Aldi, DHL, Guiness, Shell,
    Apple, Erdinger, Heineken, Singha,
    Becks, Esso, HP, Starbucks,
    BMW, Fedex, Milka, Stella Artois,
    Carlsberg, Ferrari, Nvidia, Texaco,
    Chimay, Ford, Paulaner, 3Tsingtao,
    Coca-Cola, Foster's, Pepsi, UPS
'''


@st.cache_resource
def load_model():
    model = YOLO("weight/yolov8l-best.pt")
    return model

model = load_model()

st.title("Logo Detection Demo")

st.markdown("**Note:** This web application currently supports the detection of 32 specific logos.")
st.markdown(f"Supported logos: {supported_logos}")
st.subheader("Upload an Image")
upload = st.file_uploader("Upload an image (PNG, JPG, or JPEG)", type=["png", "jpg", "jpeg"])

st.subheader("Set Confidence Threshold (Optional)")
confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

if upload:
    try:
        img = Image.open(upload)
        with st.spinner("Predicting..."):
            results = model.predict(img, conf=confidence_threshold)
        
        result = results[0] 
        
        annotated_img = cv2.cvtColor(result.plot() , cv2.COLOR_BGR2RGB)
        st.image(annotated_img, caption="Detected Logos")

        st.subheader("Detailed Results")
        if result.boxes:
            data = []
            for box in result.boxes:
                label = result.names[box.cls[0].item()]
                confidence = box.conf[0].item()
                coordinates = box.xywh[0].tolist()
                data.append([label, confidence, coordinates])

            df = pd.DataFrame(data, columns=["Label", "Confidence", "Coordinates"])

            # Split Coordinates and round values
            df[['Center X', 'Center Y', 'Width', 'Height']] = pd.DataFrame(df['Coordinates'].tolist(), index=df.index).round(0).astype("int")
            df = df.drop(columns=['Coordinates'])

            st.table(df)  # Display the modified DataFrame

        else:
            st.write("No logos detected.")

    except Exception as e:
        st.error(f"Error processing the image: {e}")
