import streamlit as st
import os
from datetime import datetime

UPLOAD_ROOT = "uploads"

st.title("SecureMed-FL Client")

hospital_name = st.text_input("Enter Hospital / Client Name")

uploaded_files = st.file_uploader(
    "Upload medical images",
    accept_multiple_files=True,
    type=["png", "jpg", "jpeg"]
)

if st.button("Save Dataset") and hospital_name:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = os.path.join(
        UPLOAD_ROOT, f"{hospital_name}_{timestamp}"
    )
    os.makedirs(folder_path, exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join(folder_path, file.name), "wb") as f:
            f.write(file.getbuffer())

    st.success(f"Dataset saved at: {folder_path}")
    st.info("You can now train the model using this dataset.")