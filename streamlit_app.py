import os

# Install Rust
os.system("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")

# Add Rust to PATH
os.system("source $HOME/.cargo/env")

import streamlit as st
import openai
import torch
from transformers import AutoTokenizer, CamembertForSequenceClassification
from sentence_transformers import SentenceTransformer, util

# Function to check and download the model files
def download_and_unzip(url, extract_to='model_directory'):
    try:
        if not os.path.exists(extract_to):
            os.makedirs(extract_to)
            st.info('Downloading model files...')
            r = requests.get(url)
            if r.status_code == 200:
                z = ZipFile(BytesIO(r.content))
                z.extractall(path=extract_to)
                st.success('Downloaded and extracted model files successfully!')
            else:
                st.error(f"Failed to download model files: Status code {r.status_code}")
        else:
            st.info('Model files already downloaded.')
    except Exception as e:
        st.error(f"Failed to download or extract model files: {e}")

# URL of the zip file on Google Drive
zip_file_url = 'https://drive.google.com/uc?export=download&id=1CfXmznt24jEHRyymtxYg7aiyyN6AdY-k'

#Link to file for visualization: https://drive.google.com/file/d/1CfXmznt24jEHRyymtxYg7aiyyN6AdY-k/view?usp=sharing
#ZIP FILE ID: 1CfXmznt24jEHRyymtxYg7aiyyN6AdY-k

# Download and unzip model files if needed
download_and_unzip(zip_file_url)
