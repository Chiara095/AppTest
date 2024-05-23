import os
import gdown
import streamlit as st
from zipfile import ZipFile
import openai
import torch
from transformers import AutoTokenizer, CamembertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pycountry

# Install Rust
os.system("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")

# Add Rust to PATH
rust_env_path = os.path.expanduser("~/.cargo/env")
if os.path.exists(rust_env_path):
    with open(rust_env_path) as f:
        for line in f:
            if line.startswith('export'):
                key, value = line.replace('export ', '').strip().split('=')
                os.environ[key] = value

# Function to check and download the model files
def download_and_unzip(url, extract_to='Downloads'):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        st.info('Downloading model files...')
        output_path = f"{extract_to}/model.zip"
        gdown.download(url, output_path, quiet=False)
        st.info('Extracting model files...')
        with ZipFile(output_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(output_path)
        st.success('Downloaded and extracted model files successfully!')
    else:
        st.info('Model files already downloaded.')

    # Log contents of the directory
    extracted_files = os.listdir(extract_to)
    st.info(f"Extracted files: {extracted_files}")  # This will show the files in the directory

# URL of the zip file on Google Drive
zip_file_url = 'https://drive.google.com/uc?export=download&id=1CfXmznt24jEHRyymtxYg7aiyyN6AdY-k'

# Download and unzip model files if needed
download_and_unzip(zip_file_url)

# Adjust the model directory path
model_dir = os.path.join('Downloads', 'camembert_full_515')  # Adjusted to point to the correct subdirectory

try:
    # Attempt to load the model
    config_path = os.path.join(model_dir, 'config.json')
    st.info(f'Looking for config file at: {config_path}')
    
    if os.path.exists(config_path):
        st.info('Config file found. Proceeding to load the model...')
        
        # Load the tokenizer and the model
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = CamembertForSequenceClassification.from_pretrained(model_dir)
        
        st.success('Model loaded successfully!')
    else:
        st.error('Config file not found. Please check the extracted files and the path.')
except Exception as e:
    st.error(f'Error loading the model: {e}')

# Set your OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Load the pre-trained Sentence Transformer model for semantic similarity
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define language levels
levels = ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']

# Initialize session state for tracking user progress and responses
if "level" not in st.session_state:
    st.session_state.level = 0
if "user_answer" not in st.session_state:
    st.session_state.user_answer = ""
if "question" not in st.session_state:
    st.session_state.question = ""
if "test_finished" not in st.session_state:
    st.session_state.test_finished = False
if "relevance_score" not in st.session_state:
    st.session_state.relevance_score = 0.0
if "confirmed_country" not in st.session_state:
    st.session_state.confirmed_country = False

# Define the raw URLs for the images
image_path_left = 'https://raw.githubusercontent.com/Chiara095/AppTest/main/Berre.png'
image_path_right = 'https://raw.githubusercontent.com/Chiara095/AppTest/main/Vaient.png'

# Display branding images and title in the app header
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("<p style='text-align: center;'>Berre</p>", unsafe_allow_html=True)
    st.image(image_path_left, width=150)

with col2:
    st.markdown("<h1 style='text-align: center;'>Welcome to Berrevaient's French Proficiency Test!</h1>", unsafe_allow_html=True)

with col3:
    st.markdown("<p style='text-align: center;'>Vaient</p>", unsafe_allow_html=True)
    st.image(image_path_right, width=150)

# Generate a new question for the current level
def generate_question(level):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": f"Generate a French question for level {levels[level]}."}]
    )
    return response['choices'][0]['message']['content']

# Compute the semantic similarity between the question and the answer
def compute_similarity(question, answer):
    embeddings = semantic_model.encode([question, answer])
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
    return similarity.item()

# Process user answer and level advancement
def process_answer():
    # Tokenize the user's answer
    inputs = tokenizer(st.session_state.user_answer, return_tensors='pt')

    # Make a prediction with your model
    outputs = model(**inputs)

     # Extract the predicted level from the outputs
    predicted_level = outputs.logits.argmax(-1).item()

    # Compute similarity
    st.session_state.relevance_score = compute_similarity(st.session_state.question, st.session_state.user_answer)
    st.write(f"Relevance score: {st.session_state.relevance_score:.2f}")

    if st.session_state.relevance_score < 0.5:
        st.write("Your answer does not seem relevant to the question. Please try again.")
        return
    
    st.write(f"Predicted level: {levels[predicted_level]}")

    # Check if answer predicts a level at or below the current level, or one level above
    if predicted_level <= st.session_state.level + 1:
        st.session_state.level = min(st.session_state.level + 1, len(levels) - 1)
    else:
        st.session_state.level = min(predicted_level, len(levels) - 1)
    
    # Check if maximum level is reached or exceeded
    if st.session_state.level >= len(levels) - 1:
        st.session_state.test_finished = True
        st.write(f'Your final French level is {levels[-1]}.')
    else:
        st.session_state.question = generate_question(st.session_state.level)

# Log the state for debugging - Define this at the top level for global access
def log_state():
    st.write(f"Current level: {st.session_state.level}")
    st.write(f"User answer: {st.session_state.user_answer}")
    st.write(f"Question: {st.session_state.question}")
    st.write(f"Test finished: {st.session_state.test_finished}")
    st.write(f"Relevance score: {st.session_state.relevance_score:.2f}")

# Country selection and check if French is an official language
if not st.session_state.confirmed_country:
    selected_country = st.selectbox("Select your country of origin:", sorted([country.name for country in pycountry.countries]))
    if st.button("Confirm Country"):
        country_obj = pycountry.countries.get(name=selected_country)
        def is_french_speaking(country_code):
            french_speaking_countries = ['FR', 'BE', 'CH', 'CA', 'LU', 'MC', 'CD', 'CI', 'SN', 'ML', 'BF', 'NE', 'TD', 'GN', 'CF', 'CG', 'GA', 'DJ', 'KM', 'MG', 'TG', 'BJ']
            return country_code in french_speaking_countries
