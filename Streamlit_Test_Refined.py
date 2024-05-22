import streamlit as st
import openai
import torch
from transformers import AutoTokenizer, CamembertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import pycountry
import os
import requests
from zipfile import ZipFile
from io import BytesIO

# Function to check and download the model files
def download_and_unzip(url, extract_to='model_directory'):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        r = requests.get(url)
        z = ZipFile(BytesIO(r.content))
        z.extractall(path=extract_to)
        st.success('Downloaded and extracted model files successfully!')
    else:
        st.info('Model files already downloaded.')

# URL of the zip file on Google Drive
zip_file_url = 'https://drive.google.com/uc?export=download&id=1CfXmznt24jEHRyymtxYg7aiyyN6AdY-k'

# Download and unzip model files if needed
download_and_unzip(zip_file_url)

# Set your OpenAI API key
openai.api_key = st.secrets["openai_api_key"]

# Load the tokenizer and the model
model_dir = 'model_directory'
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = CamembertForSequenceClassification.from_pretrained(model_dir)

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
    inputs = tokenizer(st.session_state.user_answer, return_tensors='pt')
    outputs = model(**inputs)
    predicted_level = outputs.logits.argmax(-1).item()
    st.session_state.relevance_score = compute_similarity(st.session_state.question, st.session_state.user_answer)
    st.write(f"Relevance score: {st.session_state.relevance_score:.2f}")
    if st.session_state.relevance_score < 0.5:
        st.write("Your answer does not seem relevant to the question. Please try again.")
        return
    st.write(f"Predicted level: {levels[predicted_level]}")
    if predicted_level <= st.session_state.level + 1:
        st.session_state.level = min(st.session_state.level + 1, len(levels) - 1)
    else:
        st.session_state.level = min(predicted_level, len(levels) - 1)
    if st.session_state.level >= len(levels) - 1:
        st.session_state.test_finished = True
        st.write(f'Your final French level is {levels[-1]}.')
    else:
        st.session_state.question = generate_question(st.session_state.level)

# Define the function to check if French is an official language
def is_french_speaking(country_code):
    french_speaking_countries = ['FR', 'BE', 'CH', 'CA', 'LU', 'MC', 'CD', 'CI', 'SN', 'ML', 'BF', 'NE', 'TD', 'GN', 'CF', 'CG', 'GA', 'DJ', 'KM', 'MG', 'TG', 'BJ']
    return country_code in french_speaking_countries

# Display branding images and title in the app header
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("<p style='text-align: center;'>Berre</p>", unsafe_allow_html=True)
    # Update image paths to use relative paths or URLs
    st.image('model_directory/Berre.png', width=150)

with col2:
    st.markdown("<h1 style='text-align: center;'>Welcome to Berrevaient's French Proficiency Test!</h1>", unsafe_allow_html=True)

with col3:
    st.markdown("<p style='text-align: center;'>Vaient</p>", unsafe_allow_html=True)
    # Update image paths to use relative paths or URLs
    st.image('model_directory/Vaient.png', width=150)

# Log the state for debugging - Define this at the top level for global access
def log_state():
    st.write(f"Current level: {st.session_state.level}")
    st.write(f"User answer: {st.session_state.user_answer}")
    st.write(f"Question: {st.session_state.question}")
    st.write(f"Test finished: {st.session_state.test_finished}")
    st.write(f"Relevance score: {st.session_state.relevance_score:.2f}")

# Country selection and check if French is an official language
selected_country = st.selectbox("Select your country of origin:", sorted([country.name for country in pycountry.countries]))
if st.button("Confirm Country"):
    country_obj = pycountry.countries.get(name=selected_country)
    if country_obj and is_french_speaking(country_obj.alpha_2):
        st.success(f"In {selected_country}, French is a native language. You do not need to take the proficiency test.")
        st.session_state.test_finished = True
    else:
        st.info("Proceed with the French proficiency test.")
        st.session_state.question = generate_question(st.session_state.level)
        st.markdown("""
        **Answer the following questions in French to determine your language proficiency level.**  
        **This will help you to find your Tandem Match and tailor your learning program based on your conversations.**
        """)
        st.write(f'Question for level {levels[st.session_state.level]}: {st.session_state.question}')
        st.session_state.user_answer = st.text_input('Your answer', key=f'user_answer_{st.session_state.level}')
        if st.button('Submit Answer'):
            process_answer()
        st.slider('Your Current Level:', 0, len(levels) - 1, st.session_state.level, format=levels[st.session_state.level])
        if st.session_state.test_finished:
            st.write(f'Test finished. Your final French level is {levels[st.session_state.level]}.')

# Ensure logging and state displays only activate after country confirmation
if 'question' in st.session_state and st.session_state.question:
    log_state()