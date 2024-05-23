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
