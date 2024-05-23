#curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
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

#os.system(“curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y”)
