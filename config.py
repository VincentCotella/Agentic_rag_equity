# config.py

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

CHROMA_PATH = "chroma"
DATA_PATH = "PDF"
STRUCTURED_DATA_CSV_PATH = "data/private_equity_data.csv"

TOP_K = 5
