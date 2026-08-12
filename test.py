import streamlit as st
from transformers import pipeline

pipe = pipeline("text-classification", model="VK26/disilbert-finetuned-emotion2")

st.write(pipe("Fuck you, asshole !!!"))
