import streamlit as st
from transformers import pipeline

pipe = pipeline("text-classification", model="VK26/disilbert-finetuned-emotion2")

st.header("Message")
user_input = st.text_area("label goes here", "")

st.header("Responce")
st.write(pipe(user_input))
