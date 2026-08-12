import streamlit as st
from transformers import pipeline

emotion2color = {
  "sadness" : "blue",
  "joy" : "yellow",
  "neutral": "gray",
  "anger": "red",
  "fear": "violet",
  "surprise": "orange",
  "disgust": "green"
}

pipe = pipeline("text-classification", model="VK26/disilbert-finetuned-emotion2")

st.header("Message")
user_input = st.text_area("label goes here", "")

st.header("Responce")
responce = pipe(user_input)
emotion, prob = responce.values()
st.badge(f"{emotion} with {round(prob, 3) * 100} % confidence", color = emotion2color[emotion])
