import streamlit as st
from transformers import pipeline
import pandas as pd

emotion2color = {
  "sadness" : "blue",
  "joy" : "orange",
  "neutral": "gray",
  "anger": "red",
  "fear": "violet",
  "surprise": "black",
  "disgust": "green"
}

def color_rows_in_dataframe(row):
    return [ f'background-color: {emotion2color[row["Emotion"]]}' ] * len(row)

class Memory_buffer:
    def __init__(self, *, default_size = 50):
        if "buffer" not in st.session_state:
            st.session_state.buffer = list()
        self.default_size = default_size
      
    def push(self, data, emotion, confidence):
        st.session_state.buffer.append( (data, emotion, confidence) )
        if len(st.session_state.buffer) > self.default_size:
            st.session_state.buffer = st.session_state.buffer[:-1]

    def get_dataframe(self, color_rows_in_dataframe_func = None):
        texts, emotions, confidences = zip(*st.session_state.buffer)
        
        df = pd.DataFrame({
            "id" : df.index.tolist(),
            "Message" : texts,
            "Confidence" : confidences,
            "Emotion" : emotions,
            "Your advice" : [""] * len(texts)
         })
      
        if color_rows_in_dataframe_func is None:
            return df
        return df.style.apply(color_rows_in_dataframe_func, axis=1)
    
        

memory_buffer = Memory_buffer()

pipe = pipeline("text-classification", model="VK26/disilbert-finetuned-emotion2")

st.header("Message")
user_input = st.text_area("message goes here", "Nothing", max_chars=250)

st.header("Responce")
responce = pipe(user_input)[0]
emotion, prob = responce.values()
#st.badge(f"{emotion} with {round(prob, 3) * 100} % confidence", color = emotion2color[emotion])
st.markdown( f":{emotion2color[emotion]}-badge[ {emotion} ] with :{emotion2color[emotion]}-badge[ {round(prob * 100, 3)}% ] confidence." )
memory_buffer.push( user_input, emotion, prob )

df_messages = memory_buffer.get_dataframe(color_rows_in_dataframe_func = color_rows_in_dataframe)
st.data_editor( df_messages, disable=["id", "Message", "Confidence" "Emotion"] )
