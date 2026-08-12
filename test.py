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

def color_rows_in_dataframe(df):
    return [f'background-color: {emotion2color[row.Emotion]}' for row in df.itertuples()]

class Memory_buffer:
    def __init__(self, *, defoult_size = 50):
        self.buffer = list()
        self.defoult_size = defoult_size
      
    def push(self, data, emotion, confidence):
        self.buffer.append( (data, emotion, confidence) )
        if len(self.buffer) > self.defoult_size:
            self.buffer = self.buffer[:-1]

    def get_dataframe(self, color_rows_in_dataframe_func = None):
        texts, emotions, confidences = zip(*self.buffer)
        df = pd.Dataframe({
            "Message" : texts,
            "Emotion" : emotions,
            "Confidence" : confidences
         })
      
        if color_dataframe_func is None:
            return df
        return df.style.apply(color_rows_in_dataframe_func, axis=1)
    
        

memory_buffer = Memory_buffer()

pipe = pipeline("text-classification", model="VK26/disilbert-finetuned-emotion2")

st.header("Message")
user_input = st.text_area("message goes here", "", max_chars=250)

st.header("Responce")
responce = pipe(user_input)[0]
emotion, prob = responce.values()
#st.badge(f"{emotion} with {round(prob, 3) * 100} % confidence", color = emotion2color[emotion])
st.markdown( f":violet-badge[ {emotion} ] with :violet-badge[ {round(prob, 3) * 100}% ] confidence." )
memory_buffer.push( user_input, emotion, prob )

df_messages = memory_buffer.get_dataframe(color_rows_in_dataframe_func = color_rows_in_dataframe)
st.table( df_messages )
