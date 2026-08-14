import streamlit as st
import pandas as pd


emotion2color = {
  "sadness" : "blue",
  "joy" : "orange",
  "neutral": "gray",
  "anger": "red",
  "fear": "violet",
  "surprise": "darkslategray",
  "disgust": "green"
}

column_config={
        "Your advice": st.column_config.SelectboxColumn(
            "Your advice",
            options=emotion2color.keys(),
            required=True,
        )
    }

def color_rows_in_dataframe(row):
    return [ f'background-color: {emotion2color[row["Emotion"]]}' ] * len(row)

def data_for_histogram_counter(counter):
    _dict = {"sadness" : 0, "joy" : 0, "neutral": 0, "anger": 0,
             "fear": 0, "surprise": 0, "disgust": 0}
    for key, value in counter.items():
        _dict[key] += value
    return _dict

def read_readme():
    with open("README.md", "r", encoding="utf-8") as _file:
        return _file.read()

class Memory_buffer:
    def __init__(self, *, default_size = 50):
        if "buffer" not in st.session_state:
            st.session_state.buffer = list()
          
        self.default_size = default_size
      
        if "user_advices" not in st.session_state:
            st.session_state.user_advices = list()
      
    def push(self, data, emotion, confidence):
        st.session_state.buffer.append( (data, emotion, confidence) )
        if len(st.session_state.buffer) > self.default_size:
            st.session_state.buffer = st.session_state.buffer[:-1]

    def get_dataframe(self, color_rows_in_dataframe_func = None):
        texts, emotions, confidences = list(), list(), list()
        if st.session_state.buffer:
            texts, emotions, confidences = zip(*st.session_state.buffer)
          
        st.session_state.user_advices += [""] * (len(confidences) - len(st.session_state.user_advices)) 
      
        df = pd.DataFrame({
            "Message" : texts,
            "Confidence" : confidences,
            "Emotion" : emotions,
            "Your advice" : st.session_state.user_advices
         })

        if color_rows_in_dataframe_func is None:
            return df
        return df.style.apply(color_rows_in_dataframe_func, axis=1)

    def update(self):
        for row_index, col_advice in st.session_state["user_responce"]["edited_rows"].items():
            st.session_state.user_advices[row_index] = col_advice["Your advice"]
    

def start_new_session():
    st.session_state.clear()
    st.rerun()

def commit_changes():
    memory_buffer.update()

def submit():
    st.session_state.user_input = st.session_state.message
    st.session_state.message = ""

