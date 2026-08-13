import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np
from collections import Counter
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

st.button("Start New Session", type="primary", on_click=start_new_session)

memory_buffer = Memory_buffer()

pipe = pipeline("text-classification", model="VK26/disilbert-finetuned-emotion2")

st.header("Message")
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

def submit():
    st.session_state.user_input = st.session_state.message
    st.session_state.message = ""
st.text_area("", "", max_chars=250, placeholder="text goes here", key="message", on_change=submit)


st.header("Responce")
responce = pipe(st.session_state.user_input)[0]
emotion, prob = responce.values()
st.markdown( f":{emotion2color[emotion]}-badge[ {emotion} ] with :{emotion2color[emotion]}-badge[ {round(prob * 100, 3)}% ] confidence." )
if st.session_state.user_input:
    memory_buffer.update()
    memory_buffer.push( st.session_state.user_input, emotion, prob )
st.session_state.user_input = ""

df_messages = memory_buffer.get_dataframe(color_rows_in_dataframe_func = color_rows_in_dataframe)


if st.session_state.buffer:
    st.data_editor( df_messages,
                   disabled=["Message", "Confidence", "Emotion"],
                   key="user_responce",
                   num_rows="fixed",
                   column_config=column_config )
else:
    st.data_editor( df_messages,
                   disabled=["Message", "Confidence", "Emotion"],
                   key="user_responce",
                   num_rows="fixed" )

df_messages.data.loc[df_messages.data["Your advice"] == '', 'Your advice'] = df_messages.data["Emotion"]
cnt_emo = df_messages.data["Emotion"].tolist()
cnt_adv_emo = df_messages.data["Your advice"].tolist()

def commit_changes():
    memory_buffer.update()

@st.cache_data
def seaborn_pairplot(cnt_emo, cnt_adv_emo):
    st.title("Seaborn Pairplot")
    cnt_emotions = data_for_histogram_counter( Counter(cnt_emo) )
    cnt_adv_emotions = data_for_histogram_counter( Counter(cnt_adv_emo) )

    df = pd.DataFrame({
        'Word': cnt_emotions.keys(),
        'Doc1': cnt_emotions.values(),
        'Doc2': cnt_adv_emotions.values()
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Word', y='Doc1', data=df, ax=ax, label='Predicted', color='blue', alpha=0.5)
    sns.barplot(x='Word', y='Doc2', data=df, ax=ax, label='Guided', color='red', dodge=True, alpha=0.5)

    ax.set_xlabel('Emotions')
    ax.set_ylabel('Count')
    ax.set_title('Emotion Frequencies')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

@st.cache_data
def confusion_matrix_plot(cnt_emo, cnt_adv_emo):
    st.title("Confusion matrix")
    labels = list( emotion2color.keys() )
    cm = np.zeros((len(labels), len(labels)))
    if cnt_emo and cnt_adv_emo:
        cm = confusion_matrix(cnt_emo, cnt_adv_emo,labels=labels )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    ax.set_xlabel("Guided emotions", fontsize=12, labelpad=10)
    ax.set_ylabel("Predicted emotions", fontsize=12, labelpad=10)
    plt.tight_layout()
    st.pyplot(fig)

if st.button("Commit changes", on_click=commit_changes):
    seaborn_pairplot(cnt_emo, cnt_adv_emo)
    confusion_matrix_plot(cnt_emo, cnt_adv_emo)
  
seaborn_pairplot(cnt_emo, cnt_adv_emo)
confusion_matrix_plot(cnt_emo, cnt_adv_emo)
