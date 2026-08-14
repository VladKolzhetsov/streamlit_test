import streamlit as st
from transformers import pipeline
import pandas as pd
import numpy as np

from plot_func import seaborn_pairplot, confusion_matrix_plot
from helpfull_utils import *


if "__main__" == __name__:

    st.button("Start New Session", type="primary", on_click=start_new_session)

    with st.expander("Click here to show/hide text"):
        st.write( read_readme )

    memory_buffer = Memory_buffer()

    pipe = pipeline("text-classification", model="VK26/disilbert-finetuned-emotion2")

    st.header("Message")
    if "user_input" not in st.session_state:
        st.session_state.user_input = ""
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

    if st.button("Commit changes", on_click=commit_changes):
        seaborn_pairplot(cnt_emo, cnt_adv_emo)
        confusion_matrix_plot(cnt_emo, cnt_adv_emo)
  
    seaborn_pairplot(cnt_emo, cnt_adv_emo)
    confusion_matrix_plot(cnt_emo, cnt_adv_emo)
