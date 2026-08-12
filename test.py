"""
# My first app
Here's our first attempt at using data to create a table:
"""

"""
import streamlit as st
import pandas as pd
df = pd.DataFrame({
  'first column': [1, 2, 3, 4],
  'second column': [10, 20, 30, 40]
})

df
"""


import streamlit as st
from transformers import pipeline

pipe = pipeline("text-classification", model="VK26/disilbert-finetuned-emotion2")

st.write(pipe("Fuck you, asshole !!!"))
