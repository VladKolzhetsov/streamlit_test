from collections import Counter
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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
    cm = np.zeros((len(labels), len(labels)), dtype=int)
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
