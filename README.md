# streamlit_test

This is an application that uses a fine-tuned distil-bert model for emotion classification tasks.

To use the app, simply enter text into the "Message" section. The model will then analyze the input and return the detected emotion class along with the confidence score in the "Response" section.

Each detected emotion is represented by a different color in the response table for easy visual identification. The currently supported emotion classes are:

    Anger
    Neutral
    Disgust
    Joy
    Sadness
    Fear
    Surprise

If the model's prediction is incorrect, you can correct it by selecting the most appropriate emotion from the "Your advice" column in the response table.

Clicking the "Commit changes" button updates the "Model's and Yours Responses Comparison" histograms and the "Confusion Matrix."
Alternatively, these metrics will be updated automatically if you continue to input text.
