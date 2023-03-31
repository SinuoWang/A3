import os
import json
import pandas as pd
import streamlit as st
from nltk import tokenize

from summarization import Summarizer


def main():
    st.title = ("Text Summarization")

    ### Distribution visualisation
    st.header("Text Summarisation")


    # input_text = st.text_input('Input text', '')
    input_text = st.text_area('Input text (Ctrl + Enter to apply)')

    sentences = []
    for sent in tokenize.sent_tokenize(input_text):
        sentences.append(sent)

    option = st.selectbox('Which method do you want to use?',
                          ('None', 'NLTK', 'BERT'))

    if option == 'NLTK':
        approach = st.selectbox('Which approach do you want to use?',
                              ('Weighted sentence', 'Text rank'))
        method = 1 if approach == 'Weighted sentence' else 2
    
    summary_length = st.slider('Choose number of sentences in the summary', 
                            min_value=3, max_value=len(sentences), value=3)

    if st.button('Summarize'):
        if option == "NLTK":
            summ =  Summarizer(language='english', summary_length=summary_length)
            with st.spinner(text="This may take a moment..."):
                summarize = summ.summarize(input_text, method=method)

        st.write(summarize)


if __name__ == "__main__":
    main()