import os
import json
import pickle
import pandas as pd
import streamlit as st
from nltk import tokenize

from article_retrieval import RetrieverGlove
from summarization import Summarizer



@st.cache_resource
def init_IR():
    # Load necessary files for initialization
    processed_df = pd.read_csv("data/processed_df.csv", index_col=0)
    with open("data/knowledge_base.pkl", 'rb') as f:
        KB = pickle.load(f) 
    with open(f"data/inverted_index.pkl", 'rb') as f:
        inverted_index = pickle.load(f)
    with open(f"data/glove_final_dict.pkl", 'rb') as f:
        glove_dict = pickle.load(f)
    
    IR = RetrieverGlove(inverted_index, glove_dict, KB, processed_df)
    return IR


@st.cache_data
def get_article(query, _IR):
    return _IR.retrive(query)


def main():
    # Init information retriever
    IR = init_IR()
    
    st.title = ("QA and Text Summarization")

    ### Distribution visualisation
    st.header("Question Answering Application")


    input_text = st.text_input(
        "Input query",
        placeholder="Enter input query"
    )
    # input_text = st.text_area('Input query (Ctrl + Enter to apply)')

    if not input_text:
        return 


    option = st.selectbox('Which summarization method do you want to use?',
                          ('None', 'NLTK', 'BERT'))
    
    if option == 'NLTK':
        approach = st.selectbox('Which approach do you want to use?',
                                ('Weighted sentence', 'Text rank'))
        method = 1 if approach == 'Weighted sentence' else 2
    
    if option != 'None':
        with st.spinner(text="Finding relevant aricles..."):
            # snippet, score = IR.retrive(input_text)
            snippet, score = get_article(input_text, IR)          
        
        if snippet:
            sentences = []
            for sent in tokenize.sent_tokenize(snippet):
                sentences.append(sent)
            
            summary_length = st.slider('Choose number of sentences in the answer', 
                                    min_value=3, max_value=len(sentences), value=3)

            if st.button('Get answer'):
                st.write("Score of the answer:", score)
                if option == "NLTK":
                    summ =  Summarizer(language='english', summary_length=summary_length)
                    with st.spinner(text="This may take a moment..."):
                        summarize = summ.summarize(snippet, method=method)
                
                st.write(summarize)


if __name__ == "__main__":
    main()