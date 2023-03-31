import numpy as np
import pandas as pd
import networkx as nx
from nltk.corpus import stopwords
from nltk import tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity


class Summarizer:
    """
    A class used to summarize texts.

    Attributes
    ----------
    stop_words : set
        a set of stopwords. These are ignored when searching for the most used
        words in the text
    language : str
        the current selected language. The stop words set is language specific
    summary_length : int
        the default number of sentences to use for the summary.
    balance_length : bool
        determines if the sentence weight is weighted based on sentence length

    """

    stop_words = {}
    language = None
    summary_length = 0
    balance_length = False

    def __init__(self, language='english', summary_length=3, balance_length=True):
        """
        :param str language: The language to use, defaults to 'en'
        :param int summary_length: The default number of sentences in summary, defaults to 3
        :param bool balance_length: Balance sentences on length, default to False
        """

        # Set the language to use and set the stop words to the default
        # list provided by NLTK corpus for this language
        self.stop_words = set(stopwords.words(language))
        self.language = language

        # Set the default length for the summaries to be created
        self.summary_length = summary_length

        # Sets the switch if the sentence weights need to weighted on
        # sentence length. This might improve performance if the text
        # contains a variety of short and long sentences.
        self.balance_length = balance_length

    def weighted_sentence(self, text, summary_length=None):
        """
        Summarize the given text based on the weight of sentence
        The language and stop word set have been initialized and are used. If no
        summary length is given as parameter, the default length is used.

        :param (str or list) text: The text to summarize
        :param int summary_length: The number of sentences in summary, optional
        :return (str): A string with the summary of the given text
        """

        # Length of summary to generate, if not specified use default
        if not summary_length:
            summary_length = self.summary_length

        # Make a list of all the sentences in the given text 
        sentences = []
        if type(text) == str:
            sentences.extend(tokenize.sent_tokenize(text))
        elif type(text) == list:
            for text_part in text:
                sentences.extend(tokenize.sent_tokenize(text_part))
        else:
            return None 
   
        
        # Determine for each word, not being a stop word, the number of occurences
        # in the text. This word frequency determines the importance of the word.
        word_weights={}
        for sent in sentences:
            for word in word_tokenize(sent):
                word = word.lower()
                if len(word) > 1 and word not in self.stop_words:
                    if word in word_weights.keys():            
                        word_weights[word] += 1
                    else:
                        word_weights[word] = 1

        # The weight of each sentence equals the sum of the word importance for
        # each word in the sentence
        sentence_weights = {}
        for sent in sentences:
            sentence_weights[sent] = 0
            tokens = word_tokenize(sent)
            for word in tokens:  
                word = word.lower()
                if word in word_weights.keys():            
                    sentence_weights[sent] += word_weights[word]
            if self.balance_length and (len(tokens) > 0):
                sentence_weights[sent] = sentence_weights[sent] / len(tokens)
        highest_weights = sorted(sentence_weights.values())[-summary_length:]
        

        # The summary consists of the sentences with the highest sentence weight, in the
        # same order as they occur in the original text
        summary = ""
        for sentence, strength in sentence_weights.items():  
            if strength in highest_weights:
                summary += sentence + " "
        summary = summary.replace('_', ' ').strip()
        
        return summary
    

    def text_rank(self, text, summary_length=None):
        """
        Summarize the given text using text rank algorithm (based on page rank)
        The language and stop word set have been initialized and are used. If no
        summary length is given as parameter, the default length is used.

        :param (str or list) text: The text to summarize
        :param int summary_length: The number of sentences in summary, optional
        :return (str): A string with the summary of the given text
        """

        # Length of summary to generate, if not specified use default
        if not summary_length:
            summary_length = self.summary_length

        # Make a list of all the sentences in the given text 
        sentences = []
        if type(text) == str:
            sentences.extend(tokenize.sent_tokenize(text))
        elif type(text) == list:
            for text_part in text:
                sentences.extend(tokenize.sent_tokenize(text_part))
        else:
            return None
            

        # remove punctuations, numbers and special characters
        clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

        # make alphabets lowercase
        clean_sentences = [s.lower() for s in clean_sentences]

        # function to remove stopwords
        def remove_stopwords(sen):
            sen_new = " ".join([i for i in sen if i not in self.stop_words])
            return sen_new


        # remove stopwords from the sentences
        clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

        # Extract word vectors
        word_embeddings = {}
        with open('glove.6B.100d.txt', encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                word_embeddings[word] = coefs
                

        # Init sentence vectors
        sentence_vectors = []
        for sent in clean_sentences:
            if len(sent) != 0:
                vec = sum([word_embeddings.get(word, np.zeros((100,))) for word in sent.split()]) / (len(sent.split()) + 0.001)
            else:
                vec = np.zeros((100,))
            sentence_vectors.append(vec)
                
        # similarity matrix
        sim_mat = np.zeros([len(sentences), len(sentences)])

        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), 
                                                      sentence_vectors[j].reshape(1,100))[0, 0]
                    
                
        # Compute score 
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)
        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

        summ = []
        # Generate summary
        for i in range(summary_length):
            summ.append(ranked_sentences[i][1])
        return " ".join(summ)
     

    def summarize(self, long_text, summary_length=None, split_at=50, method=1):
        """
        Summarize the long text using two method: weighted sentence and text rank.
        The language and stop word set have been initialized and are used. If no
        summary length is given as parameter, the default length is used.
        
        :param (str or list) long_text: The long text to summarize
        :param int summary_length: The number of sentences in summary, optional
        :param int split_at: The number of sentences in each text chunk
        :param int method: 1: weighted sentence, 2: text rank
    
        :return (str): A string with the summary of the given text
        """
        
        # Length of summary to generate, if not specified use default
        if not summary_length:
            summary_length = self.summary_length

        # Make a list of all the sentences in the given text and split this list
        # in chunks of n sentences, n being the split_at value
        sentences = []
        for sent in tokenize.sent_tokenize(long_text):
            sentences.append(sent)

        chunks = [sentences[x:x+split_at] for x in range(0, len(sentences), split_at)]    
        
        # Choose method applied to summarize
        method_func = self.weighted_sentence if method == 1 else self.text_rank
        summaries = []
        for sentences in chunks:
            summary = method_func(sentences, summary_length)
            summaries.append(summary)
            
        return " ".join(summaries)
    