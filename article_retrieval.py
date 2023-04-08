import math
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import stopwords


class RetrieverGlove:
    def __init__(self,inverted_index, glove_dict, KB, data_df):
        self.nlp = spacy.load("en_core_sci_sm") # for medical NER on query
        self.nlp_general = spacy.load("en_core_web_sm") # for general NER on query, such as author names, etc
        self.word_embedding = glove_dict
        self.inverted_index = inverted_index
        self.KB = KB
        self.norm_dict = self.__canonical_aliases_normalization()
        self.data_df = data_df
        return

    def __get_vec(self, word):
        if word in self.word_embedding.keys():
            return self.word_embedding.get(word)
        else:
            # if the word is unknown in GloVe, then return a zero vector with the same dim i.e. 300
            return np.zeros(300)

    def __embed_text(self, ent_lst):
        entity_embedding_dict = {}
        for ent in ent_lst:
            # if the entity is a term consists of multiple word tokens, then just avergae it
            if ' ' in ent:
                stack = np.vstack((self.__get_vec(w) for w in ent.split()))
                ent_vector = np.mean(stack,axis = 0)
            else:
                ent_vector = self.__get_vec(ent)
            entity_embedding_dict.update({ent:ent_vector})
        return entity_embedding_dict

    def __canonical_aliases_normalization(self):
        norm_dict = {}
        for can in list(set(self.KB['entities'].keys())):
            for alias in list(self.KB['entities'][can]['aliases']):
                norm_dict.update({alias:can})
        return norm_dict

    def __cal_wv_similarity(self, entity_vec_dict1, entity_vec_dict2):
        # if the text contain multiple entities,
        # then the vector that represents this text is the average of all the entity vector
        for i, embedding_dict in enumerate([entity_vec_dict1,entity_vec_dict2]):
            # if there's no embedding, then return similarity of 0
            if len(embedding_dict) == 0:
                return 0
            if len(embedding_dict) == 1:
                text_vec = list(embedding_dict.values())[0]
            else:
                stack = np.vstack((v for v in embedding_dict.values()))
                text_vec = np.mean(stack,axis = 0)
            if i == 0:
                text_vec1 = text_vec
            else:
                text_vec2 = text_vec

        if norm(text_vec1) == 0 or norm(text_vec2) == 0:
            return 0
        cos_sim = dot(text_vec1, text_vec2)/(norm(text_vec1)*norm(text_vec2))
        return cos_sim

    def __cal_tfidf_similarity(self, matched_ent, paper_id):
        tfidf_score = 0
        for ent in matched_ent:
            if len([ele[1] for ele in self.inverted_index.get(ent) if ele[0] == paper_id]) != 0:
                t_count = [ele[1] for ele in self.inverted_index.get(ent) if ele[0] == paper_id][0]
                tf_w = 1 + math.log10(t_count)
                N = 10000
                df = len(self.KB['entities'][ent]['paper_lst'])
                idf_w = math.log10((N/df))
                tfidf_score += tf_w*idf_w
        return tfidf_score

    def __show_result(self, paper_selected):
        for rank in range(1, len(paper_selected)+1):
            idx = rank -1
            snippet = self.data_df.loc[self.data_df['paper_id'] == paper_selected[idx][0]]['original_abstract'].values[0]
            title = self.data_df.loc[self.data_df['paper_id'] == paper_selected[idx][0]]['title'].values[0]
            authors = self.data_df.loc[self.data_df['paper_id'] == paper_selected[idx][0]]['authors'].values[0]
            article_identifier = self.data_df.loc[self.data_df['paper_id'] == paper_selected[idx][0]]['paper_identifier'].values[0]
            article_number = self.data_df.loc[self.data_df['paper_id'] == paper_selected[idx][0]]['paper_id'].values[0]
            score = paper_selected[idx][1]
            if rank != 1:
                print('\n')
            print(f'------------------------------ Match Ranked {rank} (similarity score: {score})------------------------------')
            print(f'article identifier: {article_identifier},  article number: {article_number} \n')
            print(f'title:  {title} \n')
            print('Authors: ')
            print('; '.join(authors),'\n')
            print('snippet: \n')
            print(snippet)

    def retrive(self, query, show=False):
        stops = stopwords.words('english') # english stopwords, because our data is in english
        query = " ".join([word for word in query.split() if word not in stops])
        query = query.replace('[^A-Za-z0-9\s+\-]',' ')
        q_doc = self.nlp(query)
        q_doc_general = self.nlp_general(query)
        q_entities = list(set([ent.text.strip() for ent in  q_doc.ents] + [ent.text.strip() for ent in  q_doc_general.ents]))
        query_ent_vec_dict = self.__embed_text(q_entities)

        doc_lst = []
        matched_ent = []
        for ent in q_entities:
            # try hard match first
            if ent in self.inverted_index.keys():
                paper_lst = [ele[0] for ele in self.inverted_index.get(ent)]
                doc_lst.extend(paper_lst)
                matched_ent.append(ent)
            elif ent in self.norm_dict.keys():
                can = self.norm_dict[ent]
                paper_lst = [ele[0] for ele in self.inverted_index.get(can)]
                doc_lst.extend(paper_lst)
                matched_ent.append(can)
            # if can't hard match, use word vector to soft match them with a threshold
            else:
                ENT_SOFT_MATCH_THRESH = 0.7
                can_lst = list(set(self.norm_dict.values()))
                sim_lst = []
                for can in can_lst:
                    can_vec_dict = self.__embed_text([can])
                    q_ent_vec_dict = {ent:query_ent_vec_dict[ent]}

                    # if any of these two entities are completely out of vocab in GloVe, skip match them
                    if np.all(can_vec_dict.get(can)==0) or np.all(q_ent_vec_dict.get(ent)==0):
                        continue

                    sim_lst.append(self.__cal_wv_similarity(q_ent_vec_dict,can_vec_dict))
                if len(sim_lst) == 0:
                    continue
                max_sim = max(sim_lst)
                if max_sim > ENT_SOFT_MATCH_THRESH:
                    max_idx = sim_lst.index(max_sim)
                    can = can_lst[max_idx]
                    paper_lst = [ele[0] for ele in self.inverted_index.get(can)]
                    doc_lst.extend(paper_lst)
                    matched_ent.append(can)

        doc_to_be_searched = list(set(doc_lst))
        wv_score = []
        tfidf_score = []
        for paper_id in doc_to_be_searched:
            doc_ent_lst = self.KB['paper'][paper_id]['entities'] + self.KB['paper'][paper_id]['authors']
            doc_ent_vec_dict = self.__embed_text(doc_ent_lst)
            wv_score.append(self.__cal_wv_similarity(doc_ent_vec_dict,query_ent_vec_dict))
            tfidf_score.append(self.__cal_tfidf_similarity(matched_ent,paper_id))

        # standardising the wv similarity score and tfidf score to 0-1 before averaging the two scores
        if len(tfidf_score) == 0 or max(tfidf_score) == min(tfidf_score):
            std_tfidf_score = [0]*len(tfidf_score)
        else:
            std_tfidf_score = [(s - min(tfidf_score))/(max(tfidf_score)-min(tfidf_score)) for s in tfidf_score]

        if len(wv_score) == 0 or max(wv_score) == min(wv_score):
            std_wv_score = [0]*len(wv_score)
        else:
            std_wv_score = [(s - min(wv_score))/(max(wv_score)-min(wv_score)) for s in wv_score]

        # The final retrieval score is the average of the standardised tfidf and the standardised wv cos similarity
        scores_lst = [(glove+tfidf)/2 for glove,tfidf in zip(std_wv_score,std_tfidf_score)]
        scores_arr = np.array(scores_lst)
        ranked_idx = np.argsort(scores_arr)[::-1]
        paper_ranked = [(doc_to_be_searched[idx],scores_arr[idx]) for idx in ranked_idx]
        if len(paper_ranked) == 0:
            print('Sorry, there is no good match')
            return None, 0
        else:
            paper_selected = paper_ranked[0]
            snippet = self.data_df.loc[self.data_df['paper_id']==paper_selected[0]]['original_abstract'].values[0]
            score = paper_selected[1]
            if show == True:
                self.__show_result(paper_ranked[:1])
            
            return snippet, score

    def eval(self, queries, labels):
        # this function return mrr metric
        RR_lst = []
        for q,l in zip(queries,labels):
            paper_ranked = self.retrive(q,show = False)
            paper_id_ranked = [tup[0] for tup in paper_ranked]
            if l in paper_id_ranked:
                RR = 1/(paper_id_ranked.index(l)+1) # '+1' here is because we are using the position not the index
            else:
                RR = 0
            RR_lst.append(RR)
        return sum(RR_lst)/len(RR_lst)

