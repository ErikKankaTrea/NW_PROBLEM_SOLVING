# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import re
import unicodedata
import gensim
from scipy.spatial.distance import cdist

from flask import Flask, request
import flasgger
from flasgger import Swagger

#Initialize app:
app=Flask(__name__)
Swagger(app)


#Functions
def canonize_language_2(df, text_var):
	symbols_to_space = re.compile(u"[/\|\n(\; )|(\: )|( \()|(\) )|( \")|(\" )|( \')|(\' )]")
	symbols_to_remove = re.compile(u"[\"\'\$\€\£\(\)\:\[\]\.\,\-]\&")
	space_repetition = re.compile(u" {2,}")
	key_words_to_remove = re.compile(u"gmbh")
	cleaned_var=df[text_var].apply(lambda x: re.sub(symbols_to_remove, "", x))
	cleaned_var=df[text_var].apply(lambda x: re.sub("GmbH|gmbh|mbH|mbh|&|und|\-|an|der|co|Co|\.|\-", "", x))
	cleaned_var=cleaned_var.apply(lambda x: re.sub(space_repetition, "", x))
	cleaned_var=cleaned_var.apply(lambda x: str.strip(x))
	return cleaned_var

def put_space(aux_input): 
	words = re.findall('[A-Z][a-z]*', aux_input) 
	result = [] 
	for word in words: 
		word = chr( ord (word[0]) + 32) + word[1:] 
		result.append(word) 
	return(result)

def remove_word_elem(aux_input):
	result=[i_element for i_element in aux_input if len(i_element)>1]
	return(result)
    
def convert_list_to_string(org_list, seperator=' '):
	""" Convert list to string, by joining all item in list with given separator.
		Returns the concatenated string """
	return seperator.join(org_list)

def annotation_weight_representation(value_annotation):
    value_vectors = []
    count_model_included = 0
    count_model_nonincluded = 0
    idx_token_vectors = 0
    tags = value_annotation.split()
    word_vectors = np.empty(shape=(len(tags), 300))
    idx_word_vectors = 0
    for tag in tags:
        tag = tag.replace('_', ' ')
        if tag in model.vocab:
            word_vectors[idx_word_vectors] = model[tag]
            idx_word_vectors += 1
        else:
            tokens = tag.split()
            token_vectors = np.empty(shape=(len(tokens), 300))
            idx_token_vectors = 0
            for token in tokens:
                if token in model.vocab:
                    token_vectors[idx_token_vectors] = model[token]
                    idx_token_vectors += 1
                else:
                    continue
            if idx_token_vectors > 0:
                word_vectors[idx_word_vectors] = np.average(token_vectors[:idx_token_vectors], axis=0)
                idx_word_vectors += 1
    
    if idx_word_vectors != 0 or idx_token_vectors != 0:
        count_model_included += 1
        value_vectors.append(np.average(word_vectors[:idx_word_vectors], axis=0))
    else:
        count_model_nonincluded += 1
        value_vectors.append(np.nan)

    return value_vectors[0]


#Load data
entities_df = pd.read_csv("enti_data.csv", encoding="utf-8", sep=";")
profiles_df = pd.read_csv("prof_data.csv", encoding="utf-8", sep=";")
#Load w2v model:
model=gensim.models.KeyedVectors.load_word2vec_format("german.model" ,binary=True)

#Entitites
entities_df["annotations"]=canonize_language_2(df=entities_df, text_var="company_name")
entities_df["annotations"]=entities_df["annotations"].apply(lambda x: put_space(x))
entities_df["annotations"]=entities_df["annotations"].apply(lambda x: remove_word_elem(x))
entities_df["annotations"]=entities_df["annotations"].apply(lambda x: convert_list_to_string(x))
entities_df["annotations"]=entities_df["annotations"]+' '+entities_df["city"]+' '+entities_df["country"]+' '+entities_df["foundation_year_cat"]
#Profiles
profiles_df["annotations"]=canonize_language_2(df=profiles_df, text_var="company_name")
profiles_df["annotations"]=profiles_df["annotations"].apply(lambda x: put_space(x))
profiles_df["annotations"]=profiles_df["annotations"].apply(lambda x: remove_word_elem(x))
profiles_df["annotations"]=profiles_df["annotations"].apply(lambda x: convert_list_to_string(x))
profiles_df["annotations"]=profiles_df["annotations"]+' '+profiles_df["city"]+' '+profiles_df["country"]+' '+profiles_df["foundation_year_cat"]

#Convert data:	
profiles_df['vector_rep'] = profiles_df['annotations'].apply(lambda x: annotation_weight_representation(x))
entities_df['vector_rep'] = entities_df['annotations'].apply(lambda x: annotation_weight_representation(x))
entities_mtx = np.matrix(entities_df['vector_rep'].tolist())
entities_mtx[np.isnan(entities_mtx)] = 0


@app.route('/')
def quick_landscape():
    return "Hola - Esta es mi primera Flask API !!"


@app.route('/match',methods=["Get"])
def match_profile():
    """Unsupervised Matcher
    ---
    parameters:  
      - name: id
        in: query
        type: integer
        required: true
      - name: gamma
        in: query
        type: number
        required: true
      - name: top
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    id_profile=request.args.get("id")
    gamma=request.args.get("gamma")
    top=request.args.get("top")

    try:
        pr_emb=profiles_df[profiles_df["id"]==id_profile]["vector_rep"].values[0]#ERROR IN THIS LINE AND I GOT STACKED :**(
        pr_emb=np.reshape(pr_emb, (1, 300))
        pr_emb[np.isnan(pr_emb)] = 0
        
        cosine_distances=cdist(entities_mtx, pr_emb, 'cosine')
        cosine_distances_df=pd.DataFrame(cosine_distances, columns=["distance"])
        
        aux_df=pd.merge(entities_df, cosine_distances_df, left_index=True, right_index=True)
        aux_df=aux_df[aux_df.distance>gamma]
        aux_df=aux_df.sort_values(by='distance', ascending=False)
        match_entities=list(aux_df.sort_values(by='distance', ascending=False)[0:top].id)
        
    except ValueError:
        match_entities=[]
        print("ID has not been found in profile table - Try again please i.e 1866009")	
        
    return "The matched entities are " +str(match_entities) if len(match_entities)>0 else "There is not any match with this profile"


if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)