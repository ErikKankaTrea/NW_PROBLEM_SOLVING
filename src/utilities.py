# -*- coding: utf-8 -*-
import numpy as np
def fill_nan_with(data_df, var, value=0, FUN=None, perc=None):
    """Count of NAN and replacement of them by
	:data_df: your pandas data frame 
	:var: variable to be modified "var1"
	:value: value to be added
	:FUN: instead of adding a value yourself, you can add "mean", "median" or "percentile" of that variable
	:perc: value of the percentile when FUN="percentile", range [0-1]
    """
    num_nan=data_df[[var]].isnull().sum().values[0]
    print("Count of NAN values in {x}: {y}".format(x=var, y=num_nan))
    
    if FUN is None and num_nan>0:
        try:
            data_df[var] = data_df[var].replace(np.nan, value)        
            print("----> Replaced by {}".format(value))
        except Exception as error:
            print(error)
            
    elif FUN is not None and num_nan>0:
        try:
            if FUN=="mean":
                stat=np.mean(data_df[var])
                data_df[var] = data_df[var].replace(np.nan, stat)
            elif FUN=="median":
                stat=np.nanmedian(data_df[var])
                data_df[var] = data_df[var].replace(np.nan, stat)
            elif FUN=="percentil":
                try:
                    stat=np.nanpercentile(data_df[var], perc)
                    data_df[var] = data_df[var].replace(np.nan, stat)
                except Exception as error:
                    #WARNING ERROR
                    print(error)
            print("----> Replaced by {} with value {}".format(FUN, stat))
        except Exception as error:
            #WARNING ERROR
            print(error) 
            
    else:
        print("----> Nothing replaced")
    pass



def canonize_language(df, text_var):
    """
    v1 of canoniza language
    """
    symbols_to_space = re.compile(u"[/\|\n(\; )|(\: )|( \()|(\) )|( \")|(\" )|( \')|(\' )]")
    symbols_to_remove = re.compile(u"[\"\'\$\€\£\(\)\:\[\]\.\,\-]\&")
    space_repetition = re.compile(u" {2,}")
    key_words_to_remove = re.compile(u"gmbh")
    text_var_clean= text_var+'_clean'
    cleaned_var=df[text_var].apply(lambda x: re.sub(symbols_to_remove, "", x))
    cleaned_var=cleaned_var.apply(lambda x: x.lower())
    cleaned_var=cleaned_var.apply(lambda x: re.sub(key_words_to_remove, "", x))
    cleaned_var=cleaned_var.apply(lambda x: unicodedata.normalize('NFKD', x))
    cleaned_var=cleaned_var.apply(lambda x: re.sub(space_repetition, "", x))
    cleaned_var=cleaned_var.apply(lambda x: str.strip(x))
    cleaned_var=cleaned_var.apply(lambda x: re.sub(u"\s", "_", x))
    return cleaned_var


def match_checker(df1, df2, key, aux_how):
    """
    summary of joining tables
    """
    df2["aux"] = 1
    aux_df = pd.merge(df1, df2, on = key, how = aux_how)
    aux_df["aux"] = aux_df.aux.fillna(0)
    num_ids_match= np.sum(aux_df.aux)
    print("{} var has {} unique values".format(key, len(np.unique(df1[key]))))
    print("{} var has {} unique values".format(key, len(np.unique(df2[key]))))
    print("Matches {} id rows of entities in profiles".format(num_ids_match))
    pass



def direct_matcher(df1, df2, key, aux_how):
    """
    Simple join
    """
    aux_df = pd.merge(df1, df2, on = key, how = aux_how)
    aux_df=aux_df[["id_x", "id_y"]]
    aux_df.rename(columns={"id_y": "id_profiles", "id_x": "id_entities"}, inplace=True)
    return(aux_df[["id_profiles", "id_entities"]])


def eval_fun(profile_ids, aux_vars_ids, matched_df):
    # The evaluation will count if all of the profiles have at least 1 entity
    # Global and local mean of % matching entitites with profile.
    gt_df = pd.read_csv(os.path.join(get_data_path(), "ground_truth.tsv"), 
                                names=aux_vars_ids,
                                encoding="utf-8", 
                                sep="\t") # .fillna({"text": "empty"})


    # First check: ¿At least all the profiles have one entity?
    at_least_all=len(np.unique(gt_df[aux_vars_ids[0]])) == len(np.unique(matched_df[aux_vars_ids[0]]))
    if at_least_all:
        print("1. All profiles have at least one entity")
    else:
        print("1. Not all profiles have one entity")
    # Second is to measure the percentages:
    res_df = pd.DataFrame() # DF with a tuple [a, b, c] where a= # of entitites assigned, b= # of real assigned, c= #matches 
    for i_id in profile_ids:
        entities_assigned=matched_df.loc[matched_df["id_profiles"]==i_id].id_entities.values
        entities_assigned = entities_assigned[~np.isnan(np.array(entities_assigned, dtype=np.float64))]
        num_assigned= len(entities_assigned)
        real_entities_assigned=gt_df.loc[gt_df["id_profiles"]==i_id].id_entities.values
        real_entities_assigned = real_entities_assigned[~np.isnan(np.array(real_entities_assigned, dtype=np.float64))]
        real_num_assigned=len(real_entities_assigned)
        num_matches=len(set(entities_assigned) & set(real_entities_assigned))
        per_match= round(num_matches/real_num_assigned, 2)*100
        aux_tuple=[i_id, num_assigned, real_num_assigned, per_match]
        aux_df=pd.DataFrame([aux_tuple], columns=["id_profile", "num_entities_matched", "real_entitites_matched", "per_match"])
        res_df=res_df.append(aux_df) 
    return(res_df)



# Generate annotation weight representation based on german pre-trained word2vec model 
def annotation_weight_representation(value_annotation):
    """
    Function that iterates a long a list vocab. Look up for its embedding. Make weighted avg of embeddings 
    """
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


def canonize_language_2(df, text_var):
    """
    Function to clean company_name variable in particular
    """
    symbols_to_space = re.compile(u"[/\|\n(\; )|(\: )|( \()|(\) )|( \")|(\" )|( \')|(\' )]")
    symbols_to_remove = re.compile(u"[\"\'\$\€\£\(\)\:\[\]\.\,\-]\&")
    space_repetition = re.compile(u" {2,}")
    key_words_to_remove = re.compile(u"gmbh")
    cleaned_var=df[text_var].apply(lambda x: re.sub(symbols_to_remove, "", x))
    cleaned_var=df[text_var].apply(lambda x: re.sub("GmbH|gmbh|mbH|mbh|&|und|\-|an|der|co|Co|\.|\-", "", x))
    cleaned_var=cleaned_var.apply(lambda x: re.sub(space_repetition, "", x))
    cleaned_var=cleaned_var.apply(lambda x: str.strip(x))
    return cleaned_var


def load_german_model():
    """
    Load w2vec
    """
    return(gensim.models.KeyedVectors.load_word2vec_format(os.path.join(get_data_path(), "german.model") ,binary=True))

def put_space(aux_input): 
    """
    Split by space between big and small characters.
    """
    words = re.findall('[A-Z][a-z]*', aux_input) 
    result = [] 
    for word in words: 
        word = chr( ord (word[0]) + 32) + word[1:] 
        result.append(word) 
    return(result)

def remove_word_elem(aux_input):
    """
    Filter from list strings with less than n characters.
    """
    result=[i_element for i_element in aux_input if len(i_element)>1]
    return(result)
    
def convert_list_to_string(org_list, seperator=' '):
    """ Convert list to string, using join """
    return seperator.join(org_list)
