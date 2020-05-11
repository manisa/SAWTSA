import nltk
#nltk.download() #uncomment this for the first time after you have installed nltk module
import os
import re
import string
import pandas as pd
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from nltk.metrics.distance import( edit_distance, jaccard_distance,)
from nltk.collocations import*
from textblob import TextBlob


def create_tokens(text):
    text = re.sub("\d+", " ", text)
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = [word for token in nltk.sent_tokenize(text) for word in tokenizer.tokenize(token)]
    print(tokens)
    return tokens


def check_spellings(text):
    checked_spelling = "".join(TextBlob(text).correct())
    return checked_spelling


def remove_punctuation(text):
    no_punct = " ".join([c for c in text if c not in string.punctuation])
    return no_punct


def remove_stopwords(text):
  
    words =" ".join([w for w in text.split() if w not in stopwords.words('english')])
    return words

#Normalization of data
def word_lemmatizer(text):
    lem = WordNetLemmatizer()
    lemmatize_text =[lem.lemmatize(i) for i in text]
    return lemmatize_text

def lemmatize_with_postag(text):
    sent = TextBlob(text)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = " ".join([wd.lemmatize(tag) for wd, tag in words_and_tags])
    return lemmatized_list

def word_stemmer(text):
    porter = nltk.PorterStemmer()
    stem_text = [porter.stem(i) for i in text]
    return stem_text 

# def word_pos_tagger(text):
#     pos_tagged_array = []
#     pos_tagges_text = nltk.pos_tag(text)
#     for tag in pos_tagges_text:
#         match = re.search('\w.*', tag[1])
#         if match:
#             pos_tagged_array.append(tag)

#     return pos_tagged_array


def get_data_for_preprocessing(hotel_name):
    #header = ['review_title', 'review', 'review_date','rate']
    file_path = os.getcwd() + '/'+ 'data'+ '/' 
    df = pd.read_csv( file_path + hotel_name, header=None,engine='python')
    #df = df.sample(frac=1).reset_index(drop=True)
    return df
  
    
def preprocess_df(review_df):
    tokenizer =nltk.RegexpTokenizer(r"\w+")


    tokens = review_df.apply(lambda x: tokenizer.tokenize(x.lower()))
    print(tokens.head())
    
    punct_df = tokens.apply(lambda x: remove_punctuation(x)) #for individual reviews
    print(punct_df.head())
        
    stop_df = punct_df.apply(lambda x: remove_stopwords(x)) #removing unnecessary words such as "is, am, are, the"
    print(stop_df.head())

    lemmatized_df = stop_df.apply(lambda x: word_lemmatizer(x))
    print(lemmatized_df.head())


    #stemmed = word_stemmer(stop_data) #word reduction algorithm to get base words
    #print(stemmed)
    return lemmatized_df

    

def preprocess_data(data):
    
    #correct_spellings = words.words()

    #correct_data = data.apply(lambda x: check_spellings(x))
    #print(correct_data.head())

    #tokenizer = nltk.RegexpTokenizer(r"\w+")
    
    #tokens = data.apply(lambda x: tokenizer.tokenize(x.lower()))
    

    tokens = data.apply(lambda x: create_tokens(x.lower()))

    
    print(tokens)
    punct_data = tokens.apply(lambda x: remove_punctuation(x)) #removing unnecessay punctuations 
    print(punct_data.head())
    
    stop_data = punct_data.apply(lambda x : remove_stopwords(x)) #removing unnecessary words such as "is, am, are, the"
    print(stop_data.head())

    lemmatized = stop_data.apply(lambda x: lemmatize_with_postag(x))

    
    #lemmatized = stop_data.apply(lambda x : word_lemmatizer(x)).sum() #lemmatization
    lemmatized_new_df = lemmatized.apply(lambda x: create_tokens(x))

    correct_data = lemmatized.apply(lambda x: check_spellings(x))

    lemmatized_new = correct_data.apply(lambda x: create_tokens(x)).sum()


    
    #lemmatized = word_lemmatizer(stop_data).sum()
    #stemmed = stop_data.apply(lambda x : word_stemmer(x)).sum() #word reduction algorithm to get base words
    
    return lemmatized_new, lemmatized_new_df

def counting_tokens_for_df(text):
    lem_counter = Counter(text)
    common_words = lem_counter.most_common()
    return common_words

def counting_tokens_allReviews(text):
    lem_counter = Counter(text)
    common_words = lem_counter.most_common(400)
    return common_words

def main():
    
    allHotelReviews = pd.DataFrame()
    with open('./id_list.txt', 'r') as filelist:
        allFiles = filelist.readlines()
        
        for fname in allFiles:
            data_to_file = pd.DataFrame()
            fname = fname.strip('\n')
            
            df = get_data_for_preprocessing(fname) 
            print(df.head(10))  
            review_title = df.iloc[:,0]
            review_rate = df.iloc[:,3]
            review_data = df.iloc[:,1]
            review_target = df.iloc[:,2]
            
            print(review_data.shape)
            print(review_title.head())
            print(review_rate.head())
            print(review_data.head())
            
            # processed_df = preprocess_df(review_data)
            
            # data_to_frame = pd.concat([review_title, processed_df, review_rate], axis=1)
            # data_to_frame.to_csv(output_filepath + '/' + str(fname), index=False)
            
            # df_counted = processed_df.apply(lambda x: counting_tokens_for_df(x))
            # df_counted.to_csv(output_filepath + '/' + 'count_' + str(fname), index=False)
            
            #allHotelReviews = pd.concat([allHotelReviews,df], axis=0)

        
        print("########################################################################3")
        print(review_data.head())
        processed_data, processed_df = preprocess_data(review_data)
        data_to_frame = pd.concat([review_title, processed_df, review_rate, review_target], axis=1)
        data_to_frame.to_csv(output_filepath + '/' + str(fname), index=False)
            
    return processed_data, review_data.size

if __name__=='__main__':
    output_filepath = 'results'
    if not os.path.isdir(output_filepath):
        os.mkdir(output_filepath)
    
    output_filename = 'result_total_count' + '.txt'
    output_best_80 = 'result_80_most_common' + '.txt'
    output_all_words = 'result_all_words' + '.txt'
    best_50 = 'best_50_bigram_trigram' + '.txt'
    
    
    fd_all_words= open(os.path.join(output_filepath, output_all_words), 'w')
    
    fd_total_count = open(os.path.join(output_filepath, output_filename), 'w')
    
    fd_count = open(os.path.join(output_filepath, output_best_80), 'w')
    
    fd_best_50 = open(os.path.join(output_filepath, best_50), 'a')
    
    
    processed_data, total_reviews = main()
    count=0
    for i in range(len(processed_data)):
        count +=1
        fd_all_words.write(str(processed_data[i]))
        fd_all_words.write(',')
    fd_all_words.write('\n\n\n' + str(count))
    
    
    best_80_counts = counting_tokens_allReviews(processed_data)
    
    for i in range(len(best_80_counts)):
        fd_count.write(str(best_80_counts[i]))
        fd_count.write('\n')
        

    
    #for i in range(len(words_count)):
     #   fd_total_count.write(str(words_count[i]))
      #  fd_total_count.write('\n')
        
    #words_count.plot(25, cumulative=False)
    
    
    lem_counter = Counter(processed_data)
    words_count = lem_counter.most_common()
    rank =0
    fd_total_count.write("rank" + "," + "word" + "," + "frequency" + "," + "frequency per review" + "," + "percentage")
    fd_total_count.write("\n")
    for i in range(len(words_count)):
        rank += 1
        word = words_count[i][0]
        freq = words_count[i][1]
        freq_per_review = freq / total_reviews
        percent = (freq / len(processed_data))  * 100
        
        fd_total_count.write(str(rank) + ',' + str(word) + ',' + str(freq) + ',' + str(freq_per_review) + ',' + str(percent))
        fd_total_count.write('\n')

    
    
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder2 = BigramCollocationFinder.from_words(processed_data,5)
    finder2.apply_freq_filter(3)
    best_finder2 = finder2.nbest(bigram_measures.likelihood_ratio, 400)

    for i in range(len(best_finder2)):
        fd_best_50.write(str(best_finder2[i]))
        fd_best_50.write('\n')
    
    
    #best_finder2.to_csv(output_filepath + '/' + best50, index=False)

    
    fd_best_50.write('\n\n\n')
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    finder3 = TrigramCollocationFinder.from_words(processed_data, 5)
    best_finder3 = finder3.nbest(trigram_measures.likelihood_ratio, 400)
    
    for i in range(len(best_finder3)):
        fd_best_50.write(str(best_finder3[i]))
        fd_best_50.write('\n')
    
    fd_all_words.close()
    fd_total_count.close()
    fd_count.close()
    fd_best_50.close()
    
    
    #scored = finder.score_ngrams(trigram_measures.raw_freq)

    #print("\n \n")
    #print(finder3.nbest(trigram_measures.likelihood_ratio, 20))