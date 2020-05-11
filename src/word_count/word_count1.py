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
#	print(tokens)
	return tokens

def remove_punctuation(text):
    no_punct = " ".join([c for c in text if c not in string.punctuation])
    return no_punct


def remove_stopwords(text):
  
    words =" ".join([w for w in text.split() if w not in stopwords.words('english')])
    return words


def lemmatize_with_postag(text):
    sent = TextBlob(text)
    tag_dict = {"J": 'a', 
                "N": 'n', 
                "V": 'v', 
                "R": 'r'}
    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    
    lemmatized_list = " ".join([wd.lemmatize(tag) for wd, tag in words_and_tags])
    return lemmatized_list


def check_spellings(text):
    checked_spelling = "".join(TextBlob(text).correct())
    return checked_spelling


def main():
	df= pd.read_csv("data/Combined_data.csv",header=0)

	reviews=df["Comment"].values


#	reviews=reviews[0:100]
	output_file_name="review_wise_count_of_selected_words.csv"


	number_of_reviews=len(reviews)
	selected_words= pd.read_csv("selected_words.txt",header=None).values.tolist()



	number_of_words=len(selected_words)
	column=[]
	for word in selected_words:
	    column.append(str(word[0]) )
	print(column) 

	Frequency_table = pd.DataFrame(np.zeros((number_of_reviews, number_of_words)),columns=[column])

	r=0
	for review in reviews:
	    print("No. of review",r)

	#     tokenizer = RegexpTokenizer(r'\w+')
	    tokenized_data = create_tokens(review.lower())
	    punct_data = remove_punctuation(tokenized_data)
	    stop_data = remove_stopwords(punct_data)

	    lemmatized = lemmatize_with_postag(stop_data)
	    correct_data = check_spellings(lemmatized)
	    final_review = create_tokens(correct_data)
#	    print(final_review)



	    for col_word1 in column:
	        for word2 in final_review:
	            if col_word1.lower()==word2.lower():
	#               #  print(tokenized_text)                           
	                Frequency_table.loc[ r , col_word1 ]=Frequency_table.loc[ r , col_word1 ]+1

	    r=r+1

	# print(Frequency_table)
	Frequency_table.to_csv(output_file_name)


if __name__ == '__main__':
	main()




