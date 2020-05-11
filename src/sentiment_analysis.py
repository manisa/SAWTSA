#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


# semantic_factors=["Guestroom","Employee_Interaction","Free_Wi_Fi","Lounge"]


# Creating an empty dictionary 
semantic_factors_words = {} 
  
# # Adding list as value 
# semantic_factors_words["Guestroom"] = ["shower", "bathroom", "bed", "water","place"] 
# semantic_factors_words["Employee_Interaction"] = ["staff", "friendly", "helpful"]  
# semantic_factors_words["Free_Wi_Fi"] = ["free", "Wi-Fi"]  
# semantic_factors_words["Lounge"] = ["coffee", "tea", "pastry","breakfast"]  


semantic_factors=["C1","C2","C3","C4", "C5", "C6", "C7", "C8", "C9", "C10"]

# Adding list as value 
semantic_factors_words["C1"] = ["shower", "water", "toilet", "bathroom"] 
semantic_factors_words["C2"] = ["guide", "knowledgeable"]  
semantic_factors_words["C3"] = ["river", "view"]  
semantic_factors_words["C4"] = ["coffee", "shop"]  
semantic_factors_words["C5"] = ["staff", "helpful", "friendly"]  
semantic_factors_words["C6"] = ["dinner", "breakfast", "lunch","buffets"]  
semantic_factors_words["C7"] = ["tiger"]  
semantic_factors_words["C8"] = ["bar", "pool"]  
semantic_factors_words["C9"] = ["food"]  
semantic_factors_words["C10"] = ["airport", "city"] 




print(semantic_factors_words) 


# In[3]:



import nltk


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()



df= pd.read_csv("data/Combined_data_with_labels.csv",header=0)

reviews=df["Comment"].values
# print(reviews[0])

number_of_reviews=len(reviews)
number_of_features_target=len(semantic_factors)*3+1

column=[]
for word in semantic_factors:
    column.append(word+"_Pos" )
    column.append(word+"_Neg" )
    column.append(word+"_Fact" )
column.append("Overall_review" )    
    
# print(column) 
Features = pd.DataFrame(np.zeros((number_of_reviews, number_of_features_target)),columns=[column])

# print(df["Rating"].values)
# print(Features)
Features[["Overall_review"]]=df["Rating"].values
r=0

for review in reviews:
# this gives us a list of sentences
    sent_text = nltk.sent_tokenize(review) 
#     print(sent_text)


    # now loop over each sentence and tokenize it separately
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence)   
#         print(tokenized_text)
        flag=0
        for factor in semantic_factors:  
            for word1 in semantic_factors_words[factor]:
                for word2 in tokenized_text:
                    if word1.lower()==word2.lower():
#                         print(sentence)
                        if(flag==0):    
                            score = analyser.polarity_scores(sentence)     
                            Features.loc[ r , factor+'_Pos' ]=Features.loc[ r , factor+'_Pos' ]+abs(score['pos'])
                            Features.loc[ r , factor+'_Neg' ]=Features.loc[ r , factor+'_Neg' ]+abs(score['neg'])
                            flag=1
                            
                        Features.loc[ r , factor+'_Fact' ]=Features.loc[ r , factor+'_Fact' ]+1

    r=r+1


# In[5]:


Features.to_csv("features_sentiment_)factors.csv")


# In[7]:


# print(Features)
filter_col_pos_neg = [col for col in column if col.endswith('Pos') or col.endswith('Neg') ]
filter_col_pos_neg.append("Overall_review")
print(Features[filter_col_pos_neg])
Features[filter_col_pos_neg].to_csv("Features1.csv")

filter_col_Factor = [col for col in column if col.endswith('Fact') ]
filter_col_Factor.append("Overall_review")
print(Features[filter_col_Factor])
Features[filter_col_pos_neg].to_csv("Features2.csv")


# In[21]:


# Linear Regresssion
import numpy as np 
from sklearn import datasets
from sklearn import linear_model
# import regressor
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X = Features[filter_col_pos_neg].values[:,:-1]
y = Features[filter_col_pos_neg].values[:,-1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

# print(X)
# print(y)
#scikit 

ols = LinearRegression()
ols.fit(X,y)


# statsmodel
x2 = sm.add_constant(X)
models = sm.OLS(y,x2)
result = models.fit()
file1=open("Regression_results_sentiment_score.txt","w")
print (result.summary(),file=file1)
file1.close()


# In[22]:


# Linear Regresssion
import numpy as np 
from sklearn import datasets
from sklearn import linear_model
# import regressor
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

X = Features[filter_col_Factor].values[:,:-1]
y = Features[filter_col_Factor].values[:,-1]


scaler = StandardScaler()
X = scaler.fit_transform(X)


#scikit 
ols = LinearRegression()
ols.fit(X,y)


# statsmodel
x2 = sm.add_constant(X)
models = sm.OLS(y,x2)
result = models.fit()
file1=open("Regression_results_factor_Wasi.txt","w")
print (result.summary(),file=file1)
file1.close()


# In[23]:


# Linear Regresssion
import numpy as np 
from sklearn import datasets
from sklearn import linear_model
# import regressor
import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv("/home/mpanta1/Wasi/sentiment_analysis_v3/data/data_with_factor_score_2.csv", header=0, low_memory=False)



print("train_df Shape:" ,train_df.shape)



# Split train and target
train = train_df.values

# print(train)
y = train[:,-1]
X = train[:,:-1]

print(y)
print(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)


#scikit 
ols = LinearRegression()
ols.fit(X,y)


# statsmodel
x2 = sm.add_constant(X)
models = sm.OLS(y,x2)
result = models.fit()
file1=open("Regression_results_factors_Manisha.txt","w")
print (result.summary(),file=file1)
file1.close()


# In[ ]:




