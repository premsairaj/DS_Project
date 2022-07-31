# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 21:06:34 2022

@author: ppallapotu
"""
import requests
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud   
from selenium import webdriver
from bs4 import BeautifulSoup
from requests import get
from selenium.webdriver.common.by import By
driver=webdriver.Chrome("C:\\Users\\ppallapotu\\Downloads\\chromedriver.exe")
iphone_review=[]
for i in range(1,10):
    ip=[]
    url = "https://www.flipkart.com/apple-iphone-13-midnight-128-gb/product-reviews/itmca361aab1c5b0?pid=MOBG6VF5Q82T3XRS&lid=LSTMOBG6VF5Q82T3XRSOXJLM9&marketplace=FLIPKART&page="+str(i)
    driver.get(url)
    s=driver.find_elements(By.XPATH,'//span[text()="READ MORE"]')
    for j in s:
        driver.execute_script("arguments[0].click();", j)
    review=driver.find_elements(By.XPATH,'//div[@class="t-ZTKy"]//div//div')
    for r in review:
        ip.append(r.text)
    iphone_review.append(ip)   

corpus=''
for k in iphone_review[:4]:
     corpus=corpus.join(k)
with open('iphone_review.txt','w',encoding='utf-8') as output:
    output.write(corpus)
from nltk.corpus import stopwords
import re
sp=stopwords.words('English')
sp.extend(['day','iphone','android'])
corpus=re.sub(r'[0-9]+', '', corpus).lower()

wordcloud_iphone=WordCloud(background_color='white',stopwords=sp,width=1800,height=1400).generate(corpus)
plt.imshow(wordcloud_iphone,interpolation='bilinear')
plt.axis('off')
plt.show()



###########bigram word cloud##########
import nltk
feedback=open(r'C:\Users\ppallapotu\DS_Project\Word_Cloud\iphone_review.txt','r',encoding='utf-8').read()
feedback=re.sub(r'[^A-Za-z ]+','',feedback).lower()
sp=stopwords.words('English')
sp.extend(['day','iphone','android','the','a'])
corpus=nltk.word_tokenize(feedback)
corpus=[i for i in corpus if i not in sp]
bigram_corpus=list(nltk.bigrams(corpus))
dictonary=[' '.join(tup) for tup in bigram_corpus]
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(ngram_range=(2,2))
bag_of_words=cv.fit_transform(dictonary)
cv.vocabulary_
sum_words=bag_of_words.sum(axis=0)
word_freq=[(word,sum_words[0,idx]) for word,idx in cv.vocabulary_.items()]
word_freq=sorted(word_freq,key=lambda x:x[1],reverse=True)
word_dict=dict(word_freq)
wordcloud_iphone_bigram=WordCloud(max_words=200,background_color='white',stopwords=sp,width=1800,height=1400).generate_from_frequencies(word_dict)
plt.imshow(wordcloud_iphone_bigram,interpolation='bilinear')
plt.axis('off')
plt.show()



