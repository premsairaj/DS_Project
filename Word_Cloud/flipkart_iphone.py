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