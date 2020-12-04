from bs4 import BeautifulSoup as bsp
from selenium import webdriver #Programmatic way to use Browser
#from selenium.webdriver.common.keys import Keys
from webdriver_manager.firefox import GeckoDriverManager
import time
import pickle
import os
import re
import csv
from IPython.display import clear_output
import pandas as pd
import numpy as np
import nltk as nl
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import WordNetLemmatizer
from scipy import spatial
from collections import OrderedDict

driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())


#Fucntion to find Book URL's
def findUrls(href,driver):
    driver.get(href)
    time.sleep(1)
    soup = bsp(driver.page_source,features='lxml')
    list_urls = soup.find_all('a',{"class":"bookTitle"},itemprop='url')
    urls = []
    for url in list_urls:
        urls.append(url.get("href"))
    return urls


def findBookUrls(href,driver,pageRange):
    Urls_Dic = {}
    change = 1
    for pageNumber in range (1,pageRange+1):
            if change != len(str(pageNumber)):
                change = len(str(pageNumber))
                href = href[:-(len(str(pageNumber))-1)] + str(pageNumber)
                Urls_Dic["page"+str(pageNumber)] = findUrls(href,driver)
                #Urls_List.append(findUrls(href,driver))
                print(href)
                continue
            href = href[:-len(str(pageNumber))] + str(pageNumber)
            Urls_Dic["page"+str(pageNumber)] = findUrls(href,driver)
            #Urls_List.append(findUrls(href,driver))
            clear_output(wait=True)
            print(href, "Done !!")
    return Urls_Dic


#Crawl and Download HTML books
def carwlBooks(FilePath,Start,End):
    with open(r'C:\{}\300PagesDic.pkl'.format(FilePath), 'rb') as handle:
        pagesDict = pickle.load(handle)
    testCount = 0
    pagesDict = dict(list(pagesDict.items())[Start-1:End]) 
        
    dir = os.path.join("C:\\",FilePath,"Best Book Ever")
    if not os.path.exists(dir):
        os.mkdir(dir)     
        
    file_location = r"{}\Best Book Ever".format(FilePath)
    for pageFolder in pagesDict:
        dir = os.path.join("C:\\",file_location,pageFolder)
        if not os.path.exists(dir):
            os.mkdir(dir)
        Urls_L = pagesDict[pageFolder]
        count = 0
        
        for urls in Urls_L:
            if testCount == 3:
                break
            testCount += 1
            href = "https://www.goodreads.com/" + urls
            driver.get(href)
            HTMLFileName = pageFolder+"_"+str(count)
            with open(r'C:\{}\Best Book Ever\{}\{}.html'.format(FilePath,pageFolder,HTMLFileName),'w+', encoding="utf-8") as Page:
                Page.write(driver.page_source)
            print(pageFolder+"==> [File:"+str(count)+" Downloaded]")
            count = count + 1
            time.sleep(1)
            
            
#Parse Downloaded Pages
def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    return TAG_RE.sub('', text)

def Parse_Web_Pages(href):
    cache = {}
    soup = bsp(open(href, encoding="utf-8"),features="lxml")
    CheckInfoBoxDetails = soup.find_all('div',{"class":'infoBoxRowTitle'})
    CheckInfoBoxDetailsList = []
    for i in CheckInfoBoxDetails:
        if i.contents[0] != '\n':
            CheckInfoBoxDetailsList.append(i.contents[0])

    #Find Book Title
    bookTitle = soup.find_all('h1',itemprop='name')
    cache['bookTitle'] = bookTitle[0].contents[0].split('\n')[1].strip()
    
    #Find Book Series
    bookSeries = soup.find_all('h2',{'id':'bookSeries'})
    if len(bookSeries[0].contents) > 1:
        bookSeries = bookSeries[0].contents[1].contents[0].split('\n')[1].split('(')[1].split(')')[0]
        cache['bookSeries'] = bookSeries
    else:
        bookSeries = None
        cache['bookSeries'] = bookSeries
    
    #Find Book Author
    bookAuthor = soup.find_all('span',itemprop='name')
    if len(bookAuthor)>1:
        bookAuthorList = []
        for i in bookAuthor:
            bookAuthorList.append(i.contents[0])
        bookAuthor = bookAuthorList
        cache['bookAuthors'] = bookAuthor
    else:
        bookAuthor = bookAuthor[0].contents[0]
        cache['bookAuthors'] = bookAuthor
    

    #Find Book Ratings Stars
    if len(soup.find_all('span',itemprop='ratingValue')) >= 1:
        bookRatingStars = soup.find_all('span',itemprop='ratingValue')
        cache['ratingValue'] = bookRatingStars[0].contents[0].split('\n')[1].strip()
    else:
        cache['ratingValue'] = None
    
    #Find Book Given Ratings
    if len(soup.find_all('meta',itemprop='ratingCount'))>=1:
        bookGivenRating = soup.find_all('meta',itemprop='ratingCount')[0].get("content")
        cache['ratingCount'] = bookGivenRating
    else:
        cache['ratingCount'] = None
    
    #Find Book Reviews
    if len(soup.find_all('meta',itemprop='reviewCount'))>=1:
        bookReviews = soup.find_all('meta',itemprop='reviewCount')[0].get("content")
        cache['reviewCount'] = bookReviews
    else:
        cache['reviewCount'] = None

    #Find Book Plot from 5 - 1 Stars (Dictionary)
    plot = soup.find_all('script',{"type":"text/javascript+protovis"})
    if not plot :
        cache['Plot_Values'] = None
    else:
        plotData = plot[0].contents[0].split('\n')[1]
        numbers = re.findall('[0-9]+', plotData)
        plotDic = {}
        count = 5
        for i in numbers:
            plotDic[count] = i
            count = count -1
        cache['Plot_Values'] = plotDic
    
    #Find Description
    actuall_des = []
    #sentence = None
    description = soup.find_all('div',id="description")
    if len(description) >= 1:
        for descrip in description[0].contents:
            if remove_tags(str(descrip)) != '\n' and remove_tags(str(descrip)) != '...more':
                actuall_des.append(remove_tags(str(descrip)))
        if len(actuall_des)>=2:
            if actuall_des[0] in actuall_des[1]:
                actuall_des = ''.join(actuall_des[1:])
            else:
                actuall_des = ''.join(actuall_des)
            cache['Plot'] = actuall_des
        elif len(actuall_des)==1:
            actuall_des = actuall_des[0]
            cache['Plot'] = actuall_des
        else:
            actuall_des = None
            cache['Plot'] = actuall_des
    else:
        cache['Plot'] = None
    
    #Find Book number of pages
    if len(soup.find_all('span',itemprop='numberOfPages')) >= 1:
        numberOfPages = soup.find_all('span',itemprop='numberOfPages')[0]
        numberOfPages = re.findall('[0-9]+', numberOfPages.contents[0])
        cache['NumberofPages'] = numberOfPages[0]
    else:
        cache['NumberofPages'] = None
    
    #Find Book Published Date
    if len(soup.find_all('div',{"class":'row'}))>1:
        publishedDate = soup.find_all('div',{"class":'row'})[1]
        if publishedDate.contents[0] != '\n':
            cache['Publishing_Date'] = publishedDate.contents[0].split('\n')[2].strip()
        else:
            cache['Publishing_Date'] = None
    else:
        cache['Publishing_Date'] = None
        
    #Find Book charactors (List)
    if 'Characters' in CheckInfoBoxDetailsList:
        char_index = CheckInfoBoxDetailsList.index('Characters')
        charactors = soup.find_all('div',{"class":'infoBoxRowItem'})
        charactorsList = []
        for i in charactors[char_index].find_all(['a']):
            if i.contents[0] != "...more" and i.contents[0] != "...less":
                charactorsList.append(i.contents[0])
    else:
        charactorsList = None
    cache['Characters'] = charactorsList

    #Find Book settings (List)
    if 'Setting' in CheckInfoBoxDetailsList:
        settings_index = CheckInfoBoxDetailsList.index('Setting')
        settings = soup.find_all('div',{"class":'infoBoxRowItem'})
        settingsList = []
        for item in settings[settings_index].find_all('a'):
            if item.contents[0] != "…more" and item.contents[0] != "…less":
                settingsList.append(item.contents[0])
    else:
        settingsList = None
    cache['Setting'] = settingsList

    #Find Book URL (Url)
    Url = soup.head.link.get('href')
    cache['Url'] = Url
    
    return cache


def Parse_Data(File_path,Start,End):
    with open(r"{}\TestDataset_{}Pages.tsv".format(File_path,str(End)), 'w', newline='', encoding='utf-8') as f_output:
        data = [] 
        Data = Parse_Web_Pages(href = r"{}\Best Book Ever\page{}\page{}_0.html".fomat(File_path,str(Start),str(Start)))
        for i in Data:
                data.append(i)
        tsv_output = csv.writer(f_output, delimiter='\t')
        tsv_output.writerow(data)
        
        for pageNumber in range(Start,End + 1):
            for fileNumber in range(0,100):
                print("Page: "+str(pageNumber),"File :"+str(fileNumber),"Added to TSV file !!!")
                href = r"{}\Best Book Ever\page{}\page{}_{}.html".format(File_path,pageNumber,pageNumber,fileNumber)
                if os.path.getsize(href)/1024 > 200:
                    Data = Parse_Web_Pages(href)
                    data = [] 
                    for i in Data:
                        data.append(Data[i])
                tsv_output.writerow(data)
                clear_output(wait=True)
                

def read_data():
    dataset = pd.read_csv("Dataset_final_filtered.tsv",sep='\t')
    
    infile = open("vocabulary.pkl",'rb')
    vocab = pickle.load(infile)
    infile.close()
    infile = open("ID_with_tfidf.pkl",'rb')
    ID = pickle.load(infile)
    infile.close()
    return dataset,vocab,ID


def clean_info1(string):
    #first we lower all the words otherwise words such as AND,IS,MY are not consider stopwords 
    tmp = [word.lower() for word in string]
     # filter the stopwords (e.g. 'the', 'my', 'an', ...)
    tmp = [word for word in tmp if not word in stopwords.words("english")]
    #we lemmatize all the words (e.g. 'dirn')
    lemma = WordNetLemmatizer()
    tmp = [lemma.lemmatize(word, pos = "v") for word in tmp]    # v for verbs
    tmp = [lemma.lemmatize(word, pos = "n") for word in tmp]    # n for nouns
    final = ' '.join(tmp)
    return final

def find_set(a,b):
    return set(a) & set(b)


def executeQuery(query,vocabulary,ID,dataset):
    lists = {}
    docs = {}
    for i in query:
        lists[i] = ID.get(vocabulary[i])
        docs[i] = [x[0] for x in lists[i][:]]
        
    data = None
    flag = True
    for i in docs:
        if flag:
            flag = False
            data = docs[i]
            continue
        data = find_set(data,docs[i])
        
    data_withVal = {}
    for i in lists:
        data_withVal[i] = [x for x in lists[i][:] if x[0] in data]
        
    data_flat = []
    for x in data_withVal.values():
        data_flat.extend(x)
        
    res = OrderedDict()
    for v in data:
        val = [cos[1] for cos in [x for x in data_flat if x[0]==v]]
        score = spatial.distance.cosine(len(query),val)
        res[v] = score
        
    sorted_books = sorted(res.items(), key=lambda x:x[1],reverse=True)
    data_index = []
    data_score = []
    for i,j in sorted_books:
        data_index.append(i)
        data_score.append(j)
    
    founddata = dataset.iloc[data_index,:][['bookTitle','Plot','Url']][:][0:5]
    founddata.insert( 3, "Similarity", data_score[0:5], True) 
    
    return founddata



