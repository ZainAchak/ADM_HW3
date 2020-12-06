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
import heapq
from collections import defaultdict
from langdetect import detect
import copy
import seaborn as sns
import matplotlib.pyplot as plt

#driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())


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
                
#special function for filtering english plots given the entire dataset
#we detect all not english plots and drop them for the dataset
#the function return an english dataset

def english_plots(df):
    plots=df['Plot']  
    for i in range(len(plots)):
        try:
            if detect(plots[i])=='en':
                continue
            else:
                df.drop(i,inplace=True)
        except:
            df.drop(i,inplace=True)
    return df

# exercise 2
''' 
Create inverted index
'''


def read_data():
    dataset = pd.read_csv("Dataset_final_filtered.tsv",sep='\t')
    
    infile = open("vocabulary.pkl",'rb')
    vocab = pickle.load(infile)
    infile.close()
    infile = open("Inverted_Index.pkl",'rb')
    ID = pickle.load(infile)
    infile.close()
    infile = open("ID_with_tfidf.pkl",'rb')
    ID_tfidf = pickle.load(infile)
    infile.close()
    return dataset,vocab,ID,ID_tfidf

def clean_info (string):
    # this command will split the string given in input in substrings by using 
    # the words given to RegexpTokenizer as argument
    

    
    # filter the punctuation
    tmp = nl.RegexpTokenizer(r"['\w-]+").tokenize(string)  
    
    #first we lower all the words otherwise words such as AND,IS,MY are not consider stopwords 
    tmp = [word.lower() for word in tmp]
    
     # filter the stopwords (e.g. 'the', 'my', 'an', ...)
    tmp = [word for word in tmp if not word in stopwords.words("english")]
    
    #we lemmatize all the words (e.g. 'dirn')
    lemma = WordNetLemmatizer()
    tmp = [lemma.lemmatize(word, pos = "v") for word in tmp]    # v for verbs
    tmp = [lemma.lemmatize(word, pos = "n") for word in tmp]    # n for nouns
    
    
    final = ' '.join(tmp)
    
    return final
# we will need first these two functions to implement the search function 


# this function takes as input a list of lists and gives back the index of the list that has minimus first element

def find_min_list (L):
    min_elem = L[0][0]
    count = 0
    index = 0
    for l in L[1:]: 
        if min_elem > l[0]:
            count = count + 1 
            index = count 
            min_elem = l[0]
        else: 
            count = count + 1 

    return index 
        

# this function takes as input a list of lists and gives back the list created from the intersection of the lists 
    
def intersect_list (L):
    results = []
    while all(len(l) > 0 for l in L):

        if all([L[0][0] == l[0] for l in L[1:]]):     
            results.append(L[0][0])
            L = [l[1:] for l in L]

        else : 
            min_index = find_min_list(L)
            L[min_index] = L[min_index][1:]
    return results

def find_query(inverted_index, vocabulary):
    
    string = input('Search : ') # asks the user a string of words to look up 
    
    string_cleaned = clean_info(string)
    list_words = string_cleaned.split(' ')
    
    
    # now translate the list of words in term_id )
    list_termID = []
    for word in list_words: 
        list_termID.append(vocabulary.get(word))
        
    if list_termID==[None]:
        print('The word you were looking for was not found,please enter a new one')
        return find_query(inverted_index, vocabulary)
        
   

    # retrieve the documents in the inverted index and collect them in a list 
    list_documents = []
    for term_id in list_termID:
        if term_id in inverted_index.keys():   # checking if the word we are looking for is in the inverted_index
            list_documents.append(inverted_index.get(term_id))
    
     
    # now intersect these lists (HERE IT'S FUNDAMENTAL TO SUPPOSE THAT THE DOCUMENTS ARE COLLECTED AS INCREASING 
    # SEQUENCES)
    results = intersect_list(list_documents)
    
    
    return results


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
    
    c_score=[]
    for v in data:
        val = [cos[1] for cos in [x for x in data_flat if x[0]==v]]
        score = spatial.distance.cosine(len(query),val)
        c_score.append([score,v])
        
    #create a heap data structure          
    H=[]
    for i in range(len(c_score)):
        heapq.heappush(H,(c_score[i][0],c_score[i][1]))
    k=5
    top_k = heapq.nlargest (k, H)
    
    score_cosine_similarity_top_k=[]
    document_ID_top_k=[]
    for i in top_k:
        score_cosine_similarity_top_k.append(i[0])
        document_ID_top_k.append(i[1])
        
    df_top_k=dataset.loc[document_ID_top_k][['bookTitle', 'Plot', 'Url']]
    df_top_k['new_similarity_score']=score_cosine_similarity_top_k
       
    return df_top_k



#exercise 3
'''
Belowe we group all the functions used for the exercise 3.
The output is the top 5 books oredered by the new score defind by us,which 
is embedded in the variable new_score
In particular we ask the user some further information:
the author ,the number of pages ,the years of publication ,the title.
The user can decide to insert them or not and leave the field blank.
If the books we find from the exercise 2.1 contains this new information we add points to these books.
The score includes also the rating and number of ratings ,very important for the popularity of the books.
We calculate this partial score using the logarithm as well as the tf_idf ,to avoid to penalize the recently 
published books which have few number of ratings.Using the logarithm we reduce the gap
between numbers with order of magnitude too different.
'''
def find_authors (df):
    authors = []
    
    for x in df.bookAuthors:
        
        if '[' in x:                # case with multiple authors
            list_authors = x.split(',')
    
            
            for author in list_authors[:-1]:
                authors.append(author[2:-1])
            authors.append(list_authors[-1][2:-2])
        
        else: 
            authors.append(x)
    
    return authors
def find_authors_popularity (list_authors_small, list_authors_big):
    
    dict_authors = {x:list_authors_big.count(x) for x in list_authors_small}
    
    return dict_authors
def year(y):
    try:
        y.split(' ')[-1]
        return y
    except:
        return 0
def new_score_func(results_query,df):
    
    #initialize the new_score
    new_score=[]
    for d in results_query:
        new_score.append([0,d])
    
    #rows that we need from the dataset
    dataset_query=df.iloc[results_query]
    n=len(dataset_query)
    
    plots=dataset_query['Plot']
    
    titles=dataset_query.apply(lambda x : x['bookTitle'].lower(),axis=1)
    
    pages=dataset_query['NumberofPages']
    
    autors=dataset_query['bookAuthors']
    
    years_pub=dataset_query.apply(lambda x : year(x),axis=1)
    
    # 1.score based on rating value and rating count
    #we add a score according to the product of the rating value and the rating count
    #we take its logarithm to avoid too different gap 
    #and for not penalizing the recent books which have few rating count
    #then we normalize all the score
    norm=0
    for x in new_score:
        tmp = dataset_query.loc[x[1]]['ratingValue']*dataset_query.loc[x[1]]['ratingCount']
        norm+=tmp
        if tmp < 1: 
            x[0] += 0
        else: 
            x[0] += np.log10(tmp)
    for x in new_score:
        x[0]=x[0]/norm
    
    # 2. score based on title 
    query_title=str.lower(input('1. What is the title of the book? Enter the keywords of the title you are looking for, otherwise press enter (e.g. Harry)\n').strip(".,;'"))
    
    #we add a score according to the number of words of the title found
    if query_title!='':
        query_title=query_title.split(' ')
        n_query=len(query_title)
        for i in range(n):
            count=0
            for q in query_title:
                if q in titles.iloc[i]:
                    count+=1
            new_score[i][0]+=count/n_query
        
    # 3. score based on number of pages
    
    range_pagine=input('2. How many pages has the book you are looking for?\
Enter a range of pages, otherwise hit enter (e.g. 200-400)\n')
    
    if range_pagine!='':
        minimo_p,massimo_p=range_pagine.split('-')
        
        # if the user exchange minimum and maximum
        if minimo_p>massimo_p:
            minimo_p,massimo_p=massimo_p,minimo_p
            
        # if the number of pages falls in the range we add a score
        for i in range(n):
            if int(minimo_p)<pages.iloc[i]<int(massimo_p):
                new_score[i][0]+=0.5
        
    # 4. score based on year of publication 
    
    range_years=input('3. When the book was published?\
Enter a range of years, otherwise hit enter (e.g. 1990-2010)\n')
    
    if range_years!='':
        minimo_y,massimo_y=range_years.split('-')
        
        # if the user exchange minimum and maximum
        if minimo_y>massimo_y:
            minimo_y,massimo_y=massimo_y,minimo_y
        
        # if the publication year falls in the range we add a score
        for i in range(n):
            if int(minimo_y)<int(years_pub.iloc[i])<int(massimo_y):
                new_score[i][0]+=0.5
                
           
    # 5.1 score based on the autors
    #the user can insert one or more authors ,only their names or surnames or either,
    #if they are between the books returned by the previous exercise we add a score 
    autor_input=input("4. Enter the author (or authors),his name ,his surname of either\
separated by commas,otherwise press enter (e.g. suzanne collins,terry brook)\n").title()
        
    if autor_input!='':
        autor_input=re.findall(r"[\w']+", autor_input)
        
        n_autors=len(autor_input)
        for i in range(n):
            count=0
            for a in autor_input:
                if a in autors.iloc[i]:
                    count+=1
            new_score[i][0]+=count/n_autors
            
    # 5.2 score based on the popularity of each autors
    all_autors=find_authors(df)
    query_autors=find_authors(dataset_query)
    
    
    #create a vocabualry where key:name_autor value:number of book he wrote
    diz_popularity=find_authors_popularity(query_autors,all_autors)
    
    #value that we use to normalize the result
    norm=sum(diz_popularity.values())
    
    for i in range(n):
        aut=autors.iloc[i]
        aut=aut.strip('[]').split(', ')
        s=0
        for a in aut:
            a=a.strip("'")
            s+=diz_popularity[a]
        new_score[i][0]+=(s/norm)
    
    # 6. we ask special words that the plot must not contains
    #if the plot contais any of these special words we decrease its score
    
    prob_words=input('5. Enter the words that the description must not contains separated by commas\
,otherwise press enter(e.g. assassin,woman)\n')
    prob_words=clean_info(prob_words)
    prob_words=prob_words.split(' ')
    n_words=len(prob_words)
    
    for i in range(n):
        p=plots.iloc[i]
        p=clean_info(p)
        count=0
        for w in prob_words:
            if w in p:
                count+=1
        new_score[i][0]-=(count/n_words)
    
    
        
                                            
    #create a heap data structure            
    H=[]
    for i in range(len(new_score)):
        heapq.heappush(H,(new_score[i][0],new_score[i][1]))
    k=5
    top_k = heapq.nlargest (k, H)
    
    score_similarity_top_k=[]
    document_ID_top_k=[]
    for i in top_k:
        score_similarity_top_k.append(i[0])
        document_ID_top_k.append(i[1])
    
    df_top_k=dataset_query.loc[document_ID_top_k][['bookTitle', 'Plot', 'Url']]
    df_top_k['new_similarity_score']=score_similarity_top_k
    
    return df_top_k
    

#Q4 Visualization 

def visualize(dataset):
    dataset = pd.read_csv("Dataset_final_filtered.tsv",sep="\t",parse_dates=['Publishing_Date'],keep_date_col=True)

    #print(dataset[dataset.document_ID==17608]["bookSeries"])
    
    dataset = dataset.dropna(subset=['bookSeries', 'Publishing_Date','NumberofPages'])
    dataset = dataset.sort_values(['bookSeries'])
    
    new_df = copy.deepcopy(dataset)
    new_df = new_df[new_df.bookSeries.str.contains(r'[#]')]
    new_df['bookSeries_No'] = [x.split('#') for x in new_df.bookSeries]
    new_df['bookSeries'] = [x[0] for x in new_df.bookSeries_No]
    new_df['bookSeries_No'] = [x[1] for x in new_df.bookSeries_No]
    series = new_df['bookSeries'].unique()
    
    dictionary = {}
    count = 0
    for i in series:
        data = new_df[new_df.bookSeries == i]
        data = data[~data.bookSeries.str.contains(r'[-#]')]
        data = data[data['bookSeries_No'].str.isdigit()]
        #data = data[data['bookSeries_No'].str.contains[1|2]]
        if len(data) == 10:
            dictionary[i] = data
            count += 1
            if count == 10:
                break
    
    
    #data = dictionary['Alex Rider ']
    F_Data = {}
    for r in dictionary:
        data = dictionary[r]
        data.bookSeries_No = data.bookSeries_No.astype(int)
        data = data.sort_values(['bookSeries_No'])
    
        Pcount = []
        sum = 0
        for i in data['NumberofPages']:
            sum=sum+i
            Pcount.append(sum)
        seriesNO = [i for i in data['bookSeries_No']]
        F_Data[r] = [Pcount,seriesNO]
    
    Rlist = []
    for i in F_Data:
        Rlist.append(F_Data[i][1])
    Rlist = [item for sublist in Rlist for item in sublist]
    Rlist = set(Rlist)
    
    CList = F_Data.keys()
    
    data = []
    for i in CList:
        data.append(F_Data[i][0])
    data = pd.DataFrame(data, index = CList,columns=[1,2,3,4,5,6,7,8,9,10])
    plt.figure(figsize=(18,12))
    sns.lineplot(data=data.T)
    return data,F_Data
