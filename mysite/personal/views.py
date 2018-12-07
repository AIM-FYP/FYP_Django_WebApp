from django.shortcuts import render
import json
from personal.classification_model import *
import pandas as pd
import numpy as np
from mysite.settings import BASE_DIR
import operator
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import scipy

def removeURL(tweet):
    tweet=re.sub('@[A-Za-z0-9_]+|https?://[^ ]+|^(RT )|( RT )', '', tweet)
    tweet=re.sub('www.[^ ]+', '', tweet)
    return tweet

def normalize(values, bounds):
    return [bounds['desired']['lower'] + (x - bounds['actual']['lower']) * (bounds['desired']['upper'] - bounds['desired']['lower']) / (bounds['actual']['upper'] - bounds['actual']['lower']) for x in values]

def changeLabelNames(x):
    if x == 'Positive':
        return 'pos'
    elif x == 'Negative':
        return 'neg'
    else:
        return 'neu'

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results

def top_tfidf_feats(row, features, top_n=25):
    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats)
    df.columns = ['feature', 'tfidf']
    return df

def top_feats_in_doc(Xtr, features, row_id, top_n=25):
    ''' Top tfidf features in specific document (matrix row) '''
    row = np.squeeze(Xtr[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    ''' Return the top n features that on average are most important amongst documents in rows
        indentified by indices in grp_ids. '''
    if grp_ids:
        D = Xtr[grp_ids].toarray()
    else:
        D = Xtr.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)
def removeDuplicates(words_d,fdist):
    keys = list(words_d.keys())
    for w1 in keys:
        for w2 in keys:
            if w1 != w2: 
                if w1.lower() == w2.lower():
                    
                    if(fdist[w1] > fdist[w2]):
                        words_d[w1] = words_d[w1] + words_d[w2]
                        words_d.pop(w2)
                        fdist[w1] = fdist[w1] + fdist[w2]
                        fdist.pop(w2)
                        keys.pop(keys.index(w2))
                        
                    else:
                        words_d[w2] = words_d[w1] + words_d[w2]
                        words_d.pop(w1)
                        fdist[w2] = fdist[w1] + fdist[w2]
                        fdist.pop(w1)
                        keys.pop(keys.index(w1))
                        
                        break
    return words_d,fdist   

def getEntities(tweet,words_d):
    arr = []
    for w in words_d:
        w=w[0]
        if w in tweet:
            arr.append(w)
    return arr

def mywordcloudData(filename,topWords):
    df = pd.read_excel(filename) 
    df.head()
    labels=classificationModel.predict_example(list(np.array(df['text'])))
    df['labels']=labels
    df['labels']=df['labels'].apply(lambda x: changeLabelNames(x))
    df['processed'] = df['text'].apply(lambda x: classificationModel.preProcessingSubfunction2(x))
    
    nltkstopwords=set(stopwords.words('english'))
    capsnltkwords=[]
    for words in nltkstopwords:
        capsnltkwords.append(words.upper())
        capsnltkwords.append(words.capitalize())

    _stopwords=list(nltkstopwords)+capsnltkwords
    
    cv1=CountVectorizer(ngram_range=(1,1),min_df=2,stop_words=_stopwords,max_features=40000,lowercase=False)
    cv=cv1.fit_transform(df['processed'])
    
    vect=TfidfVectorizer(ngram_range=(1,1),min_df=2,stop_words=_stopwords,max_features=40000,lowercase=False)
    vect=vect.fit_transform(df['processed'])
    
    words=top_mean_feats(vect,cv1.get_feature_names(),top_n=vect.shape[1])
    
    words_d=words.set_index('feature')['tfidf'].to_dict()
    
    freq = np.ravel(cv.sum(axis=0))
    
    vocab = [v[0] for v in sorted(cv1.vocabulary_.items(), key=operator.itemgetter(1))]
    fdist = dict(zip(vocab, freq)) 
    
    words_d,fdist=removeDuplicates(words_d,fdist)
    
    sorted_x = sorted(words_d.items(), key=operator.itemgetter(1))
    
    sorted_x=sorted_x[-topWords:]

    
    wordcounts=[]
    for w in sorted_x:
        w=w[0]
        wordcounts.append(fdist[w])
        
    #firstquartile = np.percentile(wordcounts, 10)
    #thirdquartile = np.percentile(wordcounts, 90)
    
    wordcounts = [number/scipy.std(wordcounts) for number in wordcounts]
    
    #wordcounts=normalize(wordcounts,{'actual':{'lower':min(wordcounts),'upper':max(wordcounts)},'desired':{'lower':17,'upper':70}})
    #print(wordcounts)
    ___i=0
    wordsData=[]
    for w in sorted_x:
        w=w[0]
        i_dict = {}
        i_dict['text']=w
        
        #wordcount=fdist[w]
        
        #if wordcount<=firstquartile:
        #    wordcount=17
        #if wordcount >= thirdquartile:
        #    wordcount=70
        
        i_dict['_size']=str(wordcounts[___i])
        wordsData.append(i_dict)
        ___i+=1
        
    return wordsData,sorted_x,df


def landing(request):
    return render(request, 'personal/landing.html')

def index(request,num='5'):
    # ********************************************************* #
    # ********************************************************* #
    # ********************************************************* #
    # ****************                       ****************** #
    # ****************       Raw Tweets      ****************** #
    # ****************                       ****************** #
    # ********************************************************* #
    # ********************************************************* #
    
    #  Change candidate files
    _file_Path=''
    if num=='0':
        _file_Path  = BASE_DIR+"/personal/static/personal/csv/ImranKhan.xlsx"
    elif num=='1':
        _file_Path  = BASE_DIR+"/personal/static/personal/csv/NawazSharif.xlsx"
    elif num=='2':
        _file_Path  = BASE_DIR+"/personal/static/personal/csv/MaryamNawaz.xlsx"
    elif num=='3':
        _file_Path  = BASE_DIR+"/personal/static/personal/csv/Zardari.xlsx"
    elif num=='4':
        _file_Path  = BASE_DIR+"/personal/static/personal/csv/ShehbazSharif.xlsx"
    elif num=='5':
        _file_Path  = BASE_DIR+"/personal/static/personal/csv/SheikhRashid.xlsx"
    elif num=='6':
        _file_Path  = BASE_DIR+"/personal/static/personal/csv/AsadUmar.xlsx"
    elif num=='7':
        _file_Path  = BASE_DIR+"/personal/static/personal/csv/BilawalBhutto.xlsx"

        
    #  Read
    #{% static 'personal/js/c3.min.js' %}
    
    df = pd.read_excel(_file_Path) 
    #  Predict
   
    labels=classificationModel.predict_example(list(np.array(df['text'])))
    df['labels']=labels
    
    df['text']=df['text'].apply(lambda x: removeURL(x))
    
    df['labels']=df['labels'].apply(lambda x: changeLabelNames(x))
    
    #Get time
    df['tweet_time']=df['time'].apply(lambda x: str(x).split()[1][:-3])
    df['line_time']=df['time'].apply(lambda x: str(x).split()[1][:-6])
    
    #Raw Tweets Cell
    #cell1
    classes = classificationModel.targetDomain()
    for i in range(len(classes)):
        classes[i] = changeLabelNames(classes[i])
    probs=np.array(classificationModel.predict_example_proba(list(np.array(df['text']))))
    #cell2
    clabels=[classes[probs[i].argmax()] for i in range(len(probs))]
    #cell3
    probs=[float("{0:.2f}".format((probs[i].max())*100)) for i in range(len(probs))]
    #cell4
    df['labels']=clabels
    df['confidence'] = list(probs)
    df=df.sort_values(['confidence'], ascending=[False]).reset_index()
    #cell5
    tweetdata=[]
    #display only two for now
    _positiveC=0
    _negativeC=0
    _neutralC =0
    for i in range(len(df[['text','labels','confidence','tweet_time']])):
        i_dict = {}
        
        if(_positiveC==2 and _negativeC==2 and _neutralC==2):
            break
        '''
        if df['labels'][i]=='pos' and _positiveC<2:
            i_dict['sentiment']=df['labels'][i]
            i_dict['percentage']=str(df['confidence'][i])
            i_dict['text']=df['text'][i]
            i_dict['time']=df['tweet_time'][i]
            
            _positiveC+=1
            
        if df['labels'][i]=='neg' and _negativeC<2:
            i_dict['sentiment']=df['labels'][i]
            i_dict['percentage']=str(df['confidence'][i])
            i_dict['text']=df['text'][i]
            i_dict['time']=df['tweet_time'][i]
            
            _negativeC+=1
            
        if df['labels'][i]=='neu' and _neutralC<2:
            i_dict['sentiment']=df['labels'][i]
            i_dict['percentage']=str(df['confidence'][i])
            i_dict['text']=df['text'][i]
            i_dict['time']=df['tweet_time'][i]
            
            _neutralC+=1
        '''
        i_dict['sentiment']=df['labels'][i]
        i_dict['percentage']=str(df['confidence'][i])
        i_dict['text']=df['text'][i]
        i_dict['time']=df['tweet_time'][i]
        
        tweetdata.append(i_dict)
        
    raw_tweets_Data=tweetdata
    
        
    # ********************************************************* #
    # ********************************************************* #
    # ********************************************************* #
    # ****************                       ****************** #
    # ****************    SentimentSummary   ****************** #
    # ****************                       ****************** #
    # ********************************************************* #
    # ********************************************************* #

    percentdata = []
    #guageData=[]
    i_dict = {}
    #_max = 0.0
    for cl in df.labels.unique():
        val = float("{0:.2f}".format((sum(df['labels']==cl)/float(len(df['labels']))*100)))
        #if val >= _max:
        #    _max = val
        i_dict[cl] = str(val)
    #guageData.append({"sentiment":str(_max)})
    percentdata.append(i_dict)
    sentiment_summary_donut_Data=percentdata
  
    
    a=df.groupby(['line_time', 'labels']).size().reset_index(name='count') 
    
    d = {}
    for i in a['line_time'].unique():
        d[i] = [{a['labels'][j]: a['count'][j]} for j in a[a['line_time']==i].index]
    
    arr=['pos','neg','neu']
    data=[]
    for time in d.keys():
        i_dict = {}

        p_count = 0
        n_count=0
        nu_count=0
        for dic in d[time]:
            if 'pos' in dic.keys():
                i_dict["pos"]= str(dic['pos'])
                p_count+=1
            if 'neg' in dic.keys():
                i_dict["neg"]=str(dic['neg'])
                n_count+=1
            if 'neu' in dic.keys():
                i_dict["neu"]=str(dic['neu'])
                nu_count+=1

        if p_count == 0:
            i_dict["pos"]="0"
        if n_count == 0:
            i_dict["neg"]="0"
        if nu_count==0:
            i_dict["neu"]="0"

        i_dict['time'] = str(time)

        data.append(i_dict)
    
    sentiment_summary_linechart_Data=data
    
    a,b,c  =  mywordcloudData(_file_Path,55)
    entity_significance_wordcloud_Data=a
    
    bardata = []
    for i in range(len(c[['labels','processed']])):
        i_dict = {}
        i_dict['entities'] = getEntities(c['processed'][i],b) 
        i_dict['sentiment'] = c['labels'][i]
        bardata.append(i_dict)
    entity_significance_bar_Data = bardata
    
    
    
    a,b,c  =  mywordcloudData(_file_Path,21)
    sentiment_summary_word_Data=a
    
    
    raw_tweets_Data              = json.dumps(raw_tweets_Data)
    sentiment_summary_donut_Data = json.dumps(sentiment_summary_donut_Data)
    sentiment_summary_word_Data  = json.dumps(sentiment_summary_word_Data)
    entity_significance_wordcloud_Data= json.dumps(entity_significance_wordcloud_Data)
    entity_significance_bar_Data = json.dumps(entity_significance_bar_Data)
    sentiment_summary_linechart_Data= json.dumps(sentiment_summary_linechart_Data)
    
    return render(request, 'personal/home.html',{"raw_tweets_Data":raw_tweets_Data,"sentiment_summary_donut_Data":sentiment_summary_donut_Data,"sentiment_summary_word_Data":sentiment_summary_word_Data,"entity_significance_wordcloud_Data":entity_significance_wordcloud_Data,"entity_significance_bar_Data":entity_significance_bar_Data,"sentiment_summary_linechart_Data":sentiment_summary_linechart_Data,"sentiment_summary_linechart_Data":sentiment_summary_linechart_Data})

    
def electoral(request):
    return render(request, 'personal/electoral.html')