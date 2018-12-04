import matplotlib
matplotlib.use('PS')
import pandas as pd
import numpy as np
import string
import re
from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from bs4 import BeautifulSoup
import emoji


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix


import seaborn as sns
import matplotlib.pyplot as plt


from nltk.corpus import stopwords
from nltk.corpus import wordnet 

from sklearn.feature_extraction.text import TfidfVectorizer


from collections import Counter

from io import StringIO


from sklearn.linear_model import LogisticRegression


class Classifier:
    
    def split_data(self,X, Y, percentage=0.8):
        """
         Split the training data into training and test set according to given percentage... 

        Parameters:
        --------
        X: training examples
        Y: training labels
        percentage: split data into train and test accorind to given %

        Returns:
        ---------    
        returns four lists as tuple: training data, training labels, test data, test labels 
        """

        testp=1-percentage

        #Split the data into train and test according to given fraction..

        #Creat a list of tuples according to the n-classes where each tuple will 
        # contain the pair of training and test examples for that class...
        #each tuple=(training-examples, training-labels,testing-examples,testing-labels)
        exdata=[]
        #Creat 4 different lists 
        traindata=[]
        trainlabels=[]
        testdata=[]
        testlabels=[]

        classes=np.unique(Y)

        for c in classes:
            # print c
            idx=Y==c
            Yt=Y[idx]
            Xt=X[idx,:]
            nexamples=Xt.shape[0]
            # Generate a random permutation of the indeces
            ridx=np.arange(nexamples) # generate indeces
            np.random.shuffle(ridx)
            ntrainex=round(nexamples*percentage)
            ntestex=nexamples-ntrainex

            ntrainex = int(ntrainex)

            traindata.append(Xt[ridx[:ntrainex],:])
            trainlabels.append(Yt[ridx[:ntrainex]])

            testdata.append(Xt[ridx[ntrainex:],:])
            testlabels.append(Yt[ridx[ntrainex:]])

            #exdata.append((Xt[ridx[:ntrainex],:], Yt[ridx[:ntrainex]], Xt[ridx[ntrainex:],:], Yt[ridx[ntrainex:]]))


        # print traindata,trainlabels
        Xtrain=np.concatenate(traindata)
        Ytrain=np.concatenate(trainlabels)
        Xtest=np.concatenate(testdata)
        Ytest=np.concatenate(testlabels)
        return Xtrain, Ytrain, Xtest, Ytest
    
    def loadData(self, filename):
        self.data = pd.read_csv(filename,sep='\t')
    
    def removeDuplicatesColumns(self):
        self.data   = self.data.drop_duplicates(subset='tweet_id')
        self.t_data = self.data[['text','sentiment','tweet_id']]
    
    def split(self):
        
        x = self.t_data["text"].values
        X = x.reshape((x.shape[0],1))
        Y = np.array(self.t_data['sentiment'])
        Xtrain, Ytrain, Xtest, Ytest = self.split_data(X,Y)
        
        
        train=pd.DataFrame(Xtrain,columns=['text'])
        train['sentiment']=Ytrain
        self.train=train
        
        test=pd.DataFrame(Xtest,columns=['text'])
        test['sentiment']=Ytest
        self.test=test
        
    def mark_neg(self, tweet):
        
        tk=nltk.casual.TweetTokenizer()
        doc=tk.tokenize(tweet)
        pos=nltk.pos_tag(doc)


        flag=False

        for i in range(0, len(pos)-2):
            if (pos[i][0]=='not' or ("n't" in pos[i][0]) or pos[i][0]=="no"):
                #print ('oye hoye0')
                flag = not flag
            if flag==True:
                #if (pos[i][1]=='JJ' or pos[i][1]=='JJR' or pos[i][1]=='JJS') or (pos[i][1]=='VB' or pos[i][1]=='VBD' or pos[i][1]=='VBG' or pos[i][1]=='VBN' or pos[i][1]=='VBP' or pos[i][1]=='VBZ') or (pos[i][1]=='NN' or pos[i][1]=='NNS' or pos[i][1]=='NNP' or pos[i][1]=='NNPS'):
                    #print ('oye hoye1')
                #    doc[i]='_'.join(['not',doc[i]])
                if ((pos[i][1]=='RB' or pos[i][1]=='RBR' or pos[i][1]=='RBS') or pos[i][1]=='DT') and (pos[i+1][1]=='JJ' or pos[i+1][1]=='JJR' or pos[i+1][1]=='JJS'):
                    #print ('oye hoye2')
                    doc[i+1]='_'.join(['not',doc[i+1]])
                    
                if pos[i][1]=='DT' and (pos[i+1][1]=='RB' or pos[i+1][1]=='RBR' or pos[i+1][1]=='RBS') and (pos[i+2][1]=='JJ' or pos[i+2][1]=='JJR' or pos[i+2][1]=='JJS'):
                    #print ('oye hoye3')
                    doc[i+2]='_'.join(['not',doc[i+2]])
                
                flag=False

        return ' '.join(doc)
    
    def custom_mark_neg(self, tweet):
        
        tk=nltk.casual.TweetTokenizer()
        doc=tk.tokenize(tweet)
        pos=nltk.pos_tag(doc)


        flag=False

        for i in range(0, len(pos)):
            if (pos[i][0]=='not' or ("n't" in pos[i][0]) or pos[i][0]=="no"):
                #print ('oye hoye0')
                flag = not flag
            if flag==True:
                
                if i+1<len(pos):
                    if pos[i+1][1]=='JJ' or pos[i+1][1]=='JJR' or pos[i+1][1]=='JJS':
                        #doc[i+1]='_'.join(['not',doc[i+1]])
                        doc[i+1]=getAntonym(doc[i+1])
                        doc[i]=''
                    elif ((pos[i][1]=='RB' or pos[i][1]=='RBR' or pos[i][1]=='RBS') or pos[i][1]=='DT') and (pos[i+1][1]=='JJ' or pos[i+1][1]=='JJR' or pos[i+1][1]=='JJS'):
                        #doc[i+1]='_'.join(['not',doc[i+1]])
                        doc[i+1]=getAntonym(doc[i+1])
                        doc[i]=''
                elif i+2<len(pos):
                    if pos[i+2][1]=='JJ' or pos[i+2][1]=='JJR' or pos[i+2][1]=='JJS':
                        #doc[i+2]='_'.join(['not',doc[i+2]])
                        doc[i+2]=getAntonym(doc[i+2])
                        doc[i]=''
                    elif pos[i][1]=='DT' and (pos[i+1][1]=='RB' or pos[i+1][1]=='RBR' or pos[i+1][1]=='RBS') and (pos[i+2][1]=='JJ' or pos[i+2][1]=='JJR' or pos[i+2][1]=='JJS'):
                        #doc[i+2]='_'.join(['not',doc[i+2]])
                        doc[i+2]=getAntonym(doc[i+2])
                        doc[i]=''
                elif i+3<len(pos):
                    if pos[i+3][1]=='JJ' or pos[i+3][1]=='JJR' or pos[i+3][1]=='JJS':
                        #doc[i+3]='_'.join(['not',doc[i+3]])
                        doc[i+3]=getAntonym(doc[i+3])
                        doc[i]=''
                elif i+4<len(pos):
                    if pos[i+4][1]=='JJ' or pos[i+4][1]=='JJR' or pos[i+4][1]=='JJS':
                        #doc[i+4]='_'.join(['not',doc[i+4]])
                        doc[i+4]=getAntonym(doc[i+4])
                        doc[i]=''
               
                
                flag=False

        return ' '.join(doc)
    
        
        
    def give_emoji_free_text(self, text):
        allchars   = [str for str in text]
        emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
        clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])

        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U0001F1F2-\U0001F1F4"  # Macau flag
            u"\U0001F1E6-\U0001F1FF"  # flags
            u"\U0001F600-\U0001F64F"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            u"\U0001f926-\U0001f937"
            u"\U0001F1F2"
            u"\U0001F1F4"
            u"\U0001F620"
            u"\u200d"
            u"\u2640-\u2642"
            "]+", flags=re.UNICODE)

        clean_text = emoji_pattern.sub(r'', clean_text)

        return clean_text

    def preProcessingSubfunction(self, tweet):
        tweet=tweet.encode().decode("utf-8-sig")
        tweet=tweet.replace(u"\ufffd", "?")
        tweet=tweet.replace(u"\\u002c","")
        tweet=re.sub('\s+', ' ', tweet)
        
        tweet=re.sub(r'[0-9]+(th|nd|st|rd)','',tweet)
        
        tweet=re.sub('@[A-Za-z0-9_]+|https?://[^ ]+|^(RT )|( RT )', ' ', tweet)
        tweet=re.sub('www.[^ ]+', ' ', tweet)
        tweet=BeautifulSoup(tweet, 'lxml').get_text().lower()
        tweet=re.sub(r"""["\?,$!@#\-$\\%\^&*()\[\]{}`“\/~\|._\+\-;<>=:]|'(?!(?<! ')[ts])""", "", tweet)
        #tweet=self.give_emoji_free_text(tweet)
        tweet = apostrapheHandler(tweet)
        #hash removal STARTS
        #tweet=re.sub(r'#[a-z0-9]*','',tweet)
        #hash removal ENDS
        
        #acronyms STARTS
        tweet = re.sub(r'(.)\1+', r'\1\1', tweet)   
        tweet = self.acronymHandler(tweet)
        #acronyms ENDS
        
        tweet=re.sub(r"[0-9]","",tweet)
        
        #tweet=" ".join(mark_negation(tweet.split()))
        #tweet=self.mark_neg(tweet)
        tweet=self.custom_mark_neg(tweet)
        
        return tweet
 
    def acronymHandler(self, tweet):
        tokenizer = nltk.casual.TweetTokenizer()
        tweetArr = tokenizer.tokenize(tweet)
        for i in range(len(tweetArr)):
            #print(tweetArr[i])
            if tweetArr[i] in self.acrDict:
                tweetArr[i] = self.acrDict[tweetArr[i]]
        
    
        tweet = ' '.join(tweetArr)
        return tweet
    
    def makeAcronymDictionary(self,filename):
        acrData = pd.read_csv(filename,sep="\t")
        acrData = acrData.apply(lambda x: x.astype(str).str.lower())
        self.acrDict = dict(zip(acrData.acronym, acrData.definition))

    def preProcessing(self):
        self.makeAcronymDictionary("acronyms.txt")
        self.train['processed_text'] = self.train['text'].apply(lambda x: self.preProcessingSubfunction(x))
        self.test ['processed_text'] = self.test['text'].apply(lambda x: self.preProcessingSubfunction(x))
    
    def align(self, arr=['text','processed_text','sentiment']):
        self.train=self.train[arr]
        self.test =self.test[arr]
    
    def targetDomain(self):
        return list(np.unique(self.train['sentiment']))
    
    def testLabels(self):
        return list(self.test['sentiment'])
    
    def underSampling(self):
        
        cnames=['Positive','Negative','Neutral']
        classes=set([0,1,2])
        PosNegNeu=[(sum(self.t_data['sentiment']=='Positive')),(sum(self.t_data['sentiment']=='Negative')),(sum(self.t_data['sentiment']=='Neutral'))]
        minone=np.argmin(PosNegNeu)


        new_t_data = self.t_data[(self.t_data['sentiment']==cnames[minone])]

        classes=classes-set([minone])
        for _class in classes:
            _class_indices = self.t_data[self.t_data.sentiment == cnames[_class]].index
            random_indices = np.random.choice(_class_indices, sum(self.t_data['sentiment']==cnames[minone]), replace=False)
            _class_sample  = self.t_data.loc[random_indices]

            new_t_data = pd.concat([new_t_data, _class_sample], ignore_index=True)

        self.t_data=new_t_data
        self.t_data=self.t_data.sample(frac=1).reset_index(drop=True)
    
  
    def train_classifier(self, tokenizer=nltk.casual.TweetTokenizer(),vectorizer=CountVectorizer(),mindif=2,mfeatures=None,stopwords=None,classifier=MultinomialNB(),ngramRng=(1,1)):
        # initialize tweet_vector object, and then turn tweet train data into a vector 
        self.tokenizer  = tokenizer
        self.vectorizer = vectorizer        
        self.vectorizer.set_params(min_df=mindif, tokenizer=self.tokenizer.tokenize,max_features=mfeatures, stop_words=stopwords,ngram_range=ngramRng)

        self.train_tweet_counts = self.vectorizer.fit_transform(self.train['processed_text'])
        self.classifier = classifier.fit(self.train_tweet_counts, list(self.train['sentiment']))

    def predict(self):
        
        test_tweet_counts = self.vectorizer.transform(self.test['processed_text'])
        return self.classifier.predict(test_tweet_counts)
      
    
    def predict_example(self, examples):
    
        processedTweets = pd.DataFrame(examples)[0].apply(lambda x: self.preProcessingSubfunction(x))
        test_tweet_counts = self.vectorizer.transform(processedTweets)
   
        return logistic.classifier.predict(test_tweet_counts)
    
    
        
        
def printings(classifier):
    labels = classifier.targetDomain()
    conmat = np.array(confusion_matrix(classifier.testLabels(), y_pred, labels=labels))
    confusion = pd.DataFrame(conmat, index=labels,
                             columns=['predicted_'+x for x in labels])
    print ("\n")
    print ("Accuracy Score: {0:.2f}%".format(accuracy_score(classifier.testLabels(), y_pred)*100))
    print ("-"*80)
    print ("Confusion Matrix\n")
    print (confusion)
    print ("-"*80)
    print ("Classification Report\n")
    print (classification_report(classifier.testLabels(), y_pred,digits=5))

def getCustomStopWords():
    stopWords=set(stopwords.words('english'))

    _stopWords=[]
    for x in stopWords:
        if "'" not in x:
            _stopWords.append(x)
    return _stopWords

def getCustomStopWords3():
    stopWords=set(stopwords.words('english'))
    li=[]

    oldValue=report_to_df(classification_report(naivebayes.testLabels(), y_pred,digits=20))['f1-score']['avg/total']

    for a in stopWords:
        fake=li.copy()
        fake.append(a)

        naivebayes.train_classifier(stopwords=set(fake),mindif=3,ngramRng=(1,5),vectorizer=TfidfVectorizer())
        y_pred = naivebayes.predict()

        newValue=report_to_df(classification_report(naivebayes.testLabels(), y_pred,digits=20))['f1-score']['avg/total']
   
        if oldValue < newValue:
            li.append(a)
            oldValue=newValue
            
    return li


def getCustomStopWords2():
    stopWords=set(stopwords.words('english'))

    _stopWords=[]
    for x in stopWords:
        if "not" in x or "n't" in x or "no" in x:
            _stopWords.append(x)
    return _stopWords

def apostrapheHandler(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"let\'s", "let us", phrase)
    phrase = re.sub(r"shan\'t", "shall not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def report_to_df(report):
    report = re.sub(r" +", " ", report).replace("avg / total", "avg/total").replace("\n ", "\n")
    report_df = pd.read_csv(StringIO("Classes" + report), sep=' ', index_col=0)        
    return(report_df)

def getAntonym(word):

    customDict={'alive':'dead',

    'backward':'forward',

    'beautiful':'ugly',

    'big':'small',

    'blunt':'sharp',

    'boring':'interesting',

    'bright' :'dark' ,

    'broad':'narrow',

    'clean':'dirty',

    'intelligent':'stupid',

    'closed':'open',

    'cool':'warm',

    'cruel':'kind',

    'dangerous':'safe',



    'deep':'shallow',

    'difficult':'easy',

    'dry':'wet',

    'early':'late',

    'fake':'real',

    'fast':'slow',

    'fat':'thin',

    'gentle':'fierce',

    'good':'bad',

    'happy':'sad',

    'hard':'soft',

    'heavy':'light' ,

    'high':'low',

    'hot':'cold',

    'ill':'well',

    'innocent':'guilty',

    'long' :'short' ,

    'loose':'tight',

    'loud' :'soft' ,

    'low':'high',

    'modern':'ancient',

    'noisy':'quiet',

    'normal':'strange',

    'old' :'new' ,

    'outgoing':'shy',

    'poor':'rich',

    'right' :'wrong',


    'rough':'smooth',

    'short' :'tall' ,

    'sour':'sweet',

    'strong':'weak',

    'terrible':'wonderful',

    'far':'near',

    'cheap':'expensive',
    
    'ok':'disapprove'
            
    }
    
    if word in customDict.keys():
        return customDict[word]
    else:
        antonyms = [] 

        
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                if l.antonyms(): 
                    antonyms.append(l.antonyms()[0].name()) 


        counter=Counter(antonyms) 
        if len(counter)>=1:
            return counter.most_common(1)[0][0]
        else:
            return word

