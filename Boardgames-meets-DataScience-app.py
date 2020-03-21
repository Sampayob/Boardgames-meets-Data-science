#!/usr/bin/env python
# coding: utf-8

## Boardgames meets Data Science app

import streamlit as st

import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import random
import ast
import time

("""
# Boardgames meets Data Science

[Author](https://github.com/Sampayob), [App Code](https://raw.githubusercontent.com/Sampayob/Boardgames-meets-Data-science/master/Boardgames-meets-DataScience-app.py)

This data science project comes up with the idea of exploring the **boardgame market** using **Machine Learning** and **Natural Language Processing** to find what makes this games **awesome**. 

- The **data** was web scraped from [BoardGameGeek](https://boardgamegeek.com/browse/boardgame) site (also known as BGG) at March 2020.

- You can find also here a [**deeper analysis**](https://github.com/Sampayob/Boardgames-meets-Data-science/blob/master/Boardgames-meets-DataScience.ipynb).
""")

url = 'https://raw.githubusercontent.com/Sampayob/Boardgames-meets-Data-science/master/BGGTop5000.csv'

@st.cache
def dataset():
    df = pd.read_csv(url,delimiter = ',')
    df['Rating'] = df['Rating'].round(2)
    df['Weight'] = df['Weight'].round(2)
    df = df.drop(['Unnamed: 0','Categories','Designer','Artists','Publishers','Family'], axis=1)
    df = df[['Name','Rating','Weight','Playing time','Number of players','Best number of players','Age','Year','Mechanisms']]  
    return df
df = dataset()

st.subheader('BGG Top 5000 Boardgames')

if st.checkbox('Show dataset'):
    st.write(df)


## Model

# Drop columns and repleace missings


df = df.drop(['Name'], axis=1)

df.replace("--", np.nan, inplace = True)
df.replace("— Best: none", np.nan, inplace = True)

# Prepare categorical and numerical data for ML
        
num_list = df._get_numeric_data().columns
df_num = df[num_list]
df_cat = df.drop(num_list, axis=1)
df_cat_list = df_cat[['Mechanisms']]  
df_cat = df_cat[['Age','Best number of players','Number of players','Playing time']]
df_cat_list['Mechanisms']=df_cat_list['Mechanisms'].apply(ast.literal_eval)
df_cat_list = df_cat_list['Mechanisms'].apply(pd.Series).stack().str.get_dummies().sum(level=0)

for i in df_cat.columns:
    keys = df_cat[i].unique()
    values = range(len(keys))
    dictionary = dict(zip(keys, values))
    df_cat[i] = df_cat[i].replace(dictionary)

weight = []

for x in df_num['Weight']:
    if x >= 1 and x <2:
        weight.append(0)
    elif x >= 3.5 and x <=4.5:     
        weight.append(1)
    elif x >= 2 and x <3.5:
        weight.append(2)
    else:
        weight.append(3)

df_num['Weight'] = weight


rating = []

for x in df_num['Rating']:
    if x >= 5.5 and x <7:
        rating.append(0)
    else:
        rating.append(1)
        
df_num['Rating'] = rating
        
df_X = pd.concat((df_cat,df_num[['Weight','Rating']],df_cat_list),axis=1)

# Feature selection

y = df_X['Rating']
X = df_X.drop(['Rating'], axis=1)

num_feats = 178

# Pearson Correlation: 
# We check the absolute value of the Pearson’s correlation between the target and numerical features in our dataset. 
# We keep the top n features based on this criterion

def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y,num_feats)

#  Chi-Squared
#calculate the chi-square metric between the target and the numerical variable and only select the variable with the maximum chi-squared values.

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
X_norm = MinMaxScaler().fit_transform(X)
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()

# Recursive Feature Elimination
# select features by recursively considering smaller and smaller sets of features

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()

# Lasso: SelectFromModel
# Embedded method. As said before, Embedded methods use algorithms that have built-in feature selection methods
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

embeded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1"), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)

embeded_lr_support = embeded_lr_selector.get_support()


# Tree-based: SelectFromModel
# We calculate feature importance using node impurities in each decision tree. 
#In Random forest, the final feature importance is the average of all decision tree feature importance.
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
embeded_rf_selector.fit(X, y)

embeded_rf_support = embeded_rf_selector.get_support()

# put all selection together
feature_name = X.columns.tolist()
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embeded_lr_support,
                                    'Random Forest':embeded_rf_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)

mechanisms_worst = []
for x,y in zip(feature_selection_df['Feature'].tolist(), feature_selection_df['Total'].tolist()):
    if y == 3:
        mechanisms_worst.append(x)
        

df_X = df_X.drop(mechanisms_worst, axis=1)
y = np.asarray(df_X['Rating'])
df_X = df_X.drop(['Rating'], axis=1)
X = np.asarray(df_X)

## APP - Boardgame rating predictor

st.header('Machine Learning')

st.info("""
Machine Learning is **an application of artificial intelligence** that provides systems the ability to **automatically learn** and **improve from experience** without being explicity programmed.

- It can resolve **clasification and clustering problems** among other ones.

- In this app, the **clasification** is being made by an **hybrid algorithm** which combine **SVM**,**AdaBoost** and **RandomForest** classifiers. 

- The actual classification **accuracy** is **0.7509**, which can be improved.
""")

if st.checkbox('Boardgame rating predictor'): 
    
    st.info("""
    Taking the **boardgames characteristics** into account, we can train a **machine learning algorithm** to **predict its rating**. This rating was **bined in two categories** for a better prediction: **Low and High rating**, allowing us to know if our game design could make the difference! 
    """)
    
    age_text = df['Age'].unique().tolist()
    age_text.pop(-5)
    age_dummies = list(range(0,len(age_text)))

    #number_text = df['Number of players'].unique().tolist()
    number_text=['1','2','3','4','5','1-2','1-3','1-4','2-3','2-4','2-5']
    number_dummies = list(range(0,len(number_text)))

    #best_text = df['Best number of players'].unique().tolist()
    best_text=['1','2','3','4','5','1-2','1-3','1-4','2-3','2-4','2-5','1+','2+','3+','4+']
    best_dummies = list(range(0,len(best_text)))
    b2= ['2','1-2']
    best_dummies_b2 = list(range(0,len(b2)))
    b3= ['2','3','1-2','1-3','2-3','1+','2+']
    best_dummies_b3 = list(range(0,len(b3)))
    b4= ['2','3','4','1-2','1-3','1-4','2-3','2-4','1+','2+','3+']
    best_dummies_b4 = list(range(0,len(b4)))
    b5 =['1','2','3','4','5','1-2','1-3','1-4','2-3','2-4','2-5','1+','2+','3+','4+']
    best_dummies_b5 = list(range(0,len(b5)))

    #time_text = df['Playing time'].unique().tolist()
    time_text=['15','30','45','60','90','120','180','15-30','30-45','30-90','45-60','60-120','240-480','120-180']
    time_dummies = list(range(0,len(time_text)))

    weight_dummies = df_num['Weight'].unique().tolist()
    weight_text = ['3.5-4.5', '2-3.5','1-2','4.5-5']

    mechanims_text = df_X.iloc[:,5:].astype(int).columns

    number = st.selectbox('Number of players', number_text)

    if number == '1':
        best_dummies = 0
    elif number == '2' or number == '1-2':
        best_dummies = random.choice(best_dummies_b2)
    elif number == '3' or number == '2-3' or number == '1-3':
        best_dummies = random.choice(best_dummies_b3)
    elif number == '4' or number == '2-4' or number == '1-4':
        best_dummies = random.choice(best_dummies_b4)
    elif number == '5' or number == '2-5':
        best_dummies = random.choice(best_dummies_b5)

    number_dummies = number_dummies[number_text.index(number)]

    age = st.selectbox('Players age range', age_text)
    age_dummies = age_dummies[age_text.index(age)]

    time = st.selectbox('Playing time range', time_text)
    time_dummies = time_dummies[time_text.index(time)]

    weight = st.selectbox('Difficulty range between 1 and 5', weight_text)
    weight_dummies = weight_dummies[weight_text.index(weight)]

    mechanisms_list = st.multiselect('Boardgame mechanisms', mechanims_text)
    mechanims_dummies = np.zeros((105,), dtype='int').tolist()

    for i in mechanisms_list:
        index = mechanims_text.tolist().index(i)
        mechanims_dummies[index]=1 

        bg = [age_dummies,number_dummies,best_dummies,time_dummies,weight_dummies]
        bg.extend(mechanims_dummies)

    if st.button('Predict rating'):
        # ML 

        clf0 = svm.SVC(kernel='linear', C = 1.0, gamma = 'scale', degree = 1) 
        clf1 = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 1, n_estimators = 500)
        clf1.fit(X, y)
        clf2 = RandomForestClassifier(criterion='gini',max_depth=5,max_features=2,min_impurity_decrease= 0.02, max_leaf_nodes=5,min_samples_split = 3,n_estimators=100)

        @st.cache
        def algorithms(classifiers):
            clf = VotingClassifier(estimators=classifiers, voting='hard', weights=[1.2130770082041478, 1.3892687725509578, 1.3054265842709998], n_jobs=-1)
            clf1 = clf.fit(X, y)
            return clf1
        algorithms([('svm', clf0),('ada', clf1), ('rf', clf2)])
        
        X_test = np.asarray(bg) 
        X_test = np.reshape(X_test,(-1,110))
        bg_class = clf1.predict(X_test)

        if bg_class == 1:
            st.success('Your game has a high rating (7-9)')
        else:
            st.error('Your game has a low rating (5.5-6.9)')

## APP - Boardgame generator

if st.checkbox('Boardgame generator'):
    
    import time
    
    st.info("""
    If we only want to know **fast** what kind of games could be interesting to develop, **aleatory designs** can be feeded to the algorithm until it **predict one (or more)** with a hypotetically future **high rating**.
    """)

    import warnings
    warnings.filterwarnings("ignore")

    #Get values of atributes : dummies and corresponding text/original value

    age_text = df['Age'].unique().tolist()
    age_text.pop(-5)
    age_dummies = list(range(0,len(age_text)))

    #number_text = df['Number of players'].unique().tolist()
    number_text=['1','2','3','1-2','1-3','1-4','2-3','2-4','2-5']
    number_dummies = list(range(0,len(number_text)))

    #best_text = df['Best number of players'].unique().tolist()
    best_text=['1','2','3','4','5','1-2','1-3','1-4','2-3','2-4','2-5','1+','2+','3+','4+']
    best_dummies = list(range(0,len(best_text)))

    #time_text = df['Playing time'].unique().tolist()
    time_text=['15','30','45','60','90','120','180','15-30','30-45','30-90','45-60','60-120','240-480','120-180']
    time_dummies = list(range(0,len(time_text)))

    weight_dummies = df_num['Weight'].unique().tolist()
    weight_text = ['3.5-4.5', '2-3.5','1-2','4.5-5']

    mechanims_text = df_X.iloc[:,5:].astype(int).columns

    number = st.sidebar.slider('Number of boardgames to generate', 0,20)
    
    bar = st.progress(0)
    status_text = st.empty()
    if st.button('Generate boardgames'):
        
        # ML 

        clf0 = svm.SVC(kernel='linear', C = 1.0, gamma = 'scale', degree = 1) 
        clf1 = AdaBoostClassifier(algorithm = 'SAMME.R', learning_rate = 1, n_estimators = 500)
        clf1.fit(X, y)
        clf2 = RandomForestClassifier(criterion='gini',max_depth=5,max_features=2,min_impurity_decrease= 0.02, max_leaf_nodes=5,min_samples_split = 3,n_estimators=100)

        @st.cache
        def algorithms(classifiers):
            clf = VotingClassifier(estimators=classifiers, voting='hard', weights=[1.2130770082041478, 1.3892687725509578, 1.3054265842709998], n_jobs=-1)
            clf1 = clf.fit(X, y)
            return clf1
        algorithms([('svm', clf0),('ada', clf1), ('rf', clf2)])
        
        status_text.text('Machines studying hard...')
        
        bar_number = 100/number
        it = 1

        bgs=[]
        rating_predic = []    

        n = number
        while n > 0:
            mechanims_dummies = np.random.choice([0, 1], size=105,p=[.98, .02])
            if 1 in mechanims_dummies:
                bg = [random.choice(age_dummies),random.choice(number_dummies),random.choice(best_dummies),random.choice(time_dummies),random.choice(weight_dummies)]
                bg = np.append(bg,mechanims_dummies)
                bg_list = bg.tolist()

                age_bg = age_text[age_dummies.index(bg_list[0])]
                number_bg = number_text[number_dummies.index(bg_list[1])]
                best_bg = best_text[best_dummies.index(bg_list[2])]
                time_bg = time_text[time_dummies.index(bg_list[3])]
                weight_bg = weight_text[weight_dummies.index(bg_list[4])]
                mechanims_bg = []
                for x,y in zip(mechanims_dummies,mechanims_text):
                    if x == 1:
                        mechanims_bg.append(y)


                if number_bg =='1':
                    bg_text = [age_bg,number_bg,'-',time_bg,weight_bg,mechanims_bg]
                else:
                    bg_text = [age_bg,number_bg,best_bg,time_bg,weight_bg,mechanims_bg]

                if bg_text[1] == '1' or (bg_text[1] == '2' and (bg_text[2] == '1' or bg_text[2] =='2')) or (bg_text[1] == '3' and (bg_text[2] == '1' or bg_text[2] =='2' or bg_text[2] == '3' or bg_text[2]== '1+'or bg_text[2] == '2+' or bg_text[2] == '1-2'or bg_text[2]== '1-3'or bg_text[2] == '2-3' )) or (bg_text[1] =='4' and (bg_text[2] == '1' or bg_text[2]=='2' or bg_text[2] == '3' or bg_text[2]== '4' or bg_text[2]== '1+'or bg_text[2] == '2+'or bg_text[2] == '3+'or bg_text[2] == '1-2'or bg_text[2] == '1-3'or bg_text[2] == '2-3' or bg_text[2] =='1-4' or bg_text[2] =='2-4')) or (bg_text[1] =='5' and (bg_text[2] == '1' or bg_text[2]=='2' or bg_text[2] == '3' or bg_text[2] == '4' or bg_text[2]=='5' or bg_text[2] == '1+'or bg_text[2] == '2+'or bg_text[2] == '3+'or bg_text[2]=='4+' or bg_text[2]== '1-2'or bg_text[2] == '1-3'or bg_text[2] == '2-3' or bg_text[2] =='1-4' or bg_text[2] =='2-4' or bg_text[2] =='2-5')):
                    if bg not in bgs:
                        X_test = np.asarray(bg)
                        X_test = np.reshape(X_test,(-1,110))
                        bg_class = clf1.predict(X_test)
                        if bg_class[0]==1:
                            rating_predic.append(bg_class)
                            bgs.append(bg_text)
                            bar.progress((bar_number*it)/100)
                            time.sleep(0.3)
                            it+=1
                            n-=1
                            
        status_text.text('Done!')
        st.table(pd.DataFrame(bgs, columns=['Age','Number of players','Best number of players','Playing time','Weight','Mechanisms']))



## NLP

st.header('Natural Language Processing')

st.info("""
NLP is a subfield of **artificial intelligence** and is the process of analyzing, understanding and deriving **meaning from human languages for computers**.

- It helps you **extract insights** from unstructured text. 

- The libraries used are **NLTK** and **spaCy**

- For this part only the 400 top boardgames descriptions are considered.
""")

if st.checkbox('Show options'):
    

    
    import nltk
    import spacy
    import pandas as pd

    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import PorterStemmer 
    from nltk.probability import FreqDist

    import matplotlib.pyplot as plt


    field_names = {'Name','Rating','Year','Description'} 
    url = 'https://raw.githubusercontent.com/Sampayob/Boardgames-meets-Data-science/master/BGGTop400-Description.csv'
    df2 = pd.read_csv(url, delimiter = ',',names=field_names, encoding='utf-8-sig')
    df2.columns = ['Year','Name','Rating','Description']

    ## Transforming boardgames description in tokens

    #**Getting each description as a string**

    descriptions = []
    for i in df2['Description']:
        descriptions.append(i)

    text = ''.join(descriptions)

   # **Making tokens from text**

    words=' '.join(x.lower() for x in text.split())
    words = word_tokenize(words)

   # **Filter tokens through stop words and punctuation marks**

    # Stopwords
    stop_words = nltk.corpus.stopwords.words('english')
    stopw = ['player','players','card','take','number' , 'nin','nthe' 'games', 'takes' , 'play','may','first','end', 'get','building' , 'victory','tiles','also','time','game','one','two','three','four','five','action','build','new','will','set','tile','color','move','rule','six','nThe','board game','board','use','even','goal','either','good','different','every','must','well','become','different','turn','order','round','nIn','wor','x80','x99','order','system','place','need','many','start','based','great','player','x93','Player','x93','start','great','game','world','unique','actions','face','choose','gain','come','side','small','location','point','cities','make','win','players take','used','placed','find','various','acquire','way','hand','special','want','allow','around','include','another','possible','develop','now','influence','back','best','available','increase','represent','rules','along','help','give','part','feature','original','enough']

    for w in stopw:
        stop_words.append(w)

    # Filter tokens
    tokens =[]
    for w in words:
        if w not in stop_words:
            if len(w) > 2:
                tokens.append(w)

    ## Text analysis
    
    ### Most common words
    
    st.subheader("""
    **Top 400 Boardgames**
    """)
    
    st.subheader("""
    Most frequent words 
    """)

    import collections
    import seaborn as sns

    counter=collections.Counter(tokens)
    most=counter.most_common()

    x, y= [], []
    for word,count in most[:20]:
        if (word not in stop_words):
            x.append(word)
            y.append(count)
    plt.figure(figsize=(10,10))
    sns.barplot(x=y,y=x,palette="GnBu_d")
    
    st.pyplot()
    
    st.subheader("""
    Bigrams
    """)
    
    ### Bigrams: pairs of words which convey a significant amount of information 
    st.write('Bigrams are pairs of words which convey a significant amount of information')
    
    bigram_number = st.selectbox('How many bigrams do you want to see?', list(range(0,100)) )
    if st.checkbox('Search for bigrams'):
    
        from nltk.collocations import *
        bigram_measures = nltk.collocations.BigramAssocMeasures()

        bigram_measures = nltk.collocations.BigramAssocMeasures()
        tokens_finder = BigramCollocationFinder.from_words(tokens)
        tokens_scored = tokens_finder.score_ngrams(bigram_measures.raw_freq)
        st.write(list(tokens_scored[0:bigram_number]))
        
        
    st.subheader("""
    Word frequency
    """)
       
    ###  Count words
    
    wf = st.text_input('Count how many times a word is present in all boardgame descriptions', 'word')
    if st.button('Count'):
        st.write(str(wf)+' appears '+str(tokens.count(wf))+' times')
    
    
     ### Most common words
        
    st.subheader("""
    **One boardgame analysis**
    """)
    
    if st.checkbox("Let's analyse"):
    
        names = []
        for d in df2['Name']:
            names.append(d.replace(" ", "").replace(":", ""))

        desc1 = st.selectbox('Select one', names)

        desc_one = descriptions[names.index(str(desc1))]

        text2 = ''.join(desc_one)

       # **Making tokens from text**

        words2=' '.join(x.lower() for x in text2.split())
        words2 = word_tokenize(words2)  

       # **Filter tokens through stop words and punctuation marks**


        # Filter tokens
        tokens2 =[]
        for w in words2:
            if w not in stop_words:
                if len(w) > 2:
                    tokens2.append(w)
        
        st.subheader("""
        Most frequent words
        """)

        if st.button('Search for frequent words'):

            counter2=collections.Counter(tokens2)
            most2=counter2.most_common()

            x, y= [], []
            for word,count in most2[:10]:
                if (word not in stop_words):
                    x.append(word)
                    y.append(count)
            plt.figure(figsize=(10,10))
            sns.barplot(x=y,y=x,palette="Blues")

            st.pyplot()

        st.subheader("""
        Bigrams
        """)

        ### Bigrams: pairs of words which convey a significant amount of information 

        bigram_number = st.selectbox('How many bigrams do you want to see? ', list(range(0,100)) )
        if st.checkbox('Search for bigrams '):

            from nltk.collocations import *
            bigram_measures = nltk.collocations.BigramAssocMeasures()

            bigram_measures = nltk.collocations.BigramAssocMeasures()
            tokens_finder = BigramCollocationFinder.from_words(tokens2)
            tokens_scored = tokens_finder.score_ngrams(bigram_measures.raw_freq)
            st.write(list(tokens_scored[0:bigram_number]))


        st.subheader("""
        Word frequency
        """)

        ###  Count words

        wf = st.text_input('Count how many times a word is present in all boardgame descriptions ', 'word')
        if st.button('Count '):
            st.write(str(wf)+' appears '+str(tokens2.count(wf))+' times')
