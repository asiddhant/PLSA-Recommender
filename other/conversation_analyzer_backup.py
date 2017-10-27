import nltk
import pandas as pd
import Levenshtein
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
import requests
import re
import csv
import os
#import nltk.data
from nltk.grammar import DependencyGrammar
from nltk.parse import (DependencyGraph,ProjectiveDependencyParser,NonprojectiveDependencyParser,)
from nltk.parse.stanford import StanfordParser
from nltk.parse.stanford import StanfordDependencyParser
import sys
import random
import copy
import time

__name__ = "__conversation_analyzer__"

# Utiltiy Functions
def separator(x):
    if type(x)==str:
        return x.split()
    else:
        return []

def integralconvertor(x):
    return [int(i) for i in x]


# Score Matcher
def scorematcher(scores):
    thres=max(scores)*0.9
    return [index for index in range(len(scores)) if scores[index]>=thres]
    
# User Profiler
def userprofiler(userids,irec,movies,users):
    users=users[['u_id','sex','age','pg']]
    irec=irec[0]
    print(userids)
    print(irec)
    print(len(movies))
    genrelist=["adventure","animation","children","comedy","fantasy","romance","drama","action","crime",
           "thriller","horror","mystery","sci.fi","imax","documentary","war","musical","western","film.noir"]
    users=users[users.u_id.isin(userids)]
    users.pg=list(map(separator,users.pg))
    users.index=list(range(len(userids)))
    imovie=movies[movies.movieid == irec]
    print(imovie)
    print(imovie.index[0])
    print(imovie.genre[imovie.index[0]])
    igenre=imovie.genre[imovie.index[0]]
    igenre=[i for i in range(19) if igenre[i]=="1"]
    if len(igenre)>3:
        igenre=igenre[0:3]
    if len(igenre)==1:
        igenre.insert(0,igenre[0])
        igenre.insert(0,igenre[0])
    if len(igenre)==2:
        igenre.insert(0,igenre[0])
    igenre=[genrelist[i] for i in igenre]
    istars=imovie.stars[imovie.index[0]]
    if len(istars)>3:
        istars=istars[0:3]
    if len(istars)==1:
        istars.insert(0,istars[0])
        istars.insert(0,istars[0])
    if len(istars)==2:
        istars.insert(0,istars[0])
    if len(istars)==0:
        istars=['0','0','0']
    current={'cg': [igenre],
             'cm': [[irec,irec,irec]],
             'ca': [istars]}
    current=pd.DataFrame(current,index=list(range(len(userids))))
    users=pd.concat([users,current],axis=1)
    puserpg=[]
    for i in range(len(userids)):
        puserpg+=users.iloc[i,3]
    randIndex = random.sample(list(range(len(puserpg))), 3)
    puserpg = [puserpg[i] for i in randIndex]
    puser={'u_id':0,'sex':users.sex.mean(),'age':users.age.mean(),
           'pg': [puserpg], 'ca' :[users.iloc[0,4]], 'cg': [users.iloc[0,5]],
           'cm': [users.iloc[0,6]]}
    puser=pd.DataFrame(puser)
    users=users.append(puser)
    users.index=users.u_id
    return users

#Sentiment Analyzer
def sentimentanalyzer(sentence,keysent):
    url='http://text-processing.com/api/sentiment/'
    payload = {'text': sentence}
    out=requests.post(url, data=payload).text
    neg_score=float(out[out.find("neg")+6:out.find("neg")+10])
    pos_score=float(out[out.find("pos")+6:out.find("pos")+10])
    #neutral_score=float(out[out.find("neutral")+10:out.find("neutral")+14])
    if pos_score>0.7:
        sentiment_score=5
    elif pos_score>0.5 and neg_score<0.3:
        sentiment_score=4
    elif pos_score<0.3 and neg_score>0.7:
        sentiment_score=1
    elif pos_score<0.3 and neg_score>0.5:
        sentiment_score=2
    else:
#        if keysent.shape[0]>0:
#            sentiment_score=round((3+(list(keysent.sent_score)[keysent.shape[0]-1]))/2)
#        else:
#            sentiment_score=3
        sentiment_score=3
    return(sentiment_score)

def relevantwords(wordset):
    relwords=[]
    for i in range(len(wordset)):
        if wordset[i][1] in ['NNP','NNPS','NN','NNS']:
            relwords=relwords+[wordset[i][0]]
    return relwords
def commoncount(arr1, arr2):
    if len(arr1)==0 or len(arr2)==0:
        return 0
    count=0
    for word in arr1:
        if word in arr2:
            count=count+1
    return count

def analyze_this(sentence,ref1):
    sentence=sentence.lower()
    pos_score,neg_score,neutral=sentimentanalyzer2(sentence)
    keys=ref1
    table={'key':keys,
            'pos_score':[pos_score]*len(keys),
            'neg_score':[neg_score]*len(keys),
            'neutral':[neutral]*len(keys)}
    table=pd.DataFrame(table)
    return table

def sentimentanalyzer2(sentence):
    url='http://text-processing.com/api/sentiment/'
    payload = {'text': sentence}
    out=requests.post(url, data=payload).text
    neg_score=float(out[out.find("neg")+6:out.find("neg")+10])
    pos_score=float(out[out.find("pos")+6:out.find("pos")+10])
    neutral_score=float(out[out.find("neutral")+10:out.find("neutral")+14])
    return(pos_score,neg_score,neutral_score)

def genreextractor(sentence,genres):
    genrelist=[]
    for i in range(genres.shape[0]):
        if genres['sub.genre'][i] in sentence:
            genrelist=genrelist+[genres['genre'][i]]
    return genrelist

def submovieranker(submovie,userprofiles,userid):
    genrelist=["adventure","animation","children","comedy","fantasy","romance","drama","action","crime",
           "thriller","horror","mystery","sci.fi","imax","documentary","war","musical","western","film.noir"]
    score=0
    t1=2
    t2=1.5
    t3=3
    t4=2.5
    t5=1
    submoviegenre=[genrelist[i] for i in range(len(genrelist)) if submovie['genre'][i]]
    for g in userprofiles.cg[int(userid)]:
        if g in submoviegenre:
            score+=t1
        t1-=0.5
    for g in userprofiles.cg[0]:
        if g in submoviegenre:
            score+=t2
        t2-=0.5
    for s in userprofiles.ca[int(userid)]:
        if s in submovie['stars']:
            score+=t3
        t3-=0.5
    for s in userprofiles.ca[0]:
        if s in submovie['stars']:
            score+=t4
        t4-=0.5
    for g in userprofiles.pg[int(userid)]:
        if g in submoviegenre:
            score+=t5
        t5-=0.5
    expyear=2015-userprofiles.age[0]+18
    if expyear-submovie['year']<10:
        score+=1
    return score

def moviesextractor(postagged,userprofiles,userid,movies):
    alpha=0.25
    beta=0.25
    if len(postagged)==1 and len(postagged[0])<=4:
        return []
    #f.write(str(postagged))
    #f.write("\n")
    movienames=[0]*len(movies['title'])
    for i in range(len(postagged)):
        for j in range(len(movies['title'])):
            li=len(postagged[i])
            lj=len(movies['title'][j])
            if li>5:
                if postagged[i] in movies['title'][j]:
                    movienames[j]=movienames[j]+ 2
                else:
                    dist=Levenshtein.distance(postagged[i],movies['title'][j]) - lj + li
                    if dist<2 and dist>-1:
                        movienames[j]=movienames[j]+ 1
            else:
                if postagged[i] in movies['title'][j]:
                    movienames[j]=movienames[j]+ 1
    if max(movienames):
        maxmatch=np.argwhere(movienames == np.amax(movienames)).flatten().tolist()
        if len(maxmatch)>1 and len(maxmatch)<15:
            submovies=movies.ix[maxmatch]
            #f.write(str(submovies.title))
            #f.write("\n")
            scores=[max(movienames)]*submovies.shape[0]
            j=0
            for i in maxmatch:
                matched=" ".join([word for word in postagged if word in submovies.title[i]])
                scores[j]+=(alpha*submovieranker(submovies.ix[i],userprofiles,userid)-beta*(abs(len(submovies.title[i])-len(matched))))
                j+=1
            #f.write(str(scores))
            #f.write("\n")    
            agreed=scorematcher(scores)
            eligible=[maxmatch[i] for i in agreed]
            #f.write(str(list(submovies.ix[eligible]['title'])))
            #f.write("\n")
            return list(submovies.ix[eligible]['movieid'])
        elif len(maxmatch)==1:
            #f.write(str(list(movies.ix[maxmatch]['title'])))
            #f.write("\n")
            return list(movies.ix[maxmatch]['movieid'])
        else:
            return []
    else:
        return []

def substarsranker(substar,userprofiles,userid):
    genrelist=["adventure","animation","children","comedy","fantasy","romance","drama","action","crime",
           "thriller","horror","mystery","sci.fi","imax","documentary","war","musical","western","film.noir"]
    score=0
    t1=6
    t2=5
    t3=3
    t4=2.5
    for g in userprofiles.cg[userid]:
        score+=((t1*substar['genre'][genrelist.index(g)])/(sum(substar['genre'])))
        t1-=1
    for g in userprofiles.cg[0]:
        score+=((t2*substar['genre'][genrelist.index(g)])/(sum(substar['genre'])))
        t2-=1
    for s in userprofiles.cm[userid]:
        if s in substar['movies']:
            score+=t3
        t3-=0.5
    for s in userprofiles.cm[0]:
        if s in substar['movies']:
            score+=t4
        t4-=0.5
    if substar['pop']>100000:
        score+=2
    elif substar['pop']>10000:
        score+=1
    expyear=2015-userprofiles.age[0]+18
    if expyear-substar['year']<10:
        score+=1
    return score

def starsextractor(postagged,userprofiles,userid,starcast):
    alpha=0.25
    beta=0.25
    if len(postagged)==1 and len(postagged[0])<=4:
        return []
    #f.write(str(postagged))
    #f.write("\n")
    stars=[0]*len(starcast['mstars'])
    for i in range(len(postagged)):
        for j in range(len(starcast['mstars'])):
            li=len(postagged[i])
            lj=len(starcast['mstars'][j])
            if li>5:
                if postagged[i] in starcast['mstars'][j]:
                    stars[j]=stars[j]+ 2
                else:
                    dist=Levenshtein.distance(postagged[i],starcast['mstars'][j]) - lj + li
                    if dist<2 and dist>-1:
                        stars[j]=stars[j]+ 1
            else:
                if postagged[i] in starcast['mstars'][j]:
                    stars[j]=stars[j]+ 1
    if max(stars):
        maxmatch=np.argwhere(stars == np.amax(stars)).flatten().tolist()
        if len(maxmatch)>1 and len(maxmatch)<20:
            substars=starcast.ix[maxmatch]
            #f.write(str(substars.mstars))
            #f.write("\n")
            scores=[max(stars)]*substars.shape[0]
            j=0
            for i in maxmatch:
                matched=" ".join([word for word in postagged if word in substars.mstars[i]])
                scores[j]+=(alpha*substarsranker(substars.ix[i],userprofiles,userid)-beta*(abs(len(substars.mstars[i])-len(matched))))
                j+=1
            #f.write(str(scores))
            #f.write("\n")    
            agreed=scorematcher(scores)    
            eligible=[maxmatch[i] for i in agreed]
            #f.write(str(list(substars.ix[eligible]['star_id'])))
            #f.write("\n")
            return list(substars.ix[eligible]['star_id'])
        elif len(maxmatch)==1:
            #f.write(str(list(starcast.ix[maxmatch]['star_id'])))
            #f.write("\n")
            return list(starcast.ix[maxmatch]['star_id'])
        else:
            return []
    else:
        return []
    
# Different Sentiment Keyword Matchers
def pastchecker(sentence,postagged):
    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    sentimentverbs=["like","love","hate"]
    negativewords=["not","never"]
    for i in range(len(postagged)):
        if postagged[i][1] in ["VBN","VBD"]:
            obj=porter_stemmer.stem_word(postagged[i][0])
            if obj in sentimentverbs:
                return 1
            else:
                for word in negativewords:
                    if word in sentence:
                        return 0
                    else:
                        return 1            
    return 0

def comparisonchecker(postagged):
    for i in range(len(postagged)):
        if postagged[i][1] == "JJR":
            return 1
    return 0
    
def conjunctivechecker(postagged):
    for i in range(len(postagged)-2):
        if postagged[i+1][1] == "CC":
            return 1
    return 0

def detpresence(postagged):
    for i in range(len(postagged)-2):
        if postagged[i+1][1] in ["DT","PRP"]:
            return 1
    return 0
    
def comparisonanalyzer(keysent,sentence,userprofiles,userid,tagged,sid,movies,starcast,genres):
    poscompwords=["more","better"]
    negcompwords=["less","worse"]
    if tagged[len(tagged)-1][1]=="JJR":
        if tagged[len(tagged)-1][1] in poscompwords:
            keysent,userprofiles=normalanalyzer(keysent,sentence,userprofiles,userid,tagged,"P",sid,movies,starcast,genres)
            return keysent,userprofiles
        elif tagged[len(tagged)-1][1] in negcompwords:
            keysent,userprofiles=normalanalyzer(keysent,sentence,userprofiles,userid,tagged,"N",sid,movies,starcast,genres)
            return keysent,userprofiles
    
    parts=[]
    temp=[]
    comp=[]
    for i in range(len(tagged)):
        if tagged[i][1]=="JJR":
            temp.append((tagged[i][0],tagged[i][1]))
            parts+=[temp]
            temp=[]
            comp+=[tagged[i][0]]
        else:
            temp.append((tagged[i][0],tagged[i][1]))
    parts+=[temp]
    
    if len(parts)>2 or len(parts)==1:
        return keysent,userprofiles
        
    p1=[a[0] for a in parts[0]]
    p1=" ".join(p1)

    p2=[a[0] for a in parts[1]]
    p2=" ".join(p2)
    
    if detpresence(parts[1]):
        if comp[0] in poscompwords:
            keysent,userprofiles=normalanalyzer(keysent,p2,userprofiles,userid,parts[1],"N",sid,movies,starcast,genres)
            keysent,userprofiles=normalanalyzer(keysent,p1,userprofiles,userid,parts[0],"P",sid,movies,starcast,genres)
        if comp[0] in negcompwords:
            keysent,userprofiles=normalanalyzer(keysent,p2,userprofiles,userid,parts[1],"P",sid,movies,starcast,genres)
            keysent,userprofiles=normalanalyzer(keysent,p1,userprofiles,userid,parts[0],"N",sid,movies,starcast,genres)
    else:
        if comp[0] in poscompwords:
            keysent,userprofiles=normalanalyzer(keysent,p1,userprofiles,userid,parts[0],"P",sid,movies,starcast,genres)
            keysent,userprofiles=normalanalyzer(keysent,p2,userprofiles,userid,parts[1],"N",sid,movies,starcast,genres)
        if comp[0] in negcompwords:
            keysent,userprofiles=normalanalyzer(keysent,p1,userprofiles,userid,parts[0],"N",sid,movies,starcast,genres)
            keysent,userprofiles=normalanalyzer(keysent,p2,userprofiles,userid,parts[1],"P",sid,movies,starcast,genres)
            
    return keysent,userprofiles
    
def verbpresence(tagged):
    for i in range(len(tagged)):
        if tagged[i][1] in ["VB","VBD","VBP","VBZ","VBN","VBG"]:
            return 1
    return 0
   
def conjunctiveanalyzer(keysent,sentence,userprofiles,userid,tagged,sid,movies,starcast,common,genres):        
    parts=[]
    temp=[]
    conj=[]
    poscompwords=["more","better"]
    negcompwords=["less","worse"]
    for i in range(len(tagged)):
        if tagged[i][1]=="CC":
            parts+=[temp]
            temp=[]
            conj+=[tagged[i][0]]
        else:
            temp.append((tagged[i][0],tagged[i][1]))
    parts+=[temp]
    
    for sen in parts:
        csentence=[a[0] for a in sen]
        csentence=" ".join(csentence)
        if verbpresence(sen):
            if comparisonchecker(sen):
                for word in poscompwords:
                    if word in csentence:
                        keysent,userprofiles=normalanalyzer(keysent,csentence,userprofiles,userid,sen,"P",sid,movies,starcast,common,genres)
                        break
                for word in negcompwords:
                    if word in csentence:
                        keysent,userprofiles=normalanalyzer(keysent,csentence,userprofiles,userid,sen,"N",sid,movies,starcast,common,genres)
                        break
            else:
                keysent,userprofiles=normalanalyzer(keysent,csentence,userprofiles,userid,sen,"0",sid,movies,starcast,common,genres)
        else:
            if parts.index(sen)==0:
                keysent,userprofiles=normalanalyzer(keysent,csentence,userprofiles,userid,sen,"0",sid,movies,starcast,common,genres)
            else:
                if conj[parts.index(sen)-1].lower()=="but":
                    keysent,userprofiles=normalanalyzer(keysent,csentence,userprofiles,userid,sen,"R",sid,movies,starcast,common,genres)
                else:
                    keysent,userprofiles=normalanalyzer(keysent,csentence,userprofiles,userid,sen,"0",sid,movies,starcast,common,genres)
    
    return keysent,userprofiles

def tagfinder(keys,exmovies,exgenre,exstars):
    #tag=keys
    tag = copy.deepcopy(keys)
    for i in range(len(keys)):
        if keys[i] in exmovies:
            tag[i]="m"
        elif keys[i] in exgenre:
            tag[i]="g"
        elif keys[i] in exstars:
            tag[i]="s"
    return tag
    
def normalanalyzer(keysent,sentence,userprofiles,userid,tagged,sentmod,sid,movies,starcast,common,genres):



    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    stop_words=get_stop_words('en')
    sentencelower=sentence.lower()
    exgenre=genreextractor(sentencelower,genres)
    relwords=relevantwords(tagged)
    relwords = [item.lower() for item in relwords if item.lower() not in exgenre+stop_words]
#    for i in range(len(relwords)):
#        relwords[i]=porter_stemmer.stem_word(relwords[i])
    relwords = [item.lower() for item in relwords if item.lower() not in common]
    exmovies=[]
    exstars=[]
    moviegenrereference=["this","that","it"]
    actorreference=["his","her","hers","him","he","she"]
    commonrefrence=["these","those","such"]
    if relwords:
        exmovies = moviesextractor(relwords,userprofiles,userid,movies)
        if exmovies:
            #relwords = [item for item in relwords if item not in list(movies.title[movies.movieid==exmovies[0]])[0]]
            exstars = starsextractor(relwords,userprofiles,userid,starcast)
        else:
            exstars = starsextractor(relwords,userprofiles,userid,starcast)
            
    detwords=[]    
    
    if not exgenre and not exmovies and not exstars:
        for i in range(len(tagged)):
            if tagged[i][1] in ['PRP','DT']:
                detwords=detwords+[tagged[i][0]]
    
    zprev=0
    for word in detwords:
        word=word.lower()
        if word in moviegenrereference:
            zprev=1
            i=1
            while i<=min(10,keysent.shape[0]):
                if keysent.iloc[keysent.shape[0]-i,2]=="m":
                    exmovies+=[keysent.iloc[keysent.shape[0]-i,0]]
                    break
                elif keysent.iloc[keysent.shape[0]-i,2]=="g":
                    exgenre+=[keysent.iloc[keysent.shape[0]-i,0]]
                    break
                else:
                    i+=1
        elif word in actorreference:
            zprev=1
            i=1
            while i<=min(10,keysent.shape[0]):
                if keysent.iloc[keysent.shape[0]-i,2]=="s":
                    exstars+=[keysent.iloc[keysent.shape[0]-i,0]]
                    break
                else:
                    i+=1
        elif word in commonrefrence:
            zprev=1
            i=1
            if keysent.iloc[keysent.shape[0]-1,2]=="s":
                exstars+=[keysent.iloc[keysent.shape[0]-i,0]]
            elif keysent.iloc[keysent.shape[0]-1,2]=="m":
                exmovies+=[keysent.iloc[keysent.shape[0]-i,0]]
            else:
                exgenre+=[keysent.iloc[keysent.shape[0]-i,0]]
    
    #f.write(str(detwords))
    #f.write("\n")
    #f.write(str(exmovies))
    #f.write("\n")
    #f.write(str(exgenre))
    #f.write("\n")
    #f.write(str(exstars))
    #f.write("\n")

    sentimentscore=sentimentanalyzer(sentence,keysent)
    ##Add the sentmod thing here 
    if sentmod=="R":
        sentimentscore=5-sentimentscore
    elif sentmod=="N":
        sentimentscore=min(1,sentimentscore-2)
    elif sentmod=="P":
        sentimentscore=max(5,sentimentscore+2)
    
    if(exmovies):
        extm=userprofiles.cm[userid]
        for exmovie in exmovies:    
            extm.insert(0,exmovie)
        extm=extm[0:3]
        userprofiles.set_value(userid,'cm',extm)
        userprofiles.set_value(0,'cm',extm)
    if(exgenre):
        extg=userprofiles.cg[userid]
        for exgen in exgenre: 
            extg.insert(0,exgen)
        extg=extg[0:3]
        userprofiles.set_value(userid,'cg',extg)
        userprofiles.set_value(0,'cg',extg)
    if(exstars):
        exts=userprofiles.ca[userid]
        for exstar in exstars:
            exts.insert(0,exstar)
        exts=exts[0:3]
        userprofiles.set_value(userid,'ca',exts)
        userprofiles.set_value(0,'ca',exts)
    
    keys=exmovies+exgenre+exstars
    watched=pastchecker(sentencelower,tagged)
    if len(keys)==0:
        return keysent,userprofiles
    else:
        table={'key':list(map(str,keys)),
               'tag':tagfinder(keys,exmovies,exgenre,exstars),
               'sent_score':[sentimentscore]*len(keys),
               'watched':[watched]*len(keys),
               'userid':[str(userid)]*len(keys),
               'zprev':[zprev]*len(keys),
               'zsid':[sid]*len(keys)}
        table=pd.DataFrame(table)
        keysent = keysent.append(table)
        return keysent,userprofiles
    
# Sentence Analyzer
def sentenceanalyzer(keysent,sentence,userprofiles,userid,sid,movies,starcast,common,genres):
    tokens = nltk.word_tokenize(sentence)
    postagged = nltk.pos_tag(tokens)
    
    #f.write(str(postagged))
    #f.write("\n")
    if conjunctivechecker(postagged)==1:
        keysent,userprofiles=conjunctiveanalyzer(keysent,sentence,userprofiles,userid,postagged,sid,movies,starcast,common,genres)
    elif comparisonchecker(postagged)==1:
        keysent,userprofiles=comparisonanalyzer(keysent,sentence,userprofiles,userid,postagged,sid,movies,starcast,common,genres)
    else:
        keysent,userprofiles=normalanalyzer(keysent,sentence,userprofiles,userid,postagged,"0",sid,movies,starcast,common,genres)
    return keysent,userprofiles

#User User interaction
def userinteraction(userid, sentences, usernames, umapping, adjlst, movies, actors, common):
    stop_words=get_stop_words('en')
    wordnet_lemmatizer = WordNetLemmatizer()
    porter_stemmer = PorterStemmer()
    recentconv=[]
    agreement=[]
    for i in range(len(sentences)):
        userref=[]
        flag=0
        string=sentences[i]
        for j in range(len(usernames)):
            if usernames[j] in string:
                userref.append(umapping[j])
        string=string.lower()
        text=nltk.word_tokenize(string)
        q=nltk.pos_tag(text)
        tagged = relevantwords(q)
        tagged = [item for item in tagged if item not in stop_words]
        for j in range(len(tagged)):
            tagged[j]=wordnet_lemmatizer.lemmatize(tagged[j])
        tagged = [item for item in tagged if item not in common[0:500]]
        q=str(q).lower()
        q=q.replace("('i', 'prp')","").replace("('me', 'prp')","").replace("('my', 'prp$'","").replace("('we', 'prp')","").replace("('us', 'prp')","").replace("('myself', 'prp')","").replace("('they', 'prp')","").replace("('them', 'prp')","")
        if 'you' in q:
            flag=1
            #print('you: ',q)
        if 'that' in q and flag!=1 and len(userref)==0:
            if movies[i]==0 and actors[i]==0:
                flag=1
        recentconv.append(tagged)
        if 'prp' in q and flag!=1 and len(userref)==0:
            if 'it' in q:
                if movies[i]==0:
                    flag=1
                    #print('it: ', q)
            elif actors[i]==0:
                flag=1
                #print('no actor: ',q)
        maxcount=0
        maxid=-1
        if len(userref)==0 and flag==0 and movies[i]!=1:
            if 'that movie' in string.lower() or 'that film' in string.lower():
                flag=1
            elif 'movie' in string.lower() or 'film' in string.lower():
                flag=1
        if len(userref)==0 and flag==1 and i>=5:
            for j in range(i-5,i):
                cnt=commoncount(recentconv[j],recentconv[i])
                if cnt>=maxcount:
                    maxcount=cnt
                    maxid=j    
        if maxid!=-1 and userid[maxid]==userid[i]:
            flag=0
            #print('restored : ',q)
            #print('maxid : ',maxid)
        if maxcount!=0 and flag!=0:
            userref.append(userid[maxid])
        if len(userref)==0 and flag==1 and i!=0 and maxcount==0 and userid[i-1]!=userid[i]:
            userref.append(userid[i-1])
        #if len(userref)==0 and flag==1 and i==0:
            #userref.append('All')
        if len(userref)!=0:
            table=analyze_this(string,userref)
            capt=[None]*3
            sentiment=[None]*len(userref)
            #print (userref)
            for tx in range(len(userref)):
                capt[0]=table.ix[tx]['neg_score']
                capt[1]=table.ix[tx]['neutral']
                capt[2]=table.ix[tx]['pos_score']
                index=capt.index(max(capt))
                if index==0:
                    sentiment[tx]='Negative'
                    adjlst[int(userid[i])][int(userref[tx])]-=1
                    print((userid[i]+1)," disagreed with ",(userref[tx]+1)," at line number ",i)
                    agreement.append(str(userid[i]+1)+" disagreed with "+str(userref[tx]+1)+" at line number "+str(i))
                elif index==1:
                    sentiment[tx]='Neutral'
                    agreement.append(str(userid[i]+1)+" was neutral to "+str(userref[tx]+1)+" at line number "+str(i))
                else:
                    sentiment[tx]='Positive'
                    adjlst[int(userid[i])][int(userref[tx])]+=1
                    print((userid[i]+1)," agreed with ",(userref[tx]+1)," at line number ",i)
                    agreement.append(str(userid[i]+1)+" agreed with "+str(userref[tx]+1)+" at line number "+str(i))
    return adjlst,agreement
#this function is to be used after a total common set of movies for users has been discovered post plsa/collaborative filtering (there is a set of movies, and each user's rating for that movie is present)
def userinfgraphavgwomisery(userdata, adjlst, numusers, maxmovid, threshold):
    #min weight 0
    #max weight numusers
    movieusermatrix=np.zeros((numusers,maxmovid))
    for i in range(0,10*numusers):
        movieusermatrix[int(userdata.ix[i][0])-1][int(userdata.ix[i][1])-1]=int(userdata.ix[i][2])
    #for i in range(0,numusers):
        #for j in range(0,maxmovid):
    for i in range(0,numusers):
        for j in range(0,numusers):
            for k in range(0,maxmovid):
                movieusermatrix[i][k]=movieusermatrix[i][k]+(adjlst[i][j]/numusers)*(movieusermatrix[j][k]-movieusermatrix[i][k])
    avgs=[0]*maxmovid
    ignore=[]
    for i in range(0,numusers):
        for j in range(0,maxmovid):
            if j in ignore:
                continue
            if movieusermatrix[i][j]<threshold:
                ignore.append(j)
                avgs[j]=0
                continue
            avgs[j]=avgs[j]+movieusermatrix[i][j]
    avgs=np.divide(avgs,maxmovid)
    maxm=max(avgs)
    print(maxm)
    i=avgs.index(maxm)
    avgs[i]=0
    maxm=max(avgs)
    j=avgs.index(maxm)
    avgs[j]=0
    maxm=max(avgs)
    k=avgs.index(maxm)
    print((i+1),(j+1),(k+1))
    return i+1,j+1,k+1

def userinfluence(movieusermatrix, adjlst, numusers, maxmovid, threshold, mode):
    for i in range(0,numusers):
        for j in range(0,numusers):
            for k in range(0,maxmovid):
                movieusermatrix[i][k]=movieusermatrix[i][k]+(adjlst[i][j]/numusers)*(movieusermatrix[j][k]-movieusermatrix[i][k])
    avgs=[0]*maxmovid
    ignore=[]
    for i in range(0,numusers):
        for j in range(0,maxmovid):
            if j in ignore:
                continue
            if movieusermatrix[i][j]<threshold:
                ignore.append(j)
                avgs[j]=-30
                continue
            avgs[j]=avgs[j]+movieusermatrix[i][j]
    #avgs=np.divide(avgs,maxmovid)
    maxm=max(avgs)
    #print(maxm)
    cnsns=maxm/3
    #i=int(np.where(avgs==maxm)[0])
    for i in range(0,len(avgs)):
        if avgs[i]==maxm:
            break
    avgs[i]=-30
    maxm2=max(avgs)
    for j in range(0,len(avgs)):
        if avgs[j]==maxm2:
            break
    cnsns2=maxm2/3
    #j=int(np.where(avgs==maxm)[0])
    avgs[j]=-30
    maxm3=max(avgs)
    for k in range(0,len(avgs)):
        if avgs[k]==maxm3:
            break
    #k=int(np.where(avgs==maxm)[0])
    print((i+1),(j+1),(k+1))
    if mode==1 and cnsns2<0.9*cnsns:
        return i+1,-1,-1
    return i+1,j+1,k+1

def main_callable(conversation,irec,adjlst,userids):
    #program to return updated user-user graph for SENTENCES
    #f=open("results.txt",'a')

    # Linking Stanford api's and Sentiment Analyzer api
    java_path = "C:/Program Files/Java/jre1.8.0_91/bin/java.exe"
    os.environ['JAVAHOME'] = java_path
    parser=StanfordParser('C:/inetpub/wwwroot/server/recommenderSystem/stanford-parser-full-2015-12-09/stanford-parser.jar',
                          'C:/inetpub/wwwroot/server/recommenderSystem/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar')
    #dparser=StanfordDependencyParser('C:/Users/shankarb/Downloads/stanford-parser-full-2015-12-09/stanford-parser-full-2015-12-09/stanford-parser.jar',
                                     #'C:/Users/shankarb/Downloads/stanford-parser-full-2015-12-09/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar')

    
    
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # Loading Required Datasets
    movies=pd.read_csv("C:/Users/nisyadav/AppData/Local/Programs/Python/Python35-32/Lib/collabrec/data/movienames.csv",header=0,sep=",",encoding='latin-1')
    starcast=pd.read_csv("C:/Users/nisyadav/AppData/Local/Programs/Python/Python35-32/Lib/collabrec/data/starcast.csv",header=0,sep=",",encoding='latin-1')
    genres=pd.read_csv("C:/Users/nisyadav/AppData/Local/Programs/Python/Python35-32/Lib/collabrec/data/genres.csv",header=0,sep=",",encoding='latin-1')
    moviegenrereference=["this","that","it"]
    actorreference=["his","her","hers","him","he","she"]
    commonrefrence=["these","those","such"]
    poscompwords=["more","better"]
    negcompwords=["less","worse"]
    common=pd.read_table('C:/Users/nisyadav/AppData/Local/Programs/Python/Python35-32/Lib/collabrec/data/common.csv',sep=',',header=None)
    users1=pd.read_csv("C:/Users/nisyadav/AppData/Local/Programs/Python/Python35-32/Lib/collabrec/data/users.csv")
    common=common[0].tolist()

    # PreProcessing Datasets
    movies.genre=list(map(separator,movies.genre))
    movies.stars=list(map(separator,movies.stars))
    starcast.genre=list(map(separator,starcast.genre))
    starcast.movies=list(map(separator,starcast.movies))
    starcast.genre=list(map(integralconvertor,starcast.genre))
    userprofiles=userprofiler(userids,irec,movies,users1)
    keysent={'key':irec,
             'tag':["m"]*3,
             'sent_score':[3]*3,
             'watched':[0]*3,
             'userid':[0]*3,
             'zprev':[0]*3,
             'zsid':[0]*3}
    keysent=pd.DataFrame(keysent)
    for i in range(conversation.shape[0]):
        sid=i+1
        userid=int(conversation.u_id[i])
        #f.write(conversation.sentence[i])
        #f.write("\n")
        keysent,userprofiles=sentenceanalyzer(keysent,conversation.sentence[i],userprofiles,userid,sid,movies,starcast,common,genres)
        #f.write("\n")
        #print(i)
    keysent.index=range(keysent.shape[0])
    sentences=[]
    userid2=[]
    usernames=[]
    umapping=[]
    #adjlst=np.zeros((4,4))
    movies=[0]*len(conversation)
    actors=[0]*len(conversation)
    genre=[]
    for i in range(0,len(users1)):
        namedat=str(users1.ix[i]['Name'])
        namedat=namedat.split(' ')
        ump=int(users1.ix[i]['u_id'])
        if ump in userids:
            for j in range(0,len(namedat)):
                usernames.append(namedat[j])
                umapping.append(ump)
    for i in range(len(conversation)):
        sentences.append(str(conversation.ix[i]['sentence']))
        userid2.append(userids.index(int(conversation.ix[i]['u_id'])))
    for i in range(1,len(keysent)):
        try:
            if int(keysent.ix[i]['zprev'])+1==1:
                if (str(keysent.ix[i]['tag'])=='m'):
                    movies[int(keysent.ix[i]['zsid'])-1]=1
                elif (str(keysent.ix[i]['tag'])=='s'):
                    actors[int(keysent.ix[i]['zsid'])-1]=1
        except:
            print('Error occured')
            print('i is: ',i)
            print(keysent.ix[i])
    #for i in range(len(movies)):
        #print(i, ': ', movies[i])
        #print(i, ': ', actors[i])
    adjlst,agreement=userinteraction(userid2,sentences,usernames,umapping,adjlst,movies,actors,common)
    #f.close()
    #print(keysent)
    #print(adjlst)
    #keysent.to_csv("session_01results.csv",sep=",")
    return adjlst,keysent,agreement

# Main Function
def mainfunction():
    #program to return updated user-user graph for SENTENCES
    f=open("C:/Python34/Lib/collabrec/results.txt",'a')

    # Linking Stanford api's and Sentiment Analyzer api
    java_path = "C:/Program Files/Java/jre1.8.0_91/bin/java.exe"
    os.environ['JAVAHOME'] = java_path
    parser=StanfordParser('C:/inetpub/wwwroot/server/The Main Code/stanford-parser-full-2015-12-09/stanford-parser.jar',
                          'C:/inetpub/wwwroot/server/The Main Code/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar')
    #dparser=StanfordDependencyParser('C:/Users/shankarb/Downloads/stanford-parser-full-2015-12-09/stanford-parser-full-2015-12-09/stanford-parser.jar',
                                     #'C:/Users/shankarb/Downloads/stanford-parser-full-2015-12-09/stanford-parser-full-2015-12-09/stanford-parser-3.6.0-models.jar')

    
    
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    # Loading Required Datasets
    movies=pd.read_csv("C:/Python34/Lib/collabrec/data/movienames.csv",header=0,sep=",",encoding='latin-1')
    starcast=pd.read_csv("C:/Python34/Lib/collabrec/data/starcast.csv",header=0,sep=",",encoding='latin-1')
    common=pd.read_csv("C:/Python34/Lib/collabrec/data/common.csv",header=0,sep=",",encoding='latin-1')
    genres=pd.read_csv("C:/Python34/Lib/collabrec/data/genres.csv",header=0,sep=",",encoding='latin-1')

    conversation=pd.read_csv("C:/Python34/Lib/collabrec/session_01.csv",header=0,sep=",",encoding='latin-1')

    moviegenrereference=["this","that","it"]
    actorreference=["his","her","hers","him","he","she"]
    commonrefrence=["these","those","such"]
    poscompwords=["more","better"]
    negcompwords=["less","worse"]
    common=pd.read_table('C:/Python34/Lib/collabrec/data/common.csv',sep=',',header=None)
    common=common[0].tolist()
    users1=pd.read_csv("C:/Python34/Lib/collabrec/data/users.csv")
    # PreProcessing Datasets
    movies.genre=list(map(separator,movies.genre))
    movies.stars=list(map(separator,movies.stars))
    starcast.genre=list(map(separator,starcast.genre))
    starcast.movies=list(map(separator,starcast.movies))
    starcast.genre=list(map(integralconvertor,starcast.genre))
    userids = [1,2,3,4]
    irec=296
    userprofiles=userprofiler(userids,irec,movies,users1)
    keysent={'key':[irec],
             'tag':["m"],
             'sent_score':[3],
             'watched':[0],
             'userid':[0],
             'zprev':[0],
             'zsid':[0]}
    keysent=pd.DataFrame(keysent)
    for i in range(conversation.shape[0]):
        sid=i+1
        userid=int(conversation.u_id[i])
        f.write(conversation.sentence[i])
        f.write("\n")
        keysent,userprofiles=sentenceanalyzer(keysent,conversation.sentence[i],userprofiles,userid,sid,movies,starcast,common,genres,f)
        f.write("\n")
        print(i)
    keysent.index=range(keysent.shape[0])
    sentences=[]
    userid2=[]
    usernames=[]
    umapping=[]
    adjlst=np.zeros((4,4))
    movies=[0]*len(conversation)
    actors=[0]*len(conversation)
    genre=[]
    for i in range(len(conversation)):
        sentences.append(str(conversation.ix[i]['sentence']))
        userid2.append(int(conversation.ix[i]['u_id']))
    for i in range(1,len(keysent)):
        try:
            if int(keysent.ix[i]['zprev'])+1==1:
                if (str(keysent.ix[i]['tag'])=='m'):
                    movies[int(keysent.ix[i]['zsid'])-1]=1
                elif (str(keysent.ix[i]['tag'])=='s'):
                    actors[int(keysent.ix[i]['zsid'])-1]=1
        except:
            print('Error occured')
            print('i is: ',i)
            print(keysent.ix[i])
    for i in range(len(movies)):
        print(i, ': ', movies[i])
        print(i, ': ', actors[i])
    adjlst=userinteraction(userid2,sentences,usernames,umapping,adjlst,movies,actors,common)
    f.close()
    print(keysent)
    print(adjlst)
    #keysent.to_csv("session_01results.csv",sep=",")
    #return keysent
