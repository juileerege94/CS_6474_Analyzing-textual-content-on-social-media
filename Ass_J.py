# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 10:58:22 2016

@author: Juilee Rege
"""
import re
import os
import numpy as np
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.cross_validation import train_test_split
from Ass_J_1 import Ass_J_1

r = re.compile("[(){},.;!:?\-<>$%\"0-9]")

def removePunc(inp):
    return r.sub('',inp)
    
def process():
    
    path = os.getcwd()
    #negative file
    mydata = pd.read_csv(path + '/neg_examples_NegSentiment.txt',header=None)
    words=[]
    ngrams1=['@']
    ngrams1=[]
    for i in range(0,len(mydata)):
        mydata[0][i] = mydata[0][i].lower()
        w=mydata[0][i].split(' ')
        words=[]
        for j in range(0,len(w)):
            if(('@' in w[j]) or ('http' in w[j]) or ('www' in w[j]) or ('.com' in w[j]) or ('happy' in w[j]) or ('sad' in w[j])):
                continue 
            else:
                words.append(w[j])
        s=""
        s = " ".join(str(x) for x in words)
        s = removePunc(s)
        ngrams1.append(s)
    
    #writing all cleaned sentences into countNeg.txt
    thefile = open(path + '/countNeg_Part1.txt', 'w')
    for item in ngrams1:
        thefile.write("%s\n" % item)

    #randomly selecting 80% of the lines from count.txt and putting into countRandom.txt
    with open(path + '/countNeg_Part1.txt','r') as source:
        dataNeg = [ (random.random(), line) for line in source ]
        dataNeg.sort()
        
    #positive file
    mydata = pd.read_csv(path + '/pos_examples_PosSentiment.txt',header=None)
    words=[]
    ngrams1=['@']
    ngrams1=[]
    for i in range(0,len(mydata)):
        mydata[0][i] = mydata[0][i].lower()
        w=mydata[0][i].split(' ')
        words=[]
        for j in range(0,len(w)):
            if(('@' in w[j]) or ('http' in w[j]) or ('www' in w[j]) or ('.com' in w[j]) or ('happy' in w[j]) or ('sad' in w[j])):
                continue 
            else:
                words.append(w[j])
        s=""
        s = " ".join(str(x) for x in words)
        s = removePunc(s)
        ngrams1.append(s)
    
    #writing all cleaned sentences into count.txt
    thefile = open(path + '/countPos_Part1.txt', 'w')
    for item in ngrams1:
        thefile.write("%s\n" % item)

    #randomly selecting 80% of the lines from count.txt and putting into countRandom.txt
    with open(path + '/countPos_Part1.txt','r') as source:
        dataPos = [ (random.random(), line) for line in source ]
        dataPos.sort()
    
    #appending both positive and negative files together
    data=[]
    for _,line in dataPos:
        data.append(line)
    for _,line in dataNeg:
        data.append(line)
    
    data1 = []
    for line in data:
        data1.append(line.rstrip('\n'))
    
    target1 = [1]*len(dataPos)
    target2 = [-1]*len(dataNeg)
    target=target1+target2
    
    X_train, X_test, y_train, y_test = train_test_split(data1, target, test_size=0.2, random_state=0)

    #passing all random cleaned lines to CountVectorizer to get ngrams
    my_words = ['just','im']
    sw = text.ENGLISH_STOP_WORDS.union(my_words)
    vectorizer = CountVectorizer(input='content',analyzer=u'word',tokenizer=None,ngram_range=(1, 3), stop_words=sw, lowercase=True,max_features=1000)
    X = vectorizer.fit_transform(X_train)
    X=X.toarray()
    vocab = vectorizer.get_feature_names()
    dist = np.sum(X, axis=0)
    vocab.to_csv(path + "/TopFrequencyWords.txt", index=False, cols=('term','occurrences'),encoding='utf-8')
    
    input1 = vectorizer.transform(X_train).toarray()
    input2 = vectorizer.transform(X_test).toarray()
    
    #implementing stratified k-fold splitting
    skf = StratifiedKFold(n_splits=5,shuffle=True)

    k=0
    max_f1 = 0
    max_f2 = 0
    max_f3 = 0    
    RF = []
    NB = []
    NN = []
    for i in range(0,5):
        RF.append(RandomForestClassifier())
        NB.append(GaussianNB())
        NN.append(MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1))
    
    for train_index, test_index in skf.split(input1, y_train):
        y_train1 = []
        y_test1 = []
        X_train1, X_test1 = input1[train_index], input1[test_index]
        for i in train_index:
            y_train1.append(y_train[i])
        for j in test_index:
            y_test1.append(y_train[j])
        
        RF[k].fit(X_train1,y_train1)
        answer1 = RF[k].predict(X_test1)
        p1 = precision_score(y_test1, answer1)
        r1 = recall_score(y_test1, answer1)
        f1 = f1_score(y_test1, answer1)
        a1 = accuracy_score(y_test1, answer1)
        print "accuracy, precision, recall, f1 for ",k," th model of RF: ", a1, p1, r1, f1
        
        if(f1>max_f1):
            model1 = RF[k]
            model_no1 = k
            max_f1 = f1

        NB[k].fit(X_train1,y_train1)
        answer2 = NB[k].predict(X_test1)
        p2 = precision_score(y_test1, answer2)
        r2 = recall_score(y_test1, answer2)
        f2 = f1_score(y_test1, answer2)
        a2 = accuracy_score(y_test1, answer2)
        print "accuracy, precision, recall, f1 for ",k," th model of NB: ", a2, p2, r2, f2
        
        if(f2>max_f2):
            model2 = NB[k]
            model_no2 = k
            max_f2 = f2
        

        NN[k].fit(X_train1,y_train1)
        answer3 = NN[k].predict(X_test1)
        p3 = precision_score(y_test1, answer3)
        r3 = recall_score(y_test1, answer3)
        f3 = f1_score(y_test1, answer3)
        a3 = accuracy_score(y_test1, answer3)
        print "accuracy, precision, recall, f1 for ",k," th model of NN: ", a3, p3, r3, f3
        
        if(f3>max_f3):
            model3 = RF[k]
            model_no3 = k
            max_f3 = f3
                
        k=k+1
    
    final_answer_1 = model1.predict(input2)
    p_1 = precision_score(y_test, final_answer_1)
    r_1 = recall_score(y_test, final_answer_1)
    f_1 = f1_score(y_test, final_answer_1)
    a_1 = accuracy_score(y_test, final_answer_1)
    
    print "The best model is: ", model_no1
    print "The accuracy, precision, recall and f1 score for the best model of Random Forest classifier is: ", a_1,p_1,r_1,f_1        
    
    final_answer_2 = model2.predict(input2)
    p_2 = precision_score(y_test, final_answer_2)
    r_2 = recall_score(y_test, final_answer_2)
    f_2 = f1_score(y_test, final_answer_2)
    a_2 = accuracy_score(y_test, final_answer_2)
    
    print "The best model is: ", model_no2
    print "The accuracy, precision, recall and f1 score for the best model of Naive Bayes classifier is: ", a_2,p_2,r_2,f_2        
    
    final_answer_3 = model3.predict(input2)
    p_3 = precision_score(y_test, final_answer_3)
    r_3 = recall_score(y_test, final_answer_3)
    f_3 = f1_score(y_test, final_answer_3)
    a_3 = accuracy_score(y_test, final_answer_3)
    
    print "The best model is: ",model_no3
    print "The accuracy, precision, recall and f1 score for the best model of Neural Networks classifier is: ", a_3,p_3,r_3,f_3

    ass = Ass_J_1()
    ass.process1(vectorizer,model1,model2,model3)
    
if __name__ == "__main__":
    process()