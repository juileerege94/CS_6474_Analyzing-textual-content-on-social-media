# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 23:42:36 2016

@author: Juilee Rege
"""

import re
import os
import random
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

r = re.compile("[(){},.;!:?\-<>$%\"0-9]")

class Ass_J_1:
    
    def __init__(self):
        pass
    
    def removePunc(self,inp):
        return r.sub('',inp)
    
    def process1(self,vectorizer,model1,model2,model3):
        
        path = os.getcwd()
        
        #negative file
        mydata = pd.read_csv(path + '/neg_examples_sad.txt',header=None,sep='delimiter')
        words=[]
        ngrams1=['@']
        ngrams1=[]
        for i in range(0,len(mydata)):
            mydata[0][i] = mydata[0][i].lower()
            w=mydata[0][i].split(' ')
            words=[]
            for j in range(0,len(w)):
                if(('@' in w[j]) or ('http' in w[j]) or ('www' in w[j]) or ('.com' in w[j]) or ('rt' in w[j])):
                    continue 
                else:
                    words.append(w[j])
            s=""
            s = " ".join(str(x) for x in words)
            s = self.removePunc(s)
            ngrams1.append(s)
        
        #writing all cleaned sentences into countNeg.txt
        thefile = open(path + '/countNeg_Part2.txt', 'w')
        for item in ngrams1:
            thefile.write("%s\n" % item)
    
        #randomly selecting 80% of the lines from count.txt and putting into countRandom.txt
        with open(path + '/countNeg_Part2.txt','r') as source:
            dataNeg = [ (random.random(), line) for line in source ]
            dataNeg.sort()
            
        #positive file
        mydata = pd.read_csv(path + '/pos_examples_happy.txt',header=None,sep='delimiter')
        words=[]
        ngrams1=['@']
        ngrams1=[]
        for i in range(0,len(mydata)):
            mydata[0][i] = mydata[0][i].lower()
            w=mydata[0][i].split(' ')
            words=[]
            for j in range(0,len(w)):
                if(('@' in w[j]) or ('http' in w[j]) or ('www' in w[j]) or ('.com' in w[j]) or ('rt' in w[j])):
                    continue 
                else:
                    words.append(w[j])
            s=""
            s = " ".join(str(x) for x in words)
            s = self.removePunc(s)
            ngrams1.append(s)
        
        #writing all cleaned sentences into count.txt
        thefile = open(path + '/countPos_Part2.txt', 'w')
        for item in ngrams1:
            thefile.write("%s\n" % item)
    
        #randomly selecting 80% of the lines from count.txt and putting into countRandom.txt
        with open(path + '/countPos_Part2.txt','r') as source:
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
        
        input2 = vectorizer.transform(data1).toarray()
        
        final_answer_1 = model1.predict(input2)
        p_1 = precision_score(target, final_answer_1)
        r_1 = recall_score(target, final_answer_1)
        f_1 = f1_score(target, final_answer_1)
        a_1 = accuracy_score(target, final_answer_1)
        print "The accuracy, precision, recall and f1 score for the best model of Random Forest classifier on Second part is: ", a_1,p_1,r_1,f_1        
        
        final_answer_2 = model2.predict(input2)
        p_2 = precision_score(target, final_answer_2)
        r_2 = recall_score(target, final_answer_2)
        f_2 = f1_score(target, final_answer_2)
        a_2 = accuracy_score(target, final_answer_2)
        print "The accuracy, precision, recall and f1 score for the best model of Naive Bayes classifier on Second part is: ", a_2,p_2,r_2,f_2
        
        final_answer_3 = model3.predict(input2)
        p_3 = precision_score(target, final_answer_3)
        r_3 = recall_score(target, final_answer_3)
        f_3 = f1_score(target, final_answer_3)
        a_3 = accuracy_score(target, final_answer_3)
        print "The accuracy, precision, recall and f1 score for the best model of Neural Network classifier on Second part is: ", a_3,p_3,r_3,f_3
        