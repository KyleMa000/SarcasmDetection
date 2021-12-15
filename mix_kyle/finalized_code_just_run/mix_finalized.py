import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import csv
import os
import pickle
from sklearn import linear_model
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

class Feature:
    def __init__(self, file_dir,spoken_write_dir):
        self.intra_sentences = []  # Intra_sentence_contradiction
        self.inter_sentences = []  # Inter_sentence_contradiction
        self.neg_c = []  # Negating coordinating conjunction
        self.neg_a = []  # Negating adverb between sentences
        self.All_cap = []  # All-cap word
        self.Counts_p = []  # Count of punctuation
        self.tokens = [] #all sentence
        self.labels = []
        self.bertIs = [] #KYLE edited adding bert is scarsm probability as a feature
        self.bertNot = [] #KYLE edited adding bert not scarsm probability as a feature
        self.num = 0
        # 需要使用训练数据中的单词构建一个词表，形式为"word":1,存在word_id.bin文件中，如果要用新的语料生成词表，可将该文件删除后引用新文件路径
        self.id_word = self.wordtransferid(file_dir)
        self.spoken_write = self.spoken_write_load(spoken_write_dir)  # load spoken_written words
        self.spoken_writen_mark = []                            #Spoken_written mark

    """
    构建输入特征，前半部分为原句的词编码，后半部分为句中矛盾、标点出现次数等特征
    注意：目前该构建好的特征可直接输入logistic regression进行训练，但如果想用lstm，需要对单词进行embedding。通过文本提取出的特征需要寻找各自的embedding方式
    另外如果想使用bert模型进行预测，目前特征构建方法和bert模型的融合有一定难度。因bert词向量输入的维度固定为768维。所以不能采用维度拼接的方式，只能在句子长度上进行拼接。
    """
    def InputFeatures(self):
        Input = []
        for i in range(0, len(self.tokens)):
            InputX = []

            tokens = self.tokens[i]
            lentokens = len(tokens)
            if lentokens > 20:
                for j in range(0, 20):
                    InputX.append(tokens[j])
            else:
                for j in range(0, lentokens):
                    InputX.append(tokens[j])
                for x in range(0, 20 - lentokens):
                    InputX.append(-1)
            BertIs = self.bertIs[i] #Kyle added feature bert is
            InputX.append(BertIs) #kyle added
            BertNot = self.bertNot[i] #kyle added feature bert not
            InputX.append(BertNot) #kyle added
            neg_c_flag = self.neg_c[i]
            InputX.append(neg_c_flag)
            neg_a_flag = self.neg_a[i]
            InputX.append(neg_a_flag)
            all_cap_flag = self.All_cap[i]
            InputX.append(all_cap_flag)
            couts_p_num = self.Counts_p[i]
            InputX.append(couts_p_num)
            intra_s_flag = self.intra_sentences[i]
            InputX.append(intra_s_flag)
            spoken_write_flag = self.spoken_writen_mark[i]
            InputX.append(spoken_write_flag)
            inter_s_flag = self.inter_sentences[i]
            InputX.append(inter_s_flag)
            Input.append(InputX)

        return Input

    def spoken_write_load(self, spoken_write_dir):
        if os.path.exists(spoken_write_dir):
            with open(spoken_write_dir, "rb") as f:
                data = pickle.load(f)
            return data
        else:
            return -1
    #按照论文的描述，对句子中是否有书面语和口语的反转进行检测
    def spoken_writen_reverse_detection(self, tokens):
        tokens = tokens.split(" ")
        spoken = 0
        nonestyle = 0
        written = 0
        for word in tokens:
            if word in self.spoken_write:
                mark = self.spoken_write.get(word)
                if mark == -1:
                    written = written + 1
                elif mark == 1:
                    spoken = spoken + 1
                else:
                    nonestyle = nonestyle + 1
        if spoken==0 or written ==0:
            return 0
        else:
            sum = spoken + written
            gap = spoken - written
            if gap < 0:
                gap = 0 - gap
            elif gap == 0:
                return 1
            if sum/gap > 2:
                return 1
            else:
                return 0

    def wordtransferid(self, fiel_dir):
        if os.path.exists("word_id.bin"):
            with open("word_id.bin", "rb") as f:
                data = pickle.load(f)
            return data
        else:
            word_id = {"UKN": 0}
            f = open(fiel_dir, "r", encoding="utf-8")
            next(f)
            csv_file = csv.reader(f)
            index = 1
            for line in csv_file:
                tokens = line[1]
                tokens = tokens.split(" ")
                for word in tokens:
                    if word_id.__contains__(word):
                        pass
                    else:
                        word_id[word] = index
                        index = index + 1
            with open("word_id.bin", "wb")as f:
                pickle.dump(word_id, f)
        return word_id


    def Neg_coordinating_conjunction(self, tokens):
        list_words = ["but", "nor"]
        tokens = tokens.split(" ")
        flag = 0
        for word in list_words:
            if word in tokens:
                flag = 1
        return flag

    def Neg_adverb(self, tokens):
        list_words = ["However", "Nevertheless", "however", "nevertheless"]
        tokens = tokens.split(" ")
        flag = 0
        for word in list_words:
            if word in tokens:
                # print("exsit Neg_adverb")
                flag = 1
        return flag

    def Inter_sentence_contradiction(self, tokens):
        analyzer = SentimentIntensityAnalyzer()
        tokens = tokens.split(",")
        neg = 0
        pos = 0
        for word in tokens:
            vs = analyzer.polarity_scores(word)
            compound = float(vs['compound'])
            if compound < 0:
                neg = 1
            elif compound > 0:
                pos = 1
        if neg == 1 and pos == 1:
            return 1
        else:
            return 0

    def Intra_sentence_contradiction(self, tokens):
        analyzer = SentimentIntensityAnalyzer()
        tokens = tokens.split(" ")
        neg = 0
        pos = 0
        for word in tokens:
            vs = analyzer.polarity_scores(word)
            compound = float(vs['compound'])
            if compound < 0:
                neg = 1
            elif compound > 0:
                pos = 1
        if neg == 1 and pos == 1:
            return 1
        else:
            return 0

    def All_cap_word(self, tokens):
        flag = 0
        tokens = tokens.split(" ")
        for word in tokens:
            if word.isupper():
                flag = 1
        return flag

    def Count_of_punctuation(self, tokens):
        num = 0
        for s in tokens:
            if s != " ":
                if s.isalnum() == False:
                    num = num + 1
        return num

    def read_train(self, fiel_dir):
        print("-------train dataset loader---------")
        f = open(fiel_dir, "r", encoding="utf-8")
        next(f)
        csv_file = csv.reader(f)
        for line in csv_file:
            self.num = self.num + 1
            if self.num % 1000 == 0:
                print("-------loading-------")
            label = line[0]
            self.labels.append(label)
            token = line[1]
            self.bertIs.append(float(line[2])) #Kyle added adding feature
            self.bertNot.append(float(line[3])) #Kyle added adding feature
            word_ids = []
            for word in token.split():
                if self.id_word.__contains__(word):
                    word_ids.append(self.id_word[word])
                else:
                    word_ids.append(0)
            self.tokens.append(word_ids)
            punctuation_num = self.Count_of_punctuation(token)
            self.Counts_p.append(punctuation_num)
            uppermark = self.All_cap_word(token)
            self.All_cap.append(uppermark)
            Intra_mark = self.Intra_sentence_contradiction(token)
            self.intra_sentences.append(Intra_mark)
            neg_c_mark = self.Neg_coordinating_conjunction(token)
            self.neg_c.append(neg_c_mark)
            neg_a_mark = self.Neg_adverb(token)
            self.neg_a.append(neg_a_mark)
            spoken_write_mark = self.spoken_writen_reverse_detection(token)
            self.spoken_writen_mark.append(spoken_write_mark)
            Inter_mask = self.Inter_sentence_contradiction(token)
            self.inter_sentences.append(Inter_mask)


    def read_test(self, fiel_dir):
        print("-------test dataset loader---------")
        f = open(fiel_dir, "r", encoding="utf-8")
        next(f)
        csv_file = csv.reader(f)
        for line in csv_file:
            self.num = self.num + 1
            if self.num % 1000 == 0:
                print("-------loading-------")
            label = line[0]
            self.labels.append(label)
            token = line[1]
            self.bertIs.append(float(line[2])) #Kyle added adding feature
            self.bertNot.append(float(line[3])) #Kyle added adding feature
            word_ids = []
            for word in token.split():
                if self.id_word.__contains__(word):
                    word_ids.append(self.id_word[word])
                else:
                    word_ids.append(0)
            self.tokens.append(word_ids)
            punctuation_num = self.Count_of_punctuation(token)
            self.Counts_p.append(punctuation_num)
            uppermark = self.All_cap_word(token)
            self.All_cap.append(uppermark)
            Intra_mark = self.Intra_sentence_contradiction(token)
            self.intra_sentences.append(Intra_mark)
            neg_c_mark = self.Neg_coordinating_conjunction(token)
            self.neg_c.append(neg_c_mark)
            neg_a_mark = self.Neg_adverb(token)
            self.neg_a.append(neg_a_mark)
            spoken_write_mark = self.spoken_writen_reverse_detection(token)
            self.spoken_writen_mark.append(spoken_write_mark)
            Inter_mask = self.Inter_sentence_contradiction(token)
            self.inter_sentences.append(Inter_mask)

if __name__ == '__main__':

    #load train file
    trainfeature = Feature("mix_train.csv","spoken_write.pkl")
    trainfeature.read_train("mix_train.csv")
    input = trainfeature.InputFeatures()
    
    #load testing file
    testfeature = Feature("mix_test.csv", "spoken_write.pkl")
    testfeature.read_test("mix_test.csv")
    testinput = testfeature.InputFeatures()
    test_error_analysis = pd.read_csv(r"mix_test.csv")
    
    #load answer key
    test = np.array(testfeature.labels)
    test = test.tolist()
    trans_test = []
    for key in test:
        trans_test.append(int(key))
    
    
    #container for logistic regression
    accuracy_list_lr = []
    precision_list_lr = []
    recall_list_lr = []
    f1score_list_lr = []
    
    
    #container for decision tree
    accuracy_list_dt = []
    precision_list_dt = []
    recall_list_dt = []
    f1score_list_dt = []
    
    
    for i in range(5):
        #logistic regression
        lr = linear_model.LogisticRegression(solver='liblinear')
        lr.fit(input, trainfeature.labels)
        
        
        #decision tree
        clf = tree.DecisionTreeClassifier()
        clf.fit(input, trainfeature.labels)
        
        
        #predict logistic regression
        predict_lr = lr.predict(testinput)
    
    
        #evaluation logistic regression
        predict_lr = predict_lr.tolist()
        trans_predict_lr = []
    
    
        for key in predict_lr:
            trans_predict_lr.append(int(key))
        a = accuracy_score(trans_test, trans_predict_lr)
        accuracy_list_lr.append(a)
        # print("accuracy:" + str(a))
        p = precision_score(trans_test, trans_predict_lr)
        precision_list_lr.append(p)
        # print("precision:" + str(p))
        r = recall_score(trans_test, trans_predict_lr)
        recall_list_lr.append(r)
        # print("recall:" + str(r))
        f1score = f1_score(trans_test, trans_predict_lr)
        f1score_list_lr.append(f1score)
        # print("f1score:" + str(f1score))
        
        
        #predict decision tree
        predict_dt = clf.predict(testinput)
        
        
        #evaluation decision tree
        predict_dt = predict_dt.tolist()
        trans_predict_dt = []
        for key in predict_dt:
            trans_predict_dt.append(int(key))
        a = accuracy_score(trans_test, trans_predict_dt)
        accuracy_list_dt.append(a)
        # print("accuracy:" + str(a))
        p = precision_score(trans_test, trans_predict_dt)
        precision_list_dt.append(p)
        # print("precision:" + str(p))
        r = recall_score(trans_test, trans_predict_dt)
        recall_list_dt.append(r)
        # print("recall:" + str(r))
        f1score = f1_score(trans_test, trans_predict_dt)
        f1score_list_dt.append(f1score)
        # print("f1score:" + str(f1score))
    
    print("Logistic Regression:")
    
    # print(accuracy_list_lr)
    # print(precision_list_lr)
    # print(recall_list_lr)
    # print(f1score_list_lr)
    
    print("Max:")
    print("max accuracy: " + str(max(accuracy_list_lr)))
    print("max precision: " + str(max(precision_list_lr)))
    print("max recall: " + str(max(recall_list_lr)))
    print("max f1score: " + str(max(f1score_list_lr)))
    print("Avg")
    print("avg accuracy: " + str(sum(accuracy_list_lr)/5))
    print("avg precision: " + str(sum(precision_list_lr)/5))
    print("avg recall: " + str(sum(recall_list_lr)/5))
    print("avg f1score: " + str(sum(f1score_list_lr)/5))
    
    print("Decision Tree:")

    # print(accuracy_list_dt)
    # print(precision_list_dt)
    # print(recall_list_dt)
    # print(f1score_list_dt)
    
    print("Max:")
    print("max accuracy: " + str(max(accuracy_list_dt)))
    print("max precision: " + str(max(precision_list_dt)))
    print("max recall: " + str(max(recall_list_dt)))
    print("max f1score: " + str(max(f1score_list_dt)))

    print("Avg")
    print("avg accuracy: " + str(sum(accuracy_list_dt)/5))
    print("avg precision: " + str(sum(precision_list_dt)/5))
    print("avg recall: " + str(sum(recall_list_dt)/5))
    print("avg f1score: " + str(sum(f1score_list_dt)/5))
    
    
    #using the 5th try's result for error analysis
    
    lr_false = []
    dt_false = []
    for i in range(len(trans_test)):
        if trans_test[i] != trans_predict_lr[i]:
            lr_false.append(True)
        else:
            lr_false.append(False)
        if trans_test[i] != trans_predict_dt[i]:
            dt_false.append(True)
        else:
            dt_false.append(False)
    
    lr_error_analysis = test_error_analysis[lr_false]
    dt_error_analysis = test_error_analysis[dt_false]
    
    lr_error_analysis.to_csv(r'mix_lr_error_analysis.csv', index = None)
    dt_error_analysis.to_csv(r'mix_dt_error_analysis.csv', index = None)
