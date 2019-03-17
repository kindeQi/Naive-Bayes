from pathlib import Path
from collections import defaultdict
import os
from math import log, log2
import sys, glob
import json

class bayes_dataset(object):
    def __init__(self, dataset):
        '''
        Arguments:
        dataset: List[List[List[str(Path), tuple(label1, label2)]]], shape(4, k)
        '''
        # self.dataset = [dataset[i][j] for i in range(len(dataset)) for j in range(len(dataset[0]))]
        self.dataset = dataset
        self.captial_list = set([chr(i) for i in range(65, 90 + 1)])
        self.lower_list   = set([chr(i) for i in range(97, 122 + 1)])
        
    def __getitem__(self, index):
        '''
        Arguments:
        index: int, the inde of required item
        
        Description:
        get the data, tokenize and get the target
        '''
        path, label = self.dataset[index]
        return self.tokenize(path, label), label
    
    def __len__(self):
        return len(self.dataset)
    
    def tokenize(self, file_path, file_label):
        '''
        Argument:
        file_path: the path of input file, Path()
        file_label: List(label1, label2)
        
        Description:
        1. ignore the word contain special character directly
        
        rules:
        remove all special character
        '''
        content = ""
        with open(file_path, 'r') as f:
            original_content = f.readline()
            content = original_content[:-1]
            content = content.split(" ")
            
            word_list = []
            for word in content:
                w = ''
                for letter in word:
                    if letter in self.captial_list:
                        w += str.lower(letter)
                    elif letter in self.lower_list:
                        w += letter
                    elif letter in set(['!', '?', '/', '&', '*']):
                        w += letter
                if w != '':
                    word_list.append(w)
            
        return word_list
    
    def stop_list(self, low_frequency, high_frequency):
        '''
        Argument:
        low_frequency: int, the low bound for frequency
        high_frequency: int, the high bound for frequency
        '''

class F1_score(object):
    def __init__(self, num_class):
        
        self.num_class = num_class
        
        self.true_positive = 'true_positive'
        self.true_negative = 'true_negative'
        self.false_positive = 'false_positive'
        self.false_negative = 'false_negative'
        
        self.class_score = [defaultdict(int) for _ in range(self.num_class)]
    
    def add_score(self, gold_cls, sys_cls):
        '''
        Arguments
        gold_cls in range(0, 4)
        sys_cls  in range(0, 4)
        '''
        if gold_cls == sys_cls:
            self.class_score[gold_cls][self.true_positive] += 1
        else:
            self.class_score[gold_cls][self.false_negative] += 1
            self.class_score[sys_cls][self.false_positive] += 1
    
    def calculate_F1_score(self):
        recall = [c[self.true_positive] / (c[self.true_positive] + c[self.false_negative]) for c in self.class_score]
        precision = [c[self.true_positive] / (c[self.true_positive] + c[self.false_positive]) for c in self.class_score]
        
        F1 = [(2 * recall[i] * precision[i]) / (recall[i] + precision[i]) for i in range(self.num_class)]
        
        # print('recall: ', recall)
        # print('precision: ', precision)
        # print('F1: ', F1)
        
        return sum(F1) / len(F1), recall, precision, F1

def train(data_loader):
    total_vocabulary = defaultdict(int)
    label_vocabulary = [defaultdict(int) for _ in range(4)]

    total_word_num = 0
    count_label = [0 for _ in range(4)]

    p_class = [0.25 for _ in range(4)]

    # trn_dataloader = bayes_dataset(trn_dataset)
    for content, label in data_loader:
        for word in content:
            total_vocabulary[word] += 1
            label_vocabulary[label[0]][word] += 1
            label_vocabulary[label[1]][word] += 1

            total_word_num += 1
            count_label[label[0]] += 1
            count_label[label[1]] += 1

    # feature selection according to information gain
    top_k = 4000
    epsilon = 1e-10
    seletcted_dict = dict()
    for word in total_vocabulary:
        prob_sum = sum([label_vocabulary[i][word] for i in range(4)])
        prob = [(label_vocabulary[i][word] + epsilon) / prob_sum for i in range(4)]
        seletcted_dict[word] = sum(prob)

    sorted_keys = sorted(seletcted_dict, key=lambda _key: seletcted_dict[_key])

    # only use top feature to predict
    

    V = len(total_vocabulary)
    return V, p_class, count_label, label_vocabulary, total_vocabulary


def train_with_stop_list_ngram(data_loader, low_frequency, high_frequency):
    # 0. get the ngram 2 and 3
    ngram_dict = defaultdict(int)
    for content, label in data_loader:
        for i in range(len(content) - 1):
            word = content[i] + content[i + 1]
            ngram_dict[word] += 1
            
    # for content, label in data_loader:
    #     for i in range(len(content) - 2):
    #         word = content[i] + content[i + 1] + content[i + 2]
    #         ngram_dict[word] += 1
    
    # 1. get the vocabulary
    big_vocabulary = defaultdict(int)
    for content, label in data_loader:
        for word in content:
            big_vocabulary[word] += 1
    
    # 2. get the stop list
    stop_list = set()
    for k in big_vocabulary.keys():
        if big_vocabulary[k] < low_frequency or big_vocabulary[k] > high_frequency:
            stop_list.add(k)
            
    for k in ngram_dict.keys():
        if ngram_dict[k] < 3:
            stop_list.add(k)
    # 3. get new stuff
    total_vocabulary = defaultdict(int)
    label_vocabulary = [defaultdict(int) for _ in range(4)]

    total_word_num = 0
    count_label = [0 for _ in range(4)]

    p_class = [0.25 for _ in range(4)]

    # trn_dataloader = bayes_dataset(trn_dataset)
    for content, label in data_loader:
        for word in content:
            if word in stop_list:
                continue
            else:
                total_vocabulary[word] += 1
                label_vocabulary[label[0]][word] += 1
                label_vocabulary[label[1]][word] += 1

                total_word_num += 1
                count_label[label[0]] += 1
                count_label[label[1]] += 1
    
    for content, label in data_loader:
        for i in range(len(content) - 1):
            word = content[i] + content[i + 1]
            if word in stop_list:
                continue
            else:
                total_vocabulary[word] += 1
                label_vocabulary[label[0]][word] += 1
                label_vocabulary[label[1]][word] += 1

                total_word_num += 1
                count_label[label[0]] += 1
                count_label[label[1]] += 1
    
    # for content, label in data_loader:
    #     for i in range(len(content) - 2):
    #         word = content[i] + content[i + 1] + content[i + 2]
    #         if word in stop_list:
    #             continue
    #         else:
    #             total_vocabulary[word] += 1
    #             label_vocabulary[label[0]][word] += 1
    #             label_vocabulary[label[1]][word] += 1

    #             total_word_num += 1
    #             count_label[label[0]] += 1
    #             count_label[label[1]] += 1
    
    V = len(total_vocabulary)
    return V, p_class, count_label, label_vocabulary, total_vocabulary, stop_list, big_vocabulary  

# data_loader = bayes_dataset(trn_dataset)
# V, p_class, count_label, label_vocabulary, total_vocabulary, stop_list, big_vocabulary = train_with_stop_list_ngram(data_loader, 5, 100000)

if __name__ == "__main__":
    
    # 1. file lists
    split_word = '\\' if sys.platform == 'win32' else '/'
    path_to_input = 'op_spam_v1.4/' if sys.platform == 'win32' else sys.argv[1]

    all_files = glob.glob(os.path.join(path_to_input,'*/*/*/*.txt'))

    idx_catagory = {0: 'negative', 1: 'positive', 2: 'deceptive', 3: 'truthful'}
    catagory_idx = {'negative': 0, 'positive': 1, 'deceptive': 2, 'truthful': 3}

    trn_dataset = []
    val_dataset = []

    for file in all_files:
        class1, class2, fold, fname = file.split(split_word)[-4:]
        class1, class2 = class1.split('_')[0], class2.split('_')[0]
        class1, class2 = catagory_idx[class1], catagory_idx[class2]
        # trn_dataset.append([file, (class1, class2)])
        if fold != 'fold5':
            trn_dataset.append([file, (class1, class2)])
        else:
            val_dataset.append([file, (class1, class2)])

    data_loader = bayes_dataset(trn_dataset)
    V, p_class, count_label, label_vocabulary, total_vocabulary = train(data_loader)
    json_object = {'V':V, 'p_class':p_class, 'count_label':count_label, 'label_vocabulary':label_vocabulary, 'total_vocabulary':total_vocabulary}
    with open('nbmodel.txt', 'w') as f:
        json.dump(json_object, f)


    data_loader = bayes_dataset(val_dataset)
    f1_score_holder = F1_score(4)

    # 2. train and save the nb model
    for content, label in data_loader:
        g_1, g_2 = label
        bayes_prob = [log(p) for p in p_class]
        for word in content:
            for i in range(4):
                bayes_prob[i] += log((label_vocabulary[i][word] + 1) / (count_label[i] + V))
        sys_1 = 0 if bayes_prob[0] > bayes_prob[1] else 1
        sys_2 = 2 if bayes_prob[2] > bayes_prob[3] else 3
        f1_score_holder.add_score(g_1, sys_1)
        f1_score_holder.add_score(g_2, sys_2)

    res, recall, precision, F1 = f1_score_holder.calculate_F1_score()
    print(res)
    print('---------')
    print(recall, precision, F1)