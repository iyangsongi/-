# -*- coding: utf-8 -*-

# auxiliary function

#@author:yangsong
import os
import jieba
import re
import math
import time
from gensim.models import Word2Vec
import numpy as np
from numpy import linalg as la
import logging
from collections import defaultdict
from functools import wraps
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
web_root = os.path.abspath('.')
path = os.path.join(web_root, 'model/mini.model')

LTP_DATA_DIR = os.path.join(web_root, 'ltp')  # ltp模型目录的路径
pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`

import sys

from pyltp import Postagger
from pyltp import Parser
from pyltp import NamedEntityRecognizer
from pyltp import Segmentor
import jieba.posseg as pseg
import functools
sys.setrecursionlimit(100000)

def split(sentence: str) ->'List(str)':
    """
    split a paragraph into sentences.
    """
    # merge two quotations
    sentence = sentence.replace('”“', '，')
    N = len(sentence)

    # clean "“”"
    stack = []
    tmp = list(sentence)
    i = 0
    while i < N:
        c = tmp[i]
        if c == '“':
            stack.append(i)
        elif c == '”':
            l = stack.pop(0)
            if not ((i + 1 < N and tmp[i+1] in ['。', '！', '？']) or tmp[i-1] in ['。', '！', '？', '，']):
                tmp[l] = '"'
                tmp[i] = '"'
        i += 1
    sentence = ''.join(tmp)

    res = []
    stack = []
    l = 0
    for i, char in enumerate(sentence):
        if char == '“':
            stack.append(i)
            continue
        if char == "。" or char == "？" or char == "！":
            if not stack:
                res.append(sentence[l:i])
                l = i+1
        if char == "”":
            stack.pop()
            # if sentence[i-1] == '。': # direct quotation
            #    res.append(sentence[l:i+1])
            if i + 1 < N and sentence[i+1] == '。': # direct quotation
                res.append(sentence[l:i+1])
                l = i + 1
    if l < i+1:
        res.append(sentence[l:i+1])
    return res

# LOG_FORMAT = "%(asctime)s - %(funcName)s - %(message)s"
# logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT)
def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print("Total time running %s: %s seconds" %
              (function.__name__, str(t1 - t0))
              )
        return result

    return function_timer

# main Model
class Model:
    def __init__(self):
        self.name_says = defaultdict(list) #定义成全局变量有可能从sentence_process()中写入，也可能从single_sentence()写入
        self.model = Word2Vec.load(path)
        self.word_total_count = self.model.corpus_total_words
        self.word_dict = self.model.wv.vocab
        self.dim = 256
        
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(pos_model_path)  # 加载模型
        
        self.say_sim = ['诊断', '交代', '说', '说道', '指出','报道','报道说','称', '警告','所说', '告诉', '声称', '表示', '时说', '地说', '却说', '问道', '写道', '答道', '感叹', '谈到', '说出', '认为', '提到', '强调', '宣称', '表明', '明确指出', '所言', '所述', '所称', '所指', '常说', '断言', '名言', '告知', '询问', '知道', '得知', '质问', '问', '告诫', '坚称', '辩称', '否认', '还称', '指责', '透露', '坦言', '表达', '中说', '中称', '他称', '地问', '地称', '地用', '地指', '脱口而出', '一脸', '直说', '说好', '反问', '责怪', '放过', '慨叹', '问起', '喊道', '写到', '如是说', '何况', '答', '叹道', '岂能', '感慨', '叹', '赞叹', '叹息', '自叹', '自言', '谈及', '谈起', '谈论', '特别强调', '提及', '坦白', '相信', '看来', '觉得', '并不认为', '确信', '提过', '引用', '详细描述', '详述', '重申', '阐述', '阐释', '承认', '说明', '证实', '揭示', '自述', '直言', '深信', '断定', '获知', '知悉', '得悉', '透漏', '追问', '明白', '知晓', '发觉', '察觉到', '察觉', '怒斥', '斥责', '痛斥', '指摘', '回答', '请问', '坚信', '一再强调', '矢口否认', '反指', '坦承', '指证', '供称', '驳斥', '反驳', '指控', '澄清', '谴责', '批评', '抨击', '严厉批评', '诋毁', '责难', '忍不住', '大骂', '痛骂', '问及', '阐明']
        self.valid_sentence = []
        
        self.parser = Parser()
        self.parser.load(par_model_path)

        self.segmentor = Segmentor()
        self.segmentor.load(cws_model_path) 

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(ner_model_path)



    # @functools.lru_cache()
    # @fn_timer
    def get_count(self, word):
        """
        O(1)
        """
        # word_count = 0 #定义默认值
        vector = np.zeros(1) #定义默认值

        if word in self.word_dict:
            wf = self.word_dict[word].count
            wv = self.model.wv[word]
        else:
            wf = 1
            wv = np.zeros(self.dim)
        return wf / self.word_total_count, wv



    #获取句子向量
    #TODO: 计算P(w)的过程可以优化
    def sentence_embedding(self, sentence):
        # 按照论文算法Vs=1/|s|*∑a/(a+p(w))*Vw
        sentences = self.process_content(sentence).replace(' ','')
        a = 1e-3 #0.001

        words = self.pyltp_cut(sentences)
        sum_vector = np.zeros(self.dim)
        for i, w in enumerate(words):
            wf, wv = self.get_count(w)
            sum_vector += a / (a + wf) * wv
        return sum_vector / (i+1)

    # 欧式距离
    def euclidSimilar(self, inA, inB):
        return 1.0/(1.0+la.norm(inA-inB))

    # 皮尔逊相关系数
    def pearsonSimilar(self, inA, inB):
        if len(inA) != len(inB):
            return 0.0
        if len(inA) < 3:
            return 1.0
        return 0.5+0.5*np.corrcoef(inA, inB, rowvar=0)[0][1]

    # 余弦相似度
    def cosSimilar(self, inA, inB):
        inA = np.mat(inA)
        inB = np.mat(inB)
        num = float(inA * inB.T)
        denom = la.norm(inA) * la.norm(inB)
        return 0.5+0.5*(num/denom)

    # 句子依存分析
    def parsing(self, sentence):
        words = self.pyltp_cut(sentence)  # pyltp分词
        postags = self.postagger.postag(words)  # 词性标注
        arcs = self.parser.parse(words, postags)  # 句法分析
        return arcs

    # 命名实体
    # @functools.lru_cache()
    def get_name_entity(self, strs):
        sentence = ''.join(strs)
        words = self.pyltp_cut(sentence)#pyltp分词更合理
        postags = self.postagger.postag(words)  # 词性标注
        netags = self.recognizer.recognize(words, postags)  # 命名实体识别
        return netags

    # 输入单个段落句子数组
    def valid_sentences_(self, sentences, res):
        expect = 0.76  
        
        tmp = ""  # 储存前一个言论
        while sentences:
            curr = sentences.pop(0)
            if curr[0] == '“':  # 当前句子或为 “言论在发言人前的直接引用”。
                print(curr)
                people = re.search('”(.+)“|”(.+)', curr)  # 提取发言人所在句段
                if people:
                    people = [i for i in people.groups() if i][0]
                elif res:
                    res[-1][1] += '。' + curr
                    continue
                else:
                    continue

                saying = curr.replace(people, '')  # 剩余部分被假设为“言论”
                if res and self.judge_pronoun(people):
                    res[-1][1] += '。' + saying
                else:
                    comb = self.single_sentence(people)
                    if comb:
                        saying += comb[1] if comb[1] else ''
                        res.append([comb[0], saying])
                continue

            # 尝试提取新闻 发言人，言论内容
            combi = self.single_sentence(curr)
            
            # 无发言人： 当前句子属于上一个发言人的言论 或 不属于言论 
            if not combi:
                if res and tmp and self.compare_sentence(tmp, curr) > expect:  #基于句子相似度判断
                    print('{} - {} : {}'.format(tmp, curr, self.compare_sentence(tmp, curr)))
                    res[-1][1] += '。' + curr
                    tmp = curr
                continue

            # 有发言人： 提取 发言人 和 言论。
            name, saying = combi
            if res and self.judge_pronoun(curr) and saying:
                res[-1][1] += '。' + saying
            elif saying:
                res.append([name, saying])
            tmp = saying
        return res


    @functools.lru_cache()
    def single_sentence(self, sentence, just_name=False, ws=False):
        sentence = '，'.join([x for x in sentence.split('，') if x])
        cuts = list(self.pyltp_cut(sentence))  # pyltp分词更合理
        # 判断是否有‘说’相关词：
        mixed = [word for word in cuts if word in self.say_sim]
        if not mixed : return False

        ne = self.get_name_entity(tuple(sentence)) #命名实体
        wp = self.parsing(sentence) #依存分析
        wp_relation = [w.relation for w in wp]
        postags = list(self.postagger.postag(cuts))
        name = ''

        stack = [] 
        for k, v in enumerate(wp):
            # save the most recent Noun
            if postags[k] in ['nh', 'ni', 'ns']:
                stack.append(cuts[k])
            
            if v.relation=='SBV' and (cuts[v.head-1] in mixed) : #确定第一个主谓句
                name = self.get_name(cuts[k], cuts[v.head-1], cuts, wp_relation,ne)

                if just_name == True: return name #仅返回名字
                says = self.get_says(cuts, wp_relation, [i.head for i in wp], v.head)
                if not says:
                    quotations = re.findall(r'“(.+?)”', sentence)
                    if quotations: says = quotations[-1]
                return name, says
            # 若找到‘：’后面必定为言论。
            if cuts[k] == '：': 
                name = stack.pop()
                says = ''.join(cuts[k+1:])
                return name, says
        return False

    # 输入主语第一个词语、谓语、词语数组、词性数组，查找完整主语
    def get_name(self, name, predic, words, property, ne):
        index = words.index(name)
        cut_property = property[index + 1:] #截取到name后第一个词语
        pre=words[:index]#前半部分
        pos=words[index+1:]#后半部分
        #向前拼接主语的定语
        while pre:
            w = pre.pop(-1)
            w_index = words.index(w)

            if property[w_index] == 'ADV': continue
            if property[w_index] in ['WP', 'ATT', 'SVB'] and (w not in ['，','。','、','）','（']):
                name = w + name
            else:
                pre = False

        while pos:
            w = pos.pop(0)
            p = cut_property.pop(0)
            if p in ['WP', 'LAD', 'COO', 'RAD'] and w != predic and (w not in ['，', '。', '、', '）', '（']):
                name = name + w # 向后拼接
            else: #中断拼接直接返回
                return name
        return name

    # 获取谓语之后的言论
    def get_says(self, sentence, property, heads, pos):
        # word = sentence.pop(0) #谓语
        if '：' in sentence:
            return ''.join(sentence[sentence.index('：')+1:])
        while pos < len(sentence):
            w = sentence[pos]
            p = property[pos]
            h = heads[pos]
            # 谓语尚未结束
            if p in ['DBL', 'CMP', 'RAD']:
                pos += 1
                continue
            # 定语
            if p == 'ATT' and property[h-1] != 'SBV':
                pos = h
                continue
            # 宾语
            if p == 'VOB':
                pos += 1
                continue
            # if p in ['ATT', 'VOB', 'DBL', 'CMP']:  # 遇到此性质代表谓语未结束，continue
            #    continue
            else:
                if w == '，':
                    return ''.join(sentence[pos+1:])
                else:
                    return ''.join(sentence[pos:])


    #解析处理语句并返回给接口
    def sentence_process(self,sentence):
        # 文章 -->清除空行
        # 文章 -->句号分割：如果句号分割A.B, 若B存在‘说’，对B独立解析，否则判断A | B是否相似，确定A是否抛弃B句。
        # 句子 -->确定主谓宾: 依存分析、命名实体识别 -->首先要找到宾语，然后确定宾语是否与说近似，若存在多个与‘说’近似，确定第一个为陈述。在说前找命名实体，说后面到本句结尾为宾语
        # 命名实体 -->通过命名实体识别，若S - NE, NE = S - NE。若B - NE / I - NE / E - NE，NE = B - NE + I - NE + E - NE

        self.name_says = defaultdict(list)
        sentence = sentence.replace('\r\n','\n')
        sections = sentence.split('\n') #首先切割成段落
        sections = [s for s in sections if s.strip()]
        valids=''
        
        res = []
        for sec in sections: #段落
            sentence_list = split(sec)
            sentence_list = [s.strip() for s in sentence_list if s.strip()]
            self.cut_sententce_for_name = [s for s in sentence_list if s]
            # valids = self.valid_sentences(sentence_list)
            res += self.valid_sentences_(sentence_list, [])
        if res:
            self.name_says = defaultdict()
            for name, saying in res:
                if name and saying:
                    self.name_says[name] = self.name_says.get(name, '') + saying + ' | '
        return self.name_says

    # 判断是否为代词结构句子“他认为...，他表示....”
    #@fn_timer
    def judge_pronoun(self, sentence):
        subsentence = re.search('(.+)“|”(.+)', sentence)
        if subsentence:
            sentence = subsentence.group(1)
        cuts = list(self.pyltp_cut(sentence))  # 确定分词
        wp = self.parsing(sentence)  # 依存分析
        postags = list(self.postagger.postag(cuts))
        for k, v in enumerate(wp):
            if v.relation == 'SBV' and postags[k] == 'r':  # 确定第一个主谓句
                return True
        return False


    #句子比对皮尔逊系数
    def compare_sentence(self, inA, inB):
        inC = self.sentence_embedding(inA)
        inD = self.sentence_embedding(inB)
        return self.pearsonSimilar(inC, inD) #皮尔逊
        # print(self.euclidSimilar(inC,inD))
        # print(self.pearsonSimilar(inC,inD))
        # print(self.cosSimilar(inC,inD))
        # print('------------------------')

    #pyltp中文分词
    def pyltp_cut(self, sentence):
        # segmentor = Segmentor()  # 初始化实例
        # segmentor.load(cws_model_path)  # 加载模型
        words = self.segmentor.segment(sentence)  # 分
        # segmentor.release()  # 释放模型
        return words

    #结巴词性标注
    def jieba_pseg(self,sentence):
        return pseg.cut(sentence)


    def document_frequency(self,word,document):
        if sum(1 for n in document if word in n)==0:
            print(word)
            print(type(document))
            print(len(document))
            print(document[0])
        return sum(1 for n in document if word in n)

    def idf(self, word, content, document):
        """Gets the inversed document frequency"""
        return math.log10(len(content) / self.document_frequency(word,document))

    def tf(self,word, document):
        """
        Gets the term frequemcy of a @word in a @document.
        """
        words = document.split()

        return sum(1 for w in words if w == word)


    def process_content(self,content):
        content=re.sub('[+——() ? 【】“”！，：。？、~@#￥%……&*（）《 》]+', '', content)
        content=' '.join(jieba.cut(content))
        return content
    
    def release_all(self):
        self.segmentor.release()
        self.recognizer.release()
        self.parser.release()
        self.postagger.release()




