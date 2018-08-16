#!/usr/bin/env python
# -*- encoding:utf-8 -*-
'''
Created on 2018年8月14日
使用朴素贝叶斯进行新闻分类
词频(TF:term Frequency)=某个词在文章中出现的次数/词的总数
逆文档频率(IDF:inverse Document Frequency)=log(语料库的文档总数/(包含该词的文档数+1))
关键词提取：TF－IDF=词频(TF)*逆文档频率(IDF)
示例：在<<中国蜜蜂养殖>>：假定该文长度为1000个词，中国、蜜蜂、养殖各出现20次，则三个词的词频(TF)为0.02
      搜索google发现，包含"的"字的网页共有250亿张，假定这就是中文网页总数，
      包含"中国"的网页共有62.3亿张，包含"蜜蜂"的网页共有0.484亿张，包含"养殖"的网页共有0.973亿张，
                包含该词的文档总数(亿)        IDF                              TF-IDF
   中国               62.3                log(250/62.3)=0.60               0.02*0.603=0.0121
   蜜蜂               0.484               log(250/0.484)=2.713             0.02*2.713=0.0543
   养殖               0.973               log(250/0.973)=2.41              0.02*2.41=0.0482

文本分析：
1. 分词
2. 去掉停用词(存在停用词库时)
3. 得到语料库
4. 生成词频向量
@author: guan.xianchun
'''
import pandas as pd
import jieba
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
from wordcloud import WordCloud
from gensim import corpora
import gensim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
label_mapping = {u"汽车": 1, u"财经": 2, u"科技": 3, u"健康": 4, u"体育":5, u"教育": 6, u"文化": 7, u"军事": 8, u"娱乐": 9, u"时尚": 0}

class NewClassify(object):
    def __init__(self):
        self.new_datas = None
    def load_datas(self):
        """
                加载样本数据
                数据源：http://www.sogou.com/labs/resource/ca.php
        """
        new_datas = pd.read_table('val.txt', names=['category', 'theme', 'URL', 'content'], encoding='utf-8', keep_default_na=False)
        self.new_datas = new_datas.dropna(subset=['category'])

    def get_stop_words(self):
        """
                获取停用词
        """
        stop_words = pd.read_csv("stopwords.txt", index_col=False, sep="\t", quoting=3, names=['stopword'], encoding='utf-8')
        return stop_words
    
    def cut_words(self):
        """
        分词：使用结吧分词器来分词
        """
        content = self.new_datas.content.values.tolist()
        contents = []
        for line in content:
            current_segment = jieba.lcut(line)
            if len(current_segment) > 1 and current_segment != '\r\n':  # 换行符
                contents.append(current_segment) 
        return contents
    
    def drop_stop_word(self):
        """
                去掉停用词
        """
        # 获取数据中所有的词
        contents = self.cut_words()
        self.extract_key_words(contents)
        # 获取停用词
        drop_words = self.get_stop_words()
        stopwords = drop_words.stopword.values.tolist()
        contents_clean = []
        all_words = []
        for line in contents:
            line_clean = []
            for word in line:
                if isinstance(word, list):
                    print '-->', word
                if word in stopwords or word == u'NaN':
                    continue
                line_clean.append(word)
                all_words.append(word)
            contents_clean.append(line_clean)
        df_contents_clean = pd.DataFrame({"contents_clean":contents_clean})
        df_all_words = pd.DataFrame({'all_words':all_words})
        return df_contents_clean, df_all_words
    
    def calc_word_TF(self, df_all_words):
        """
                计算词频(TF)
        """
        words_count = df_all_words.groupby(by=['all_words'])['all_words'].agg({'count':np.size})
        words_count = words_count.reset_index().sort_values(by=['count'], ascending=False)
        return words_count
    
    def draw_top_word(self, words_count):
        matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
        wordcloud = WordCloud(font_path="simhei.ttf", background_color="white", max_font_size=80)
        word_frequence = {x[0]:x[1] for x in words_count.head(100).values}
        wordcloud = wordcloud.fit_words(word_frequence)
        plt.imshow(wordcloud)
        plt.show()
    def extract_key_words(self, contents):
        """
                提取关键词方法
        """
        import jieba.analyse
        index = 2400
        print self.new_datas['content'][index]
        print " ".join(jieba.analyse.extract_tags("".join(contents[index]), topK=5, withWeight=False))
        
    def get_gensim_subjects(self, content_clean):
        # 生成字典
        dictionary = corpora.Dictionary(content_clean)
        # 可以将字典保存，方便以后使用
#         dictionary.save("news_dictory.dict")
        # 从文件中加载字典
#         corpora.Dictionary.load("news_dictory.dict")
        # 将标记化的文档转化为向量,对每个不同单词的出现次数进行了计数，并将单词转换为其编号，然后以稀疏向量的形式返回结果
        corpus = [dictionary.doc2bow(sentence) for sentence in content_clean]
        # 可以保存语料库，方便以后使用
        corpora.MmCorpus.serialize("news_corpus.mm", corpus)
        # 从文件加载语料库
        # corpora.MmCorpus.load("news_corpus.mm")
        # 建立LDA主题模型
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)
        # 打印topic编号为1的排名前5的词
        print lda.print_topic(1, topn=5)
        return lda
    
    def test_module_by_tf(self, words, train_result, test_words, test_result):
        vec = CountVectorizer(analyzer='word', max_features=4000, lowercase=True)
        vec.fit(words)
        classifier = MultinomialNB()
        classifier.fit(vec.transform(words), train_result)
        print classifier.score(vec.transform(test_words), test_result)
        
    def test_module_by_tfidf(self, words, train_result, test_words, test_result):
        # 使用TF-IDF
        classifier = MultinomialNB()
        vectorizer = TfidfVectorizer(analyzer='word', max_features=4000, lowercase=True)
        vectorizer.fit(words)
        classifier.fit(vectorizer.transform(words), train_result)
        print classifier.score(vectorizer.transform(test_words), test_result)
        
    def train_module(self, contents):
        df_train = pd.DataFrame({'contents_clean':contents, 'label':self.new_datas['category']})
        for item in df_train.label.unique():
            print item
        df_train['label'] = df_train['label'].map(label_mapping)
        x_train, x_test, y_train, y_test = train_test_split(df_train['contents_clean'].values, df_train['label'].values, random_state=1)
        words = []
        for i in range(len(x_train)):
            words.append(' '.join(x_train[i]))
        test_words = []
        for i in range(len(x_test)):
            test_words.append(' '.join(x_test[i]))
        self.test_module_by_tf(words, y_train, test_words, y_test)
        # 使用TF-IDF
        self.test_module_by_tfidf(words, y_train, test_words, y_test)

if __name__ == '__main__':
    new_classify = NewClassify()
    new_classify.load_datas()
    df_contents_clean, df_all_words = new_classify.drop_stop_word()
    new_classify.get_gensim_subjects(df_contents_clean.contents_clean.values.tolist())
    new_classify.train_module(df_contents_clean.contents_clean.values.tolist())
