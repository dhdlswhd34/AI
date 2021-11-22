import pandas as pd 
from krwordrank.word import KRWordRank
from krwordrank.hangle import normalize

# -*- coding: utf-8 -*-
import pandas as pd 
import time

# 키워드 추출
from keybert import KeyBERT 
from collections import Counter
# KR-WordRank
from krwordrank.word import KRWordRank
from krwordrank.hangle import normalize


# 문장 전처리
from konlpy.tag import Mecab
from pykospacing import Spacing



def lib_fn_remove_special(fileSource):
    bufSource = fileSource
    # ------------------------------------------------------------
    # 개행문자/Tab문자 제거
    # ------------------------------------------------------------
    bufSource = bufSource.replace("\n", " ")
    bufSource = bufSource.replace("\t", " ")
    # ------------------------------------------------------------
    # 특수문자 제거
    # ------------------------------------------------------------
    bufSource = bufSource.replace("\\", " ")
    bufSource = bufSource.replace("\"", " ")
    bufSource = bufSource.replace('"', '')

    specialChars = "!#$%^&*()?~.,/-_"
    for specialChr in specialChars:
        bufSource = bufSource.replace(specialChr, ' ')
    bufSource = bufSource.strip()

    # bufSource = bufSource.replace("  ", " ")

    bufSource = bufSource.replace(" ", "")
    
    return bufSource

def lib_fn_remove_digits(fileSource):
    # ------------------------------------------------------------
    # 정규 표현식으로 전화번호를 *로 치환
    # ------------------------------------------------------------
    #import re
    #text = """\
    #010-1234-5678 Kim
    #011-1234-5678 Lee
    #016-1234-5678 Han
    #"""
    ## 정규 표현식 사용 치환
    #text_mod = re.sub('^[0-9]{3}-[0-9]{4}-[0-9]{4}',"***-****-****",text) 
    #print (text_mod)    
    # ------------------------------------------------------------
    # 숫자 제거 ( 개인정보:전화번호, 주민번호 )
    # ------------------------------------------------------------
    bufSource = ''.join([i for i in fileSource if not i.isdigit()])

    return bufSource

fileName = "reranking_10000_re.csv"
# csv_f = pd.read_csv("조달검색행태조사응답.xlsx", encoding='utf-8', error_bad_lines=False)
csv_f = pd.read_csv(fileName, encoding='UTF-8')
csv_w = pd.DataFrame()
kw_model = KeyBERT(model='paraphrase-multilingual-mpnet-base-v2')

print(len(csv_f.index))
print(len(csv_f.columns))
# print(csv_f['Column1'][2])
# print('------------------')
# print(csv_f['Column2'][2])


min_count = 5   # 단어의 최소 출현 빈도수 (그래프 생성 시)
max_length = 10 # 단어의 최대 길이
wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)

beta = 0.85    # PageRank의 decaying factor beta
max_iter = 10
texts = [csv_f['Column2'][5]]


texts = [normalize(text, english=False, number=False) for text in texts]
keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)
temp = ''
for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:30]:
    temp =  temp + ' ' + word
    print('%8s:\t%.4f' % (word, r))

"""
for k in range(0, 10):
    temp_pd = []
    doc = csv_f['Column1'][k] + ' ' + csv_f['Column2'][k]
    doc = lib_fn_remove_special(doc)
    doc = lib_fn_remove_digits(doc)
    spacing = Spacing()
    doc = spacing(doc)
    temp_pd.append([doc])

    objArrKeywords = kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 3), top_n=20)

    #print ('KeyBert.002.objArrKeywords : %s' %(objArrKeywords))
    # ------------------------------------------------------------
    # KeyBERT Result
    # ------------------------------------------------------------
    temp = ''
    for j, (keyword, score) in enumerate(objArrKeywords):
        temp = temp + ' ' + keyword
        print('KeyBert.003 : ', keyword, score)

    mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
    #objArrOkts = mecab.morphs(srcSpacing)    # 형태소
    objArrOkts = mecab.nouns(temp)    # 명사만 추출
    #objArrOkts = mecab.phrases(srcSpacing)    # 어절만 추출

    temp = ' '.join([i for i in objArrOkts])
    #print ('Mecab.001.objArrOkts : %s' %(bufSource))

    # 한글자 명사는 제외 처리
    for i, v in enumerate(objArrOkts):
        if len(v) < 2:
            objArrOkts.pop(i)

    # 명사 빈도 순으로 추출
    count = Counter(objArrOkts)
    nounlist = count.most_common(20)
    for v in nounlist:
        print('Mecab.002.nouns : ', v)
        temp_pd.append(v)
    csv_w[str(k)] = pd.Series(temp_pd)
    # csv_w.to_csv("keyword_one.csv", mode='a', encoding='utf-8-sig')
    # csv_w.to_csv("keyword_two.csv", mode='a', en
    # 
    # 
    # coding='utf-8-sig')
    # print(csv_w)
csv_w.to_csv("keyword_after_mecab.csv", mode='a', encoding='utf-8-sig')
"""
