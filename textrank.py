# -*- coding: utf-8 -*-
import pandas as pd 
import time
# 문장 전처리
from konlpy.tag import Mecab, Komoran
from pykospacing import Spacing
from collections import Counter

def mecab_tokenizer(doc):
    mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
    # objArrOkts = mecab.nouns(doc)    # 명사만 추출

    # srcSpacing = ' '.join([i for i in objArrOkts])
    
    # # 한글자 명사는 제외 처리
    # for i, v in enumerate(objArrOkts):
    #     if len(v) < 2:
    #         objArrOkts.pop(i)

    # # 명사 빈도 순으로 추출
    # count = Counter(objArrOkts)
    # nounlist = count.most_common(20)
    # for v in nounlist:
    #     print('Mecab.002.nouns : ', v)
    srcSpacing = []
    objArrOkts = mecab.pos(doc)
    srcSpacing = ' '.join([i for (i, j) in objArrOkts if ('NN' in j or 'XR' in j or 'VA' in j or 'VV' in j)])
    # objArrOkts = [w for w in objArrOkts if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)]

    # print(srcSpacing)
    return srcSpacing

def komoran_tokenizer(sent):
    komoran = Komoran()
    words = komoran.pos(sent, join=True) 
    words = [w for w in words if ('/NN' in w or '/XR' in w or '/VA' in w or '/VV' in w)] 
    print(words)
    return words


def lib_fn_remove_special(fileSource):
    bufSource = fileSource
    # ------------------------------------------------------------
    # 개행문자/Tab문자 제거
    # ------------------------------------------------------------
    # bufSource = bufSource.replace("\n", " ")
    bufSource = bufSource.replace("\t", " ")
    # ------------------------------------------------------------
    # 특수문자 제거
    # ------------------------------------------------------------
    bufSource = bufSource.replace("\\", " ")
    bufSource = bufSource.replace("\"", " ")
    bufSource = bufSource.replace('"', '')

    specialChars = "!#$%^&*()?~.,/-_○』『"
    for specialChr in specialChars:
        bufSource = bufSource.replace(specialChr, ' ')
    bufSource = bufSource.strip()

    bufSource = bufSource.replace("  ", " ")

    # bufSource = bufSource.replace(" ", "")

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


# from krwordrank.word import KRWordRank

# min_count = 5   # 단어의 최소 출현 빈도수 (그래프 생성 시)
# max_length = 10 # 단어의 최대 길이
# wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)


# beta = 0.85    # PageRank의 decaying factor beta
# max_iter = 10
# texts = ['예시 문장 입니다', '여러 문장의 list of str 입니다', ... ]
# keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)


# for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:30]:
#         print('%8s:\t%.4f' % (word, r))



from krwordrank.sentence import summarize_with_sentences

start = time.time()
texts = csv_f['Column1'][3] + csv_f['Column2'][3] 
texts = lib_fn_remove_special(texts)
texts = lib_fn_remove_digits(texts)
# spacing = Spacing()
# texts = spacing(texts)

texts = texts.split('\n')
# print(texts)


for value in texts:
    mecab_tokenizer(value)

penalty = lambda x: 0 if (25 <= len(x) <= 80) else 1

stopwords = {'입찰', '조달청', '합니다', '이용', '낙찰', 'www', 'http'}

keywords, sents = summarize_with_sentences(
    texts,
    penalty=penalty,
    stopwords=stopwords,
    diversity=0.5,
    num_keywords=100,
    num_keysents=10,
    verbose=False
)
print(keywords)
# print(sents)
print(time.time() - start)
print('-------------------------------------------------------')


from textrank import KeywordSummarizer
start = time.time()
summarizer = KeywordSummarizer(tokenize=mecab_tokenizer, min_count=2, min_cooccurrence=1)
print(summarizer.summarize(texts))
print(time.time() - start)