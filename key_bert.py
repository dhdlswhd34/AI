# -*- coding: utf-8 -*-
from re import S
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
 
    specialChars = "!#$%^&*()?~.,/-_○』『☎"
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

if __name__ == '__main__':
    USE_MECAB = 1
    USE_KEYBERT = 1
    USE_KRWR = 0
    USE_Spacing = 0
    keylen = 2

    fileName = "reranking_10000_re.csv"

    #  keybert 사전 학습 모델
    # ------------------------------------------------------------
    # distiluse-base-multilingual-cased-v1 : 다국어 범용 문장 인코더 의 다국어 지식 증류 버전 . 15개 언어 지원: 아랍어, 중국어, 네덜란드어, 영어, 프랑스어, 독일어, 이탈리아어, 한국어, 폴란드어, 포르투갈어, 러시아어, 스페인어, 터키어.
    # distiluse-base-multilingual-cased-v2 : 다국어 범용 문장 인코더 의 다국어 지식 증류 버전 . 이 버전은 50개 이상의 언어를 지원하지만 v1 모델보다 성능이 약간 떨어집니다.
    # paraphrase-multilingual-MiniLM-L12-v2 - paraphrase-MiniLM-L12-v2 의 다국어 버전으로 , 50개 이상의 언어에 대한 병렬 데이터로 훈련되었습니다.
    # paraphrase-multilingual-mpnet-base-v2 - paraphrase-mpnet-base-v2 의 다국어 버전으로 , 50개 이상의 언어에 대한 병렬 데이터로 훈련되었습니다.
    # ------------------------------------------------------------
    kw_model = KeyBERT(model='distiluse-base-multilingual-cased-v1')
    # kw_model = KeyBERT(model='distiluse-base-multilingual-cased-v2')
    # kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')
    # kw_model = KeyBERT(model='paraphrase-multilingual-mpnet-base-v2')

    csv_f = pd.read_csv(fileName, encoding='UTF-8')
    csv_w = pd.DataFrame()
    # print(len(csv_f.index))
    # print(len(csv_f.columns))

    temp = []
    
    
    doc = str(csv_f['Column1'][0]) + ' ' + str(csv_f['Column2'][0])

    doc = lib_fn_remove_digits(doc)
    doc = lib_fn_remove_special(doc)
    print(doc)

    if USE_Spacing == 1:
        spacing = Spacing()
        doc = spacing(doc)
        # print(doc)

    if USE_MECAB == 1:
        mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
        
        # objArrOkts = mecab.morphs9(텍스트)    # 형태소
        # objArrOkts = mecab.nouns(텍스트)    # 명사만 추출
        # # objArrOkts = mecab.morphs(doc)    # 어절만 추출

        objArrOkts = mecab.pos(doc)

        L = 0
        while(L < 2):
            for i, (v, t) in enumerate(objArrOkts):
                if len(v) < 2:
                    objArrOkts.pop(i)
                    L = len(v)
                    break
                else:
                    L = len(v)
                    
        print(objArrOkts)

        srcSpacing = ' '.join([i for (i, j) in objArrOkts if ('NN' in j or 'XR' in j or 'VA' in j or 'VV' in j)])

        # srcSpacing = ' '.join([i for i in objArrOkts])
        print(srcSpacing)
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
        
    if USE_KRWR == 1:
        min_count = 5   # 단어의 최소 출현 빈도수 (그래프 생성 시)
        max_length = 10 # 단어의 최대 길이
        wordrank_extractor = KRWordRank(min_count=min_count, max_length=max_length)

        beta = 0.85    # PageRank의 decaying factor beta
        max_iter = 10
        
        texts = [doc]
        texts = [normalize(text, english=False, number=False) for text in texts]
        keywords, rank, graph = wordrank_extractor.extract(texts, beta, max_iter)

        for word, r in sorted(keywords.items(), key=lambda x:x[1], reverse=True)[:30]:
            print('%8s:\t%.4f' % (word, r))

    if USE_KEYBERT == 1:
        # kw_model = KeyBERT()
        # keyphrase_ngram_range=(1, 1) : 한 단어 기준
        # print(srcSpacing)
        # objArrKeywords = kw_model.extract_keywords(srcSpacing, keyphrase_ngram_range=(1, 3), top_n=20)
        # objArrKeywords = kw_model.extract_keywords([doc, keyphrase_ngram_range=(1, 3), top_n=20)
        objArrKeywords = kw_model.extract_keywords(srcSpacing, keyphrase_ngram_range=(1, 3), top_n=20)
        # objArrKeywords = kw_model.extract_keywords(srcSpacing, keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.5)
        # objArrKeywords = kw_model.extract_keywords(srcSpacing, keyphrase_ngram_range=(1, 2))    # 2개 단어추출
        # objArrKeywords = kw_model.extract_keywords(srcSpacing, keyphrase_ngram_range=(3, 3))    # 3개 단어추출
        # objArrKeywords = kw_model.extract_keywords(srcSpacing)    # top_n=5 : 5 개 추출 ( default )

        # ------------------------------------------------------------
        # KeyBERT Result
        # ------------------------------------------------------------
        # temp.append([srcSpacing])
        print(objArrKeywords)
        for (keyword, score) in enumerate(objArrKeywords):
            print('KeyBert.003 : ', keyword, score)
            # temp.append([keyword, score])

        # temp.append([time.time() - start])
        # csv_w['1'] = temp
        # print(csv_w)
