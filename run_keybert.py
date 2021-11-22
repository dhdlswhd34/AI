import pandas as pd
from keybert import KeyBERT
from konlpy.tag import Mecab
import time

# +-----------------------+
# | MECAB FUNC            |
# +-----------------------+
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
def use_mecab(passage):
    # objArrMecab = mecab.pos(passage)
    objArrMecab = mecab.nouns(passage)

    # DELETE LEN == 1 WORD
    L = 0
    while(L < 2):
        for i, v in enumerate(objArrMecab):
            if len(v) < 2:
                objArrMecab.pop(i)
                L = len(v)
                break
            else:
                L = len(v)

    # srcSpacing = ' '.join([i for (i, j) in objArrMecab if ('NN' in j or 'XR' in j or 'VA' in j or 'VV' in j)])
    nouns_passage = ' '.join([i for i in objArrMecab])

    return nouns_passage

keyword_result = []


# +-------------------------------------+
# | KEYBERT MODEL DEFINE  
# +-------------------------------------+
start = time.time()
kw_model = KeyBERT(model="distiluse-base-multilingual-cased-v1")
print(time.time() - start)


# +-------------------------------------+
# | READ FILE             
# +-------------------------------------+
filename = "reranking_10000_re.csv"
csv_f = pd.read_csv(filename, encoding='UTF-8')


# +-------------------------------------+
# | START                 
# +-------------------------------------+
start = time.time()
# for i in range(0, len(csv_f.index)):
for i in range(0, 10):
    passage = str(csv_f['Column1'][i]) + ' ' + str(csv_f['Column2'][i])

    passage = use_mecab(passage)

    # KEY EXTRACTION
    keyword_result.append(kw_model.extract_keywords(passage, keyphrase_ngram_range=(1, 3), top_n=5))
print(time.time() - start)


# +-------------------------------------+
# | RESULT                
# +-------------------------------------+
data_df = pd.DataFrame(keyword_result)
data_df.to_excel('keyword_result.xlsx')

