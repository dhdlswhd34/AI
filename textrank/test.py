import pandas as pd
from konlpy.tag import Mecab
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords

"""
fileName = "reranking_10000_re.csv"
csv_f = pd.read_csv(fileName, encoding='UTF-8')
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
srcSpacing = []
for i in range(0, len(csv_f.index)):
# for i in range(0, 20):
    passage = str(csv_f['Column1'][i]) + ' ' + str(csv_f['Column2'][i])
    objArrMecab = mecab.pos(passage)
    srcSpacing += [i for (i, j) in objArrMecab if ('SL' in j or 'SH' in j)]

count = Counter(srcSpacing)
nounlist = count.most_common()
data_df = pd.DataFrame(nounlist, columns=['word', 'count'])
data_df.to_excel('eng_word.xlsx', index=False)
# for v in nounlist:
#     print('Mecab.002.nouns : ', v)
"""

result = ''
fileName = "reranking_10000_re.csv"
csv_f = pd.read_csv(fileName, encoding='UTF-8')
mecab = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")
srcSpacing = []
nltk.download('stopfwords')
# for i in range(0, len(csv_f.index)):
for i in range(0, 100):
    passage = str(csv_f['Column1'][i]) + ' ' + str(csv_f['Column2'][i])

    passage = re.sub('[^a-zA-Z]', ' ', passage)

    passage = passage.lower().split()

    stops = set(stopwords.words('english'))
    no_stops = [word for word in passage if word not in stops]

    stemmer = nltk.stem.SnowballStemmer('english')
    stemmer_words = [stemmer.stem(word) for word in no_stops]
    result += ' '.join(stemmer_words)

print(result)

# count = Counter(srcSpacing)
# nounlist = count.most_common()
# data_df = pd.DataFrame(nounlist, columns=['word', 'count'])
# data_df.to_excel('eng_word.xlsx', index=False)

# for v in nounlist:
#     print('Mecab.002.nouns : ', v)

