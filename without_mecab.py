import pandas as pd
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)

temp = '또한 일반적인 위키에서 텍스트는 단순화된 마크업 언어(위키 마크업)을 이용하여 작성되며, 리치 텍스트 에디터의 도움을 받아 편집하기도 한다.[1] 위키는 지식경영이나 기록 등 다양한 용도로 이용된다. 공동체용 웹사이트나 조직 내 인트라넷에 쓰이기도 한다. 그러나 주로 개인적인 용도로 이용되는 위키도 있는데, 이를 개인 위키라고 한다.'

temp = '[CLS] ' + str(temp) + ' [SEP]'

result = tokenizer.tokenize(temp)

print(result)

MAX_LEN = 128
 
input_ids = [tokenizer.convert_tokens_to_ids(result)]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
print(input_ids)
# input_ids

# # BERT의 단어 집합을 vocabulary.txt에 저장
# with open('vocabulary.txt', 'w') as f:
#     for token in tokenizer.vocab.keys():
#         f.write(token + '\n')
