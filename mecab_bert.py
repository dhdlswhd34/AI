from konlpy.tag import Mecab


def whitespace_tokenize(data):
    data = data.strip()    # 문자열의 맨앞, 맨끝 공백 지움
    if not data:
        return []
    tokens = data.split()  # 문자열을 스페이스,탭,엔터 단위로 분리하여 배열에 집어넣음
    return tokens

temp = '또한 일반적인 위키에서 텍스트는 단순화된 마크업 언어(위키 마크업)을 이용하여 작성되며, 리치 텍스트 에디터의 도움을 받아 편집하기도 한다.[1] 위키는 지식경영이나 기록 등 다양한 용도로 이용된다. 공동체용 웹사이트나 조직 내 인트라넷에 쓰이기도 한다. 그러나 주로 개인적인 용도로 이용되는 위키도 있는데, 이를 개인 위키라고 한다.'

tokenizer = Mecab(dicpath=r"C:\mecab\mecab-ko-dic")

output_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
for wst in whitespace_tokenize(temp):   # wst : 공백,탭,엔터 기준 문자열 하나
    count = 0
    for token in tokenizer.morphs(wst):       # token : wst를 형태소 분석한 토큰 하나
        tk = token

        if count > 0:
            tk = "##" + tk
            if tk in output_tokens:   # 토큰이 중복되면 저장하지 않음
                continue
            output_tokens.append(tk)
        else:  # count==0
            count += 1
            if tk in output_tokens:   # 토큰이 중복되면 저장하지 않음
                continue
            output_tokens.append(tk)  # 맨 처음 token만 앞에 ##을 붙이지 않음

print(output_tokens)

txt_file = open("vocab.txt", 'w+',encoding='utf-8')  # 텍스트 파일을 쓰기 모드로 생성

for token in output_tokens:
    txt_file.write(token + '\n')

txt_file.close()
