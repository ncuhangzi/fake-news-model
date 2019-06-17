import pandas as pd
import numpy as np
import csv
import keras
import jieba.posseg as pseg

# 分詞
def jieba_tokenizer(text):
    words = pseg.cut(text)
    return ' '.join([word for word, flag in words if flag != 'x']) ## 去標點 : flag!='x'

MAX_NUM_WORDS = 10000
tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NUM_WORDS)

# 一個標題最長有幾個詞彙
MAX_SEQUENCE_LENGTH = 20


#########################
#                       #
#        predict        #
#                       #
#########################
test = pd.read_csv('./test.csv')


# 以下步驟分別對新聞標題 A、B　進行文本斷詞 / Word Segmentation
test['title1_tokenized'] = test['title1_zh'].apply(jieba_tokenizer)
test['title2_tokenized'] = test['title2_zh'].astype(str).apply(jieba_tokenizer)
print('\n\nx1 分詞結果:\n',test['title1_tokenized'])
print('\n\nx2 分詞結果:\n',test['title2_tokenized'])


# 將詞彙序列轉為索引數字的序列
corpus_x1 = test['title1_tokenized']
corpus_x2 = test['title2_tokenized']
corpus = pd.concat([corpus_x1, corpus_x2])
pd.DataFrame(corpus.iloc[:5], columns=['title'])
tokenizer.fit_on_texts(corpus)

x1_test = tokenizer.texts_to_sequences(test.title1_tokenized)
x2_test = tokenizer.texts_to_sequences(test.title2_tokenized)
print('\n\nx1 test:\n',x1_test)
print('\n\nx2 test:\n',x2_test)

# 為數字序列加入 zero padding
x1_test = keras.preprocessing.sequence.pad_sequences(x1_test, maxlen=MAX_SEQUENCE_LENGTH)
x2_test = keras.preprocessing.sequence.pad_sequences(x2_test, maxlen=MAX_SEQUENCE_LENGTH)    
print('\n\nx1 test:\n',x1_test)
print('\n\nx2 test:\n',x2_test)


# 利用已訓練的模型做預測
model = keras.models.load_model('Model.h5')
predictions = model.predict([x1_test, x2_test])
print('\n\n預測結果\n\n',predictions)


# 定義每一個分類對應到的索引數字
label_to_index = { 'unrelated': 0, 'agreed': 1, 'disagreed': 2}
index_to_label = {v: k for k, v in label_to_index.items()}

test['Category'] = [index_to_label[idx] for idx in np.argmax(predictions, axis=1)]
submission = test.loc[:, ['Category']].reset_index()
submission.columns = ['Id', 'Category']
outcome = submission.values[0][1]
print(outcome)