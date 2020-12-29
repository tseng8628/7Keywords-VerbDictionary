import os
import time
import pandas as pd
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from collections import defaultdict
import csv
from itertools import zip_longest
import jieba

tStart = time.time()    # 計時開始

file_path = '/Users/tsengyiwen/PycharmProjects/Python/Lab/7Keywords_VerbDic'
os.chdir(file_path)

keyword_list = ['選擇權', '期貨', '台指期', '賣權', '買權', '多單', '空單']

# ------------------------- Load Content ----------------------------
df = pd.read_csv('!Total_Result 2.csv', names=['ID', 'Title', 'Time', 'Content'])
content_list = list(df['Content'][:2])      # 目前抓1000筆來玩

# ---------------------------- Load model without CPU ----------------------------------
print('[INFO]正在載入模型...')
tLoadStart = time.time()
ws = WS("./data", disable_cuda=False)           # (WS) 斷詞
pos = POS("./data", disable_cuda=False)         # (POS) 詞性標注
ner = NER("./data", disable_cuda=False)         # (NER) 實體辨識
tLoadEnd = time.time()
print('[INFO]模型載入完成。')
print("載入模型 總共花費 %f sec\n" % (tLoadEnd - tLoadStart))

# ----------------------- 建立強制分詞字典 with ckip -----------------------------
word_to_weight = {
    "選擇權": 1, "期貨": 1, "台指期": 1, "賣權": 1, "買權": 1, "多單": 1, "空單": 1, "三大法人": 1
}
forced_dictionary = construct_dictionary(word_to_weight)
# print('強制分詞字典：' + str(forced_dictionary) + '\n')

# --------------------- 斷詞斷字、詞性標注 with ckip -----------------
#  停用詞目前去不掉！！！已加入強制分詞字典～
print('[INFO]進行斷詞斷字、詞性標注中...')
tLoadStart = time.time()
word_sentence_list = ws(content_list,
                        # sentence_segmentation=True,   # To consider delimiters
                        # segment_delimiter_set={",", "。", ":", "?", "!", ";"},  # This is the defualt set of delimiters
                        # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
                        coerce_dictionary=forced_dictionary)  # words in this dictionary are forced
print('[INFO]斷詞斷字完成。')
pos_sentence_list = pos(word_sentence_list)
tLoadEnd = time.time()
print('[INFO]詞性標注完成。')
print("斷詞斷字、詞性標注 總共花費 %f sec\n" % (tLoadEnd - tLoadStart))

# ------------------------- Release model -----------------------------
del ws
del pos
del ner

"""
# -------------------- 建立強制分詞字典以及去除停用詞 with jieba -----------------------------
# https://www.itread01.com/content/1543561982.html

forced_dictionary = ['選擇權', '期貨', '台指期', '賣權', '買權', '多單', '空單', '三大法人']
for word in forced_dictionary:
    jieba.add_word(word)

stopwords = []
with open('stopWords.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        stopwords.append(data)

# --------------------- 斷詞斷字、詞性標注? with jieba -----------------
word_sentence_list = []
for i in range(len(content_list)):
     seg_list1 = jieba.cut(content_list[i], cut_all=False)
     result_1 = ",".join(seg_list1)
     word_sentence_list.append(result_1.split(','))

# ------------------------------
# 移除停用詞及跳行符號
# ------------------------------
word_sentence_list = list(filter(lambda a: a not in stopwords and a != '\n', word_sentence_list))
"""
# --------------------------- 關鍵字演算法 -----------------------------------

voc = []            # 儲存關鍵字前後n斷詞斷字文字
verb_voc = []       # 儲存關鍵字前後n斷詞斷字文字 (詞性)
index_list = []     # 儲存關鍵字斷詞索引值
stride = 2         # 擷取關鍵字前後 stride 筆 斷詞段字結果

for _index_wordSentence, element in enumerate(word_sentence_list):

    temp_list = []

    for line in keyword_list:
        if line in element:
            temp_list.extend([i for i, v in enumerate(word_sentence_list[_index_wordSentence]) if v == line])

    index_list.append(temp_list)

    for i in range(0, len(temp_list)):
        if temp_list[i] - stride < 0:
            voc.append(word_sentence_list[_index_wordSentence][:temp_list[i] + stride])
            verb_voc.append(pos_sentence_list[_index_wordSentence][:temp_list[i]+stride])
        else:
            voc.append(word_sentence_list[_index_wordSentence][temp_list[i]-stride:temp_list[i]+stride])
            verb_voc.append(pos_sentence_list[_index_wordSentence][temp_list[i]-stride:temp_list[i]+stride])

# ------------------------ 取出動詞，建立關鍵字 ”動詞字典“ --------------------------------
# https://python3-cookbook.readthedocs.io/zh_CN/latest/c01/p06_map_keys_to_multiple_values_in_dict.html

verbList = []
with open('Verb_list.txt', 'r', encoding='UTF-8') as file:
    for data in file.readlines():
        data = data.strip()
        verbList.append(data)

# verb_dic = dict.fromkeys(tuple(keyword_list))
verb_dic = defaultdict(list)

verb_num = 0                                                 # 迭代VOC次數用
for i, _indexKeyword in enumerate(index_list):

    for c in range(len(_indexKeyword)):
        h = word_sentence_list[i][_indexKeyword[c]]          # 取得關鍵字是期貨

        for b, ele in enumerate(verb_voc[verb_num]):
            if ele in verbList:                              # 找出動詞
                if voc[verb_num][b] not in verb_dic[h]:      # 判斷動詞是否在字典裡出現過
                    verb_dic[h].append(voc[verb_num][b])

        verb_num += 1

# VerbDic 存 csv
transposed_data = list(zip_longest(*verb_dic.values()))

with open('VerbDic.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(verb_dic.keys())
    for items in transposed_data:
        writer.writerow(items)

tEnd = time.time()  # 計時結束
print("This program total cost %f sec" % (tEnd - tStart))
