import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from itertools import chain
from transformers import AutoTokenizer, AutoModel
from transformers import logging
logging.set_verbosity_error()
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
PRETRAINED_TOKENIZER_NAME = "bert-base-uncased"
PRETRAINED_MODEL_NAME = "bert-base-uncased" #英文pretrain(不區分大小寫)
#載入bert token
# get pre-train tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_TOKENIZER_NAME)
embedding = AutoModel.from_pretrained(PRETRAINED_MODEL_NAME)

def bert_embedding(origin_text):
    
        text = tokenizer.tokenize(origin_text)
        if len(text) > 512:
            content_arr = []
            content_arr.append(origin_text)
            origin_text = ["".join([c for c in x if c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz)( :''\.0123456789\n\r+"]) for x in content_arr]
            origin_text = ''.join( repr(e) for e in origin_text )
            #這邊200是切Content前200個字，切300的話tokenize後還是有機會超過512個
            origin_text = origin_text.split()[:200]
            origin_text = ''.join( e+" " for e in origin_text )
        
#         print(type(contents))
        contents = torch.tensor(tokenizer.encode("[CLS]"+origin_text)).unsqueeze(0)
        
        #上述處理完還是有可能token超過512，直接截斷到前512
        if len(contents[0]) > 512:
            contents = contents[:,:512]

        out = embedding(contents)
        
        # BERT embedding的三個維度代表:Batch、序列長度、Hidden State Size
        # BERT作者推薦將輸出的[CLS]的768維向量Embedding當作句子的語義表示。
        # 因為在BERT模型設計中，[CLS]會有一個全連接層與其他所有token相連，微調後可以當作聚合了整句的資訊。
        embeddings_of_last_layer = out[0]
        cls_embeddings = embeddings_of_last_layer[0]

        return torch.mean(cls_embeddings,0)


if __name__ == '__main__':

    df = pd.read_csv('./Data/ALL_ORIGIN_DATA_review.csv')
    df['bert_embedding'] = df['review'].progress_apply(lambda x:bert_embedding(x).tolist())
    df.to_csv('./Data/ALL_ORIGIN_DATA_review_bert.csv',index=False, encoding='utf-8') 
   

 