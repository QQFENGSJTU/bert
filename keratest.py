import tensorflow
import keras
from bert_serving.client import BertClient
import numpy as np

'''bc = BertClient()
print('here encode')
a = bc.encode(['上海'])
print('ok')
print(a)'''

def cos_similar(sen_a_vec,sen_b_vec):
    vector_a = np.mat(sen_a_vec)
    vector_b = np.mat(sen_b_vec)
    num = float(vector_a * vector_b.T)
    print('here')
    print(num)
    denom = np.linalg.norm(vector_a)* np.linalg.norm(vector_b)
    print(f'demon {denom}')
    cos = num/denom
    return cos


def similarity_bert(word1, word2):
    bc = BertClient()
    vec1 = bc.encode([word1])
    vec2 = bc.encode([word2])
    print('sim')
    sim = cos_similar(vec1,vec2)
    bc.close()
    return sim

def script_start_bert():
    from bert_serving.server.helper import get_args_parser
    from bert_serving.server import BertServer
    args = get_args_parser().parse_args(['-model_dir', '/home/qqfeng/Desktop/nlp4b/chinese_L-12_H-768_A-12',
                                     '-pooling_strategy', 'NONE',
                                     '-max_seq_len','60'])

    server = BertServer(args)
    server.start()
    from bert_serving.client import BertClient
    bc = BertClient(ip='localhost')
    test=bc.encode(['你好','bert'])
    print(test)


if __name__ == '__main__':
    t1 = '上海'
    t2 = '北京'
    #sim1 = similarity_bert(t1,t2)
    t3 = '人工智能'
    #sim2 = similarity_bert(t1,t3)
    #print(f'sim1 = {sim1}')
    #print(f'sim2 = {sim2}')
    script_start_bert()