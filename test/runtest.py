

import sys        
sys.path.append("/Users/felixmeng/Desktop/RL_Decoder/src")
sys.path.append("/Users/felixmeng/Desktop/RL_Decoder/model")
from agent import *
from DQN import *
from dataReader import *
from env import *
from classifier import *
from train import *
import datasets

## Dataset
from datasets import load_dataset
dataset=load_dataset('gigaword',split = 'train[:20000]')
train_data, test_data= dataset.train_test_split(test_size=0.1).values()
small_dataset= datasets.DatasetDict({'train':train_data,'test':test_data})
input_sentences=small_dataset['train']['document'][:100]
output_sentences=small_dataset['train']['summary'][:100]


## Model Initialization
model_name = "facebook/bart-base"
# embedding_model=gensim.models.KeyedVectors.load_word2vec_format("/Users/felixmeng/Desktop/RL_Decoder/model/GoogleNews-vectors-negative300.bin.gz", binary=True)
pretrained_model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
pretrained_tokenizer=AutoTokenizer.from_pretrained(model_name)
# stop_words = stopwords.words('english')

##Environment Test
input_sentence=input_sentences[0]
output_sentence=output_sentences[0]
replay_memory= ReplayMemory(100)
DQN=DQN_with_attention(pretrained_model,pretrained_tokenizer)
agent=DQNAgent(pretrained_model,pretrained_tokenizer,DQN,ReplayMemory)
classifier=Classifier("model/model_save")
# env= Env(pretrained_model=pretrained_model,pretrained_tokenizer=pretrained_tokenizer,input_sentence=input_sentence,target_sentence=output_sentence,classifier=classifier)

### Trainer Test
trainer=Trainer(Env,agent,DQN,ReplayMemory,pretrained_model,pretrained_tokenizer,input_sentences,output_sentences,classifier)
trainer.train(16,30)

