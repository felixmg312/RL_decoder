

import sys        
sys.path.append("src")
sys.path.append("model")
from agent import *
from DQN import *
from dataReader import *
from env import *
from classifier import *
from train import *
from tester import *
import datasets

## Dataset
from datasets import load_dataset
dataset=load_dataset('gigaword',split = 'train[:20000]')
train_data, test_data= dataset.train_test_split(test_size=0.1).values()
small_dataset= datasets.DatasetDict({'train':train_data,'test':test_data})
input_sentences=small_dataset['train']['document'][:10]
output_sentences=small_dataset['train']['summary'][:10]


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
agent=DQNAgent(pretrained_model,pretrained_tokenizer,DQN_with_attention,replay_memory)
classifier=Classifier("model/model_save")
# env= Env(pretrained_model=pretrained_model,pretrained_tokenizer=pretrained_tokenizer,input_sentence=input_sentence,target_sentence=output_sentence,classifier=classifier)

### Trainer Test
# trainer=Trainer(Env,agent,replay_memory,pretrained_model,pretrained_tokenizer,input_sentences,output_sentences,classifier)
# trainer.train(16)

# print(trainer.get_reward())
# print(trainer.get_generated_sentences_so_far())

## Test Test
data_reader=Dataset_Reader(data_name="gigaword", test_size=0.1, data_set_size=20000,mode="training")
input_train,output_train=data_reader.get_training()
input_test,output_test=data_reader.get_testing()
test_obj = Test(Env,agent,pretrained_model,pretrained_tokenizer,classifier,max_action_length=50)
test_obj.test(input_test,output_test)