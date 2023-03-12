

import sys        
sys.path.append("/Users/felixmeng/Desktop/RL_Decoder/src")
sys.path.append("/Users/felixmeng/Desktop/RL_Decoder/model")
from agent import *
from DQN import *
from dataReader import *
from env import *

print("test")

model_name = "facebook/bart-base"
# embedding_model=gensim.models.KeyedVectors.load_word2vec_format("/Users/felixmeng/Desktop/RL_Decoder/model/GoogleNews-vectors-negative300.bin.gz", binary=True)
# pretrained_model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
# pretrained_tokenizer=AutoTokenizer.from_pretrained(model_name)
# stop_words = stopwords.words('english')

### Environment Test
env= Env(input_sentence="I want to ",model_name=model_name,classifier=2)
env.update_state("add_word")
print(env.generated_sentence_so_far())
x=env.rollout_simulator()
print("x is",x)
print("id 2 word is",env.id2word(x))
