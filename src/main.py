import sys        
sys.path.append("model")
sys.path.append("result")

from agent import *
from classifier import *
from dataReader import *
from DQN import *
from env import *
from train import *


if __name__ == '__main__':

    data_reader=Dataset_Reader(data_name="gigaword", test_size=0.1, data_set_size=20000,mode="training")
    input_train,output_train=data_reader.get_training()
    input_test,output_test=data_reader.get_testing()

    ## Model Initialization
    #model_name = "facebook/bart-base"
    model_name = "model/finetuneBart" 

    checkpoint_path1="checkpoints_model1/model_epoch_2200.pt"
    checkpoint_path2="checkpoints_model2/model_epoch_2200.pt"
    # agent.load_model(checkpoint_path1,checkpoint_path2) ## uncomment when u want to load checkpoint

    pretrained_model=AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pretrained_tokenizer=AutoTokenizer.from_pretrained(model_name)

    ##Initalizing the constructors
    replay_memory= ReplayMemory(100)
    agent=DQNAgent(pretrained_model,pretrained_tokenizer,DQN_with_attention,replay_memory)
    classifier=Classifier("model/model_save")

    ### Trainer Test
    trainer=Trainer(Env,agent,replay_memory,pretrained_model,pretrained_tokenizer,input_train,output_train,classifier)
    batch_size=16
    trainer.train(batch_size=batch_size)

    print(trainer.get_reward())
    print(trainer.get_generated_sentences_so_far())
    average_rewards=trainer.get_reward()
    generated_sentences=trainer.get_generated_sentences_so_far()
    trainer.write_result()