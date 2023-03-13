from agent import *
from DQN import *
from dataReader import *
from env import *

class Trainer():
    def __init__(self,Env,agent,DQN,ReplayMemory,pretrained_model,pretrained_tokenizer,input_sentences,output_sentences,classifier,batch_size=16):
        self.agent=agent
        self.epoch_rewards=[]
        self.input_sentences=input_sentences
        self.output_sentences=output_sentences
        self.batch_size=batch_size
        self.classifer=classifier
        self.pretrained_model=pretrained_model
        self.pretrained_tokenizer=pretrained_tokenizer
        self.Env=Env
        self.DQN=DQN
        self.ReplayMemory= ReplayMemory
    def train(self,batch_size,max_gen_length):
        for input_sentence,output_sentence in zip(self.input_sentences,self.output_sentences):
            # print("training")
            env= self.Env(pretrained_model=self.pretrained_model,pretrained_tokenizer=self.pretrained_tokenizer,input_sentence=input_sentence,target_sentence=output_sentence,classifier=self.classifer)
            state= env.reset()
            input_sentence,input_vec=state
            input_sentence=self.pretrained_tokenizer(input_sentence,return_tensors='pt',padding='max_length', max_length=80)
            max_gen_length=30
            epoch_reward=0
            for epoch in range(max_gen_length):    
                action=self.agent.get_action(input_sentence,input_vec)
                next_state,reward,done=env.step(action)
                print(state,next_state)
                self.agent.replay_buffer.push(state, action, next_state,reward, done)
                epoch_reward+=reward
                if len(self.agent.replay_buffer) > batch_size:
                    self.agent.update(batch_size)
                    # print('in agent update')
                if done or epoch==max_gen_length-1:
                    # print("currently generate is",env.generate_sentence_so_far())
                    self.epoch_rewards.append(epoch_reward)
                print("epoch is",epoch,"reward is",epoch_reward)
                state=next_state