from agent import *
from DQN import *
from dataReader import *
from env import *
import tqdm as tq

class Trainer():
    def __init__(self,Env,agent,ReplayMemory,pretrained_model,pretrained_tokenizer,input_sentences,output_sentences,classifier,max_action_length=50,batch_size=16):
        """
        Env: Constructor
        agent: Constructed object
        DQN: Constructor
        ReplayMemory: Constructed Object
        Pretrained_model:loaded model
        Pretrained_tokenizer: loaded tokenizer
        input_sentences: input data set
        output_sentence: output data set
        Classifier: constructed object
        """
        self.agent=agent
        self.epoch_rewards=[]
        self.generated_sentences=[]
        self.input_sentences=input_sentences
        self.output_sentences=output_sentences
        self.batch_size=batch_size
        self.classifer=classifier
        self.pretrained_model=pretrained_model
        self.pretrained_tokenizer=pretrained_tokenizer
        self.Env=Env
        self.ReplayMemory= ReplayMemory
        self.max_action_length=max_action_length
        self.sentence_counter=0
    def train(self,batch_size):
        for input_sentence,output_sentence in zip(self.input_sentences,self.output_sentences):
            self.sentence_counter+=1
            env= self.Env(pretrained_model=self.pretrained_model,pretrained_tokenizer=self.pretrained_tokenizer,input_sentence=input_sentence,target_sentence=output_sentence,classifier=self.classifer)
            state= env.reset()
            # input_sentence,input_vec=state
            # input_sentence=self.pretrained_tokenizer(input_sentence,return_tensors='pt',padding='max_length', max_length=80)
            epoch_reward=0
            for epoch in tq.tqdm(range(self.max_action_length)):
                input_sentence,input_vec=state   
                input_vec=input_vec.to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
                print("actually summary is: ",output_sentence)
                print("currently generate is",env.generated_sentence_so_far())
                input_sentence=self.pretrained_tokenizer(input_sentence,return_tensors='pt',padding='max_length', max_length=80)
                action=self.agent.get_action(input_sentence,input_vec.float())
                next_state,reward,done=env.step(action)
                self.agent.replay_buffer.push(state, action, next_state,reward, done)
                epoch_reward+=reward
                if len(self.agent.replay_buffer) > batch_size:
                    self.agent.update(batch_size)
                if done or epoch==self.max_action_length-1:
                    print("currently generate is",env.generated_sentence_so_far())
                    avg_epoch_reward= epoch_reward/(epoch+1)
                    self.epoch_rewards.append(avg_epoch_reward)
                    self.generated_sentences.append(env.generated_sentence_so_far())
                    break
                print("epoch is",epoch,"reward is",epoch_reward)
                state=next_state
            if (self.sentence_counter)%10==0:
                self.write_result()
        self.agent.writer1.close()
        self.agent.writer2.close()
    def get_reward(self):
        return self.epoch_rewards
    def get_generated_sentences_so_far(self):
        return self.generated_sentences
    def write_result(self,location="result/"):
        average_rewards=self.get_reward()
        generated_sentences=self.get_generated_sentences_so_far()
        reward_file_name=location+"average_rewards.txt"
        generated_sentences_file_name= location+'generated_sequences.txt'
        with open(reward_file_name, "w") as f:
            f.writelines( "%s\n" % item for item in average_rewards)

        with open(generated_sentences_file_name, 'w') as f:
            # iterate over the list and write each element to a new line
            for item in generated_sentences:
                f.write(f"{item}\n")
    

