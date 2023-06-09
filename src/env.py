import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch 
import gensim
import random
# from nltk.corpus import stopwords
# from nltk import download
# download('stopwords')


class Env():
    def __init__(self,input_sentence,pretrained_model,pretrained_tokenizer,target_sentence,classifier,max_action_length=50,classifier_vector_length=4):
        """
        input_sentence: the input sentence x 
        output_sentence: the target sentence, only used to calculate reward
        model_name: the transformer model that we are using
        sentence_helper: the sentence class that contains helper function for the currently decoded word
        reward: the reward class that will return the reward of a new action
        """
       
        ##Initialize variables
        self.target_sentence=target_sentence
        self.input_sentence=input_sentence
        self.classifier_vector_length=classifier_vector_length
        self.classifier= classifier
        self.done= False
        self.max_action_length=max_action_length
        ## Intialize Model and Tokenizer
        self.model = pretrained_model
        self.tokenizer = pretrained_tokenizer
        self.input_ids = self.get_ids(input_sentence).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu')) ## tokenized the input sentence
        self.decoder_input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu')) ## id for start token
        self.next_state = self.model(self.input_ids, decoder_input_ids= self.decoder_input_ids, return_dict=True)
        self.last_hidden_encoder_state = (self.next_state.encoder_last_hidden_state.to(T.device('cuda:0' if T.cuda.is_available() else 'cpu')),) ## the hidden state
        self.eos_token_id = self.model.config.eos_token_id
        self.id2action={0:"add_word",1:"remove_word",2:"replace_word"}
        self.reward=0
        ##For reward
        self.add_word_counter=0
        self.remove_word_counter=0
        self.replace_word_counter=0
        self.action_counter=0
        ## Freezing the model
        for param in self.model.parameters():
            param.requires_grad = False
        """
        vector_state= array(float)
        sentence_state=input_sentence+ currently decoded sentence
        """
        ## Initialize the state
        self.vector_state=torch.zeros(1,classifier_vector_length).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
        self.sentence_state= self.input_sentence+self.generated_sentence_so_far()
        
    def reset(self):
        """
        returns the initial state of the environment 
        """
        self.done=False
        self.sentence_state= self.input_sentence
        self.vector_state=torch.zeros(1,self.classifier_vector_length).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
        self.reward=0
        return self.sentence_state,self.vector_state
    def get_top_k(self,k=30):
        """
        get the next possible choices given current decode state
        return top k prob words
    
        """
        logits  = self.model(None, encoder_outputs= self.last_hidden_encoder_state, decoder_input_ids= self.decoder_input_ids, return_dict=True).logits
#         print(logits,logits.shape)
        logits=logits[:,-1,:].squeeze()

        softmax = nn.Softmax(dim=0)
        ## Embedding of top k words
        values, idx = torch.topk(logits, k=k, axis=-1)
        # print("value",values, "idx",idx)
        probs= softmax(values)
        # print("probability is", probs, "idxa is", idx)
        return idx, probs

    def sample_word(self,idx,probs):
        """
        Sample a word given idx and probabilities
        """
        assert(len(idx)==len(probs))
        idx=idx.detach().numpy()
        probs=probs.detach().numpy()
        next_word_id=np.random.choice(idx, p=probs)
        # print("chosen index is", next_word_id)
        return next_word_id
    def get_ids(self,sentence):
        """
        Get the tokenized id of a given sentence
        """
        return self.tokenizer(sentence,return_token_type_ids=False,return_tensors='pt')["input_ids"]
    def get_mask(self,sentence):
        """
        Get the attention mask for a given sentence
        """
        return self.tokenizer(sentence,return_token_type_ids=False,return_tensors='pt')["attention_mask"]
    def id2word(self,ids):
        """
        id is list of list
        given ids return sentence
        """
        return self.tokenizer.decode(ids[0], skip_special_tokens = True)
        
    def remove_word(self):
        """
        remove a word
        """
        self.decoder_input_ids=self.decoder_input_ids[0][:-1].unsqueeze(0)
        
    def add_word(self,k=30,is_rollout=False):
        """
        Choose a word according to top k and update the current decoder input ids
        """
        idx,probs=self.get_top_k(k)
        decoder_idx_chosen=self.sample_word(idx,probs)
        if decoder_idx_chosen==self.eos_token_id and is_rollout==False:
            self.done=True
        next_decoder_input_ids = torch.tensor([[decoder_idx_chosen]]).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
        self.decoder_input_ids = torch.cat([self.decoder_input_ids, next_decoder_input_ids], axis=-1).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
        if decoder_idx_chosen==self.eos_token_id and is_rollout==True:
            return True
        return False
    def update_state(self,action):
        """
        action: add_word, remove_word, replace_word
        Update the current state and returns next state given an action word
        """
        action= self.id2action[action]
        self.action_counter+=1
        if action == "remove_word":
            self.remove_word_counter+=1
            if len(self.generated_sentence_so_far())<1:
                return
            self.remove_word()
           
        if action== "add_word":
            self.add_word_counter+=1
            self.add_word()
            self.add_word()

            
        if action == "replace word":
            self.replace_word_counter+=1
            self.remove_word()
            self.add_word()
            # self.add_word()
            self.add_word()
 
    def rollout_simulator(self,max_length=30):
        """
        use top k for rollout
        """
        is_rollout_done=False
        current_decoded_input_id=self.decoder_input_ids
        count=0
        while is_rollout_done==False and count< max_length:
            is_rollout_done=self.add_word(k=2000,is_rollout=True)
            if(is_rollout_done==True):
                break
            count+=1
        rollout_decoded_input_id=self.decoder_input_ids
        self.decoder_input_ids= current_decoded_input_id
        return rollout_decoded_input_id
        
    
    def get_next_state(self):
        """
        Update the crossentropy vector and return the sentence state
        """
        rollout_sentence=self.id2word(self.rollout_simulator())
        self.vector_state= torch.tensor(self.classifier.get_scores(self.input_sentence,rollout_sentence))
        self.vector_state= self.vector_state.unsqueeze(0)
        self.sentence_state= self.input_sentence+self.generated_sentence_so_far()
        return self.sentence_state,self.vector_state
    def generated_sentence_so_far(self):
        """
        returns the currently decoded words so far
        """
        return self.tokenizer.decode(self.decoder_input_ids[0], skip_special_tokens = True)
    
    def is_termination(self):
        return self.done
    ### decoding method:
 
    def decode_whole_sentence(self,decoding_strategy="greedy"):
        """
        Decode the whole sentence given an input:
        returns a string
        """
        if decoding_strategy=="greedy":
            greedy_output = self.model.generate(self.input_ids)
            return self.tokenizer.decode(greedy_output[0], skip_special_tokens=True)
        if decoding_strategy== "beam":
            beam_output = self.model.generate(
              self.input_ids,
              num_beams=1, 
              no_repeat_ngram_size=2, 
              early_stopping=True
              )
            return self.tokenizer.decode(beam_output[0], skip_special_tokens=True)
        if decoding_strategy== "topK":
            topk_output = self.model.generate(
              self.input_ids, 
              do_sample=True, 
              top_k=50,
              temperature=0.7
              )
            return self.tokenizer.decode(topk_output[0], skip_special_tokens=True)
        if decoding_strategy== "topP":
            topP_output= self.model.generate(
                input_ids=self.input_ids,
                max_length=50,
                top_p=0.9,
                do_sample=True
            )
            return self.tokenizer.decode(topP_output[0], skip_special_tokens=True)

                

    def step(self,action):
        """
        returns next_state, reward, termination
        """
        self.update_state(action)
        next_state= self.get_next_state()
        termination= self.is_termination()
        reward=self.get_reward(action,termination=termination)
            
        return next_state,reward,termination



    def get_reward(self,action,termination,weight_vec=[1,1,0.001,1]):
        if termination is False:
            simulated_input_id=self.rollout_simulator()
            simulated_sentence= self.id2word(simulated_input_id)
            score_vec = np.array(self.classifier.get_scores(self.input_sentence,simulated_sentence))
            reward= score_vec.dot(weight_vec)
            if (action == 1 or action == 2) and len(self.generated_sentence_so_far())<=1:
                reward-=5
            # if action == 0:
            #     reward*=2
            # if self.add_word_counter <= (self.remove_word_counter+self.replace_word_counter):
            #     reward-=5
            # if self.action_counter == self.max_action_length:
            #     print("not generating a sentence")
            #     reward -=10
        else:
            generated_sentence=self.generated_sentence_so_far()
            score_vec = np.array(self.classifier.get_scores(self.target_sentence,generated_sentence))
            reward= score_vec.dot(weight_vec)
            baseline_sentence=self.decode_whole_sentence()
            baseline_distance,generated_distance=self.classifier.sentence_mover_distance(generated_sentence,baseline_sentence,self.target_sentence)
            print("baseline_distance:",baseline_distance)
            print("generated_dsitance:",generated_distance)
            if len(generated_sentence)<=3:
                reward-=10
            if generated_distance<= baseline_distance:
                reward+=20
            if generated_distance<=1:
                reward+=20
       
        return reward