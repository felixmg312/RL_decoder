from agent import *
from DQN import *
from dataReader import *
from env import *
import tqdm as tq
from torchmetrics.text.rouge import ROUGEScore


class Test():
    def __init__(self,Env,agent,pretrained_model,pretrained_tokenizer,input_sentences,output_sentences,classifier,max_action_length=50):
        self.agent=agent
        self.generated_sentences=[]
        self.input_sentences=input_sentences
        self.output_sentences=output_sentences
        self.classifer=classifier
        self.pretrained_model=pretrained_model
        self.pretrained_tokenizer=pretrained_tokenizer
        self.Env=Env
        self.max_action_length=max_action_length
        self.sentence_counter=0     
        self.rogue_scores=[]
    def test(self,input_test,output_test):
        for input_sentence,output_sentence in zip(input_test,output_test):
            self.sentence_counter+=1
            env= self.Env(pretrained_model=self.pretrained_model,pretrained_tokenizer=self.pretrained_tokenizer,input_sentence=input_sentence,target_sentence=output_sentence,classifier=self.classifer)
            state= env.reset()
            for epoch in tq.tqdm(range(self.max_action_length)):    
                input_sentence,input_vec=state
                input_sentence=self.pretrained_tokenizer(input_sentence,return_tensors='pt',padding='max_length', max_length=80)
                action=self.agent.get_action(input_sentence,input_vec,test=True)
                next_state,_,done=env.step(action)
                if done or epoch==self.max_action_length-1:
                    # print("currently generate is",env.generated_sentence_so_far())
                    sentence_generated=env.generated_sentence_so_far()
                    self.generated_sentences.append(sentence_generated)
                    rouge_score=self.rouge_score(sentence_generated,output_sentence)
                    self.rogue_scores.append(rouge_score)
                    break
                state=next_state
            if (self.sentence_counter)%10==0:
                self.write_result()
    def rouge_score(self,generated_sentence,target_sentence):
        rouge = ROUGEScore()
        score = rouge(generated_sentence, target_sentence)
        return score['rouge1_fmeasure']
    def get_rouge_score(self):
        return self.rogue_scores
    def write_result(self,location="result/"):
        rouge_scores=self.get_rouge_score()
        generated_sentences=self.get_generated_sentences_so_far()
        score_file_name=location+"rouge_scores.txt"
        generated_sentences_file_name= location+'generated_sequences_for_test.txt'
        with open(score_file_name, "w") as f:
            f.writelines( "%s\n" % item for item in rouge_scores)

        with open(generated_sentences_file_name, 'w') as f:
            # iterate over the list and write each element to a new line
            for item in generated_sentences:
                f.write(f"{item}\n")