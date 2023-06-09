from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import logging
import numpy as np
import torch as T
import gensim
class Classifier:
    def __init__(self,path,embedding_path="model/GoogleNews-vectors-negative300.bin.gz"):
        self.grammar_model = BertForSequenceClassification.from_pretrained(path)
        self.grammar_tokenizer = BertTokenizer.from_pretrained(path)
        fluency_pipe = pipeline(model="prithivida/parrot_fluency_model",task = 'text-classification',return_all_scores = True)
        sentiment_pipe = pipeline(model="siebert/sentiment-roberta-large-english",task = 'text-classification',return_all_scores = True)
        topic_pipe = pipeline(model="ebrigham/EYY-Topic-Classification",task = 'text-classification',return_all_scores = True)
        self.pipes = [fluency_pipe,sentiment_pipe,topic_pipe]
        self.loss = torch.nn.CrossEntropyLoss()
        self.embedding_model=gensim.models.KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    def sentiment_pred(self,sent):
        return self.pipes[1](sent)

    def grammar_pred(self,sent):
        encoded_dict = self.grammar_tokenizer.encode_plus(
                sent,                     
                add_special_tokens = True, 
                max_length = 64,         
                pad_to_max_length = True,
                return_attention_mask = True,   
                return_tensors = 'pt', 
            )
        input_id = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        input_id = torch.LongTensor(input_id)
        attention_mask = torch.LongTensor(attention_mask)
        outputs = self.grammar_model(input_id, token_type_ids=None, attention_mask=attention_mask)
        logits = outputs[0]
        return torch.nn.Softmax(dim=1)(logits)

    def fluency_pred(self,sent):
        return self.pipes[0](sent)

    def topic_pred(self,sent):
        return self.pipes[2](sent)
    
    def sentence_mover_distance(self,generated_sentence,base_line_sentence,target_sentence):
        """
        Expect all 3 inputs to be a sentence string object, returns baseline_distance and generated_distance
        """
        target_sentence = target_sentence.lower().split()
        base_line_sentence= base_line_sentence.lower().split()
        generated_sentence= generated_sentence.lower().split()
        print(generated_sentence,base_line_sentence, target_sentence)

        # generated_sentence = [w for w in generated_sentence if w not in self.stopwords]
        # target_sentence = [w for w in target_sentence if w not in self.stopwords]
        # base_line_sentence=[w for w in base_line_sentence if w not in self.stopwords]
        generated_distance= self.embedding_model.wmdistance(generated_sentence, target_sentence)
        baseline_distance= self.embedding_model.wmdistance(base_line_sentence, target_sentence)
        # print("baseline_distance",baseline_distance,"generated distance",generated_distance)
        return baseline_distance,generated_distance
    
    def get_scores(self,target,decoded):
    #cross entropy scores between target and decoded = [fluency, sentiment, topic, grammar]
      fluency_decoded = self.pipes[0](decoded)
      scores = [-np.log(fluency_decoded[0][0]['score'])]
      fluency_decoded = self.pipes[0](decoded)
      for p in self.pipes[1:]:
        target_pred = p(target)[0]
        decoded_pred = p(decoded)[0]
        prob_target = []
        prob_decoded = []
        for i in range(len(target_pred)):
          prob_target.append(target_pred[i]['score'])
          prob_decoded.append(decoded_pred[i]['score'])
        prob_decoded = torch.tensor(prob_decoded).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))
        prob_target = torch.tensor(prob_target).to(T.device('cuda:0' if T.cuda.is_available() else 'cpu'))

        target_max_idx = torch.argmax(prob_target, 0, keepdim=True)
        target_one_hot = torch.FloatTensor(prob_target.shape)
        target_one_hot.zero_()
        target_one_hot.scatter_(0, target_max_idx, 1)
        scores.append(- self.loss(prob_decoded,target_one_hot).item())
      grammar_decoded = self.grammar_pred(decoded)
      scores.append(-np.log(grammar_decoded[0,0].item()))
      # print("the score is:",scores)
      return scores