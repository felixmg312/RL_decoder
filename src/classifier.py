from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from transformers import logging

class Classifier:
    def __init__(self,path):
        self.grammar_model = BertForSequenceClassification.from_pretrained(path)
        self.grammar_tokenizer = BertTokenizer.from_pretrained(path)
        fluency_pipe = pipeline(model="prithivida/parrot_fluency_model",task = 'text-classification',return_all_scores = True)
        sentiment_pipe = pipeline(model="siebert/sentiment-roberta-large-english",task = 'text-classification',return_all_scores = True)
        topic_pipe = pipeline(model="ebrigham/EYY-Topic-Classification",task = 'text-classification',return_all_scores = True)
        self.pipes = [fluency_pipe,sentiment_pipe,topic_pipe]
        self.loss = torch.nn.CrossEntropyLoss()

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

    def get_scores(self,target,decoded):
      scores = []
      for p in self.pipes:
        target_pred = p(target)[0]
        decoded_pred = p(decoded)[0]
        prob_target = []
        prob_decoded = []
        for i in range(len(target_pred)):
          prob_target.append(target_pred[i]['score'])
          prob_decoded.append(decoded_pred[i]['score'])
        scores.append(self.loss(torch.tensor(prob_decoded),torch.tensor(prob_target)).item())
      
      grammar_target = self.grammar_pred(target)
      grammar_decoded = self.grammar_pred(decoded)
      scores.append(self.loss(grammar_decoded,grammar_target).item())
      return scores
        


