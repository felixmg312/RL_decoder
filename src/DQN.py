import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pyemd import emd
import gensim
from nltk.corpus import stopwords
from nltk import download
download('stopwords')
from collections import namedtuple, deque
from collections import defaultdict
class DQN_with_attention(nn.Module):
    def __init__(self, pretrained_model,hidden_size_1=50,hidden_size_2=100,embedding_size=768,max_seq_length=30,vector_size=4,action_size=3,lr=5e-5):
        super(DQN_with_attention,self).__init__()
        ## for encoder model
        self.max_seq_length=max_seq_length
        self.encoder = pretrained_model.get_encoder()
        self.flatten= nn.Flatten(1)
        self.resize_embedding= nn.Linear(in_features=max_seq_length*embedding_size, out_features=hidden_size_2)
        
        ## for fcn
        self.fc = nn.Linear(in_features=vector_size, out_features=hidden_size_1)
        self.relu = nn.LeakyReLU()
    
        ## combined
        self.final_linear = nn.Linear(in_features=hidden_size_1+hidden_size_2, out_features=3)
        
        ##Optimizer and loss
        self.optimizer= optim.Adam(self.parameters(),lr=lr)
        self.loss= nn.HuberLoss()
        self.device= T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, input_sequence, input_vector):
        """
        Input sequence : sentence => encodes to 30,768 => fcn to (1,100)
        Input vector: 5 classifer vector => fcn to (1,10)
        Concatenate them and output the 3 actions
        """
        x=pretrained_tokenizer(input_sequence,return_tensors='pt',padding='max_length', max_length=self.max_seq_length)
        encoder_output = self.encoder(x['input_ids'], attention_mask=x['attention_mask'])["last_hidden_state"]
        encoder_output = self.flatten(encoder_output)
        encoder_output= self.resize_embedding(encoder_output)
        fc_output = self.fc(input_vector)
        fc_output = self.relu(fc_output)
        print(fc_output.shape,encoder_output.shape)
        combined = torch.cat((encoder_output, fc_output), dim=-1)
        action = self.final_linear(combined)
        return action