import torch as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple, deque
from collections import defaultdict


class DQN_with_attention(nn.Module):
    def __init__(self,pretrained_model,pretrained_tokenizer,hidden_size_1=50,hidden_size_2=100,embedding_size=768,max_seq_length=80,vector_size=4,action_size=3,lr=5e-5):
        super(DQN_with_attention,self).__init__()
        ## for encoder model
        self.max_seq_length=max_seq_length
        self.encoder = pretrained_model.get_encoder()
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.flatten= nn.Flatten(1)
        self.resize_embedding= nn.Linear(in_features=max_seq_length*embedding_size, out_features=hidden_size_2)
        self.pretrained_tokenizer=pretrained_tokenizer
        ## for fcn
        self.fc = nn.Linear(in_features=vector_size, out_features=hidden_size_1)
        self.relu = nn.LeakyReLU()
    
        ## combined
        self.final_linear = nn.Linear(in_features=hidden_size_1+hidden_size_2, out_features=action_size)
        
        ##Optimizer and loss
        self.optimizer= optim.Adam(self.parameters(),lr=lr)
        self.loss= nn.HuberLoss()
        self.device= T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    def forward(self, tokenized_sequence, input_vector):
        """
        Input sequence : tokenized sequence
        Input vector: 5 classifer vector => fcn to (1,10)
        Concatenate them and output the 3 actions
        """
        # print("input vector is ",input_vector)
        encoder_output = self.encoder(**tokenized_sequence)["last_hidden_state"]
        encoder_output = self.flatten(encoder_output)
        encoder_output= self.resize_embedding(encoder_output)
#         print("encoder output shape",encoder_output.shape)
        fc_output = self.fc(input_vector)
        fc_output = self.relu(fc_output)         
#         print("fc_outputshape",fc_output.shape)
        combined = torch.cat((encoder_output, fc_output), dim=-1)
        action = self.final_linear(combined)
        return action