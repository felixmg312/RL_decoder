
from DQN import *
from env import *

Transition= namedtuple('Transition',('state', 'action', 'next_state', 'reward','termination'))
class ReplayMemory(object):
    def __init__(self, capacity):
        """
        Used deque so that if our counter is greater than maxlen, the previous will be removed
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition with each corresponding state"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        sample a random sample from the batch
        return 'state', 'action', 'next_state', 'reward','termination'
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Return current length of deque
        """
        return len(self.memory)

class DQNAgent:
    """
    Using Clipped double Q learning, so we have two deep Q networks and we are always updating with the min of the two Q values
    """
    def __init__(self,pretrained_model,pretrained_tokenizer,DQN,replay_memory,num_actions=3,learning_rate=3e-4,gamma=0.99, buffer_size=100,epsilon=0.2,eps_dec=5e-4,eps_min=0.01):
        self.learning_rate= learning_rate
        self.gamma= gamma
        self.eps_dec=eps_dec
        self.eps_min=eps_min
        self.epsilon=epsilon
        self.buffer_size=buffer_size
        self.replay_buffer= ReplayMemory(buffer_size)
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space=[i for i in range(num_actions)]
        self.pretrained_tokenizer=pretrained_tokenizer
        self.pretrained_model=pretrained_model
        self.id2action={0:"add_word",1:"remove_word",2:"replace_word"}
        self.max_seq_length=80
        ## 
        self.model1=DQN_with_attention(pretrained_model,self.pretrained_tokenizer)
        self.model2=DQN_with_attention(pretrained_model,self.pretrained_tokenizer)
  
        self.optimizer1 = torch.optim.Adam(self.model1.parameters())
        self.optimizer2 = torch.optim.Adam(self.model2.parameters())
    def get_action(self,input_sentence,input_vec):
#         print(input_sentence)
        qvals=self.model1.forward(input_sentence,input_vec)
#         print("qval is",qvals)
        action= np.argmax(qvals.cpu().detach().numpy())
        if(np.random.randn()<self.epsilon):
            return random.choice(self.action_space)
        return action
    def id2action(self,action_id):
        """
        given action id return action string
        """
        assert(action_id<3 and action_id>=0)
        return self.id2action[action_id]
    def state_to_tensor(self,state):
        """
        given a state, returns vector returns input id and attention mask
        """
        sentences,vectors=zip(*state)
        state_index_mask=self.pretrained_tokenizer(list(sentences), return_tensors="pt", padding='max_length', max_length=self.max_seq_length)
        vectors=torch.stack(vectors,dim=0)
        vectors=torch.squeeze(vectors,dim=1)
        return state_index_mask,vectors
    
    def read_from_replay_buffer(self,batch_size=10):
        """
        Given a sample buffer return the embedded states
        """
        transitions= self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))
#         print("batch next state",batch.next_state)
        sentences_state,vectors_state=self.state_to_tensor(batch.state)
        sentences_next_state,vectors_next_state=self.state_to_tensor(batch.next_state)
#         print("vectors next state",vectors_next_state)
        actions=torch.Tensor(batch.action).to(self.device)
        rewards=torch.Tensor(batch.reward).to(self.device)
        terminations=torch.Tensor(batch.termination).to(self.device)
        return sentences_state,vectors_state, actions,rewards,sentences_next_state,vectors_next_state,terminations
        
    def compute_loss(self,batch_size):
        ## reading from batch and reinitializing the state
        sentences_state,vectors_state, actions,rewards,sentences_next_state,vectors_next_state,terminations= self.read_from_replay_buffer(batch_size)
        print("vectors state",vectors_state)
        print("vectors next state",vectors_next_state)
        
        ## possible resizing necessary
        
        """
        compute the current-state-Q-values and the next-state-Q-values of both models,
        but use the minimum of the next-state-Q-values to compute the expected Q value
        """
#         print("sentences state shape",sentences_state)
#         print("vectors state shape", vectors_state.shape)
        curr_Q1= self.model1.forward(sentences_state,vectors_state.float())
#         print("curr_Q1",curr_Q1.shape)
        curr_Q2= self.model2.forward(sentences_state,vectors_state.float())
#         print("sentences state shape",sentences_next_state)
#         print("vectors state shape", vectors_next_state.shape)
        next_Q1= self.model1.forward(sentences_next_state,vectors_next_state.float())
        next_Q2= self.model2.forward(sentences_next_state,vectors_next_state.float())
        print(curr_Q1.shape,curr_Q2.shape,next_Q1.shape,next_Q2.shape)
        next_Q= torch.min(
            torch.max(next_Q1,1)[0],
            torch.max(next_Q2,1)[0]
        )
#         print(next_Q)
#         print(next_Q)
#         print("next q",next_Q.shape)
#         print(rewards.shape)
#         print("self gamma multiply next q",(self.gamma*next_Q).shape)
        expected_Q= rewards+ self.gamma*(1-terminations)*next_Q
        expected_Q= expected_Q.view(expected_Q.size(0),1)

#         print(expected_Q.shape)
        
        loss1= F.huber_loss(curr_Q1, expected_Q.detach())
        loss2= F.huber_loss(curr_Q2, expected_Q.detach())
        print(loss1,loss2)
        return loss1,loss2
    def update(self,batch_size):
        loss1,loss2=self.compute_loss(batch_size)
        self.optimizer1.zero_grad()
        loss1.backward()
        self.optimizer1.step()
        self.optimizer2.zero_grad()
        loss2.backward()
        self.optimizer2.step()
        self.epsilon= self.epsilon - self.eps_dec if self.epsilon >self.eps_min else self.eps_min
        
        