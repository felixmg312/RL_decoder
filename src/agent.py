
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
    def __init__(self,env,pretrained_model,DQN,replay_memory,learning_rate=3e-4,gamma=0.99, buffer_size=1000,eps_dec=5e-4,eps_min=0.01):
        self.env=env()
        self.learning_rate= learning_rate
        self.gamma= gamma
        self.eps_dec=eps_dec
        self.eps_min=eps_min
        self.buffer_size=buffer_size
        self.replay_buffer= ReplayMemory(buffer_size)
        self.device=torch.device("cuda" if torch.cuda.is_availabe() else "cpu")
        ## 
        self.model1=DQN_with_attention(pretrained_model)
        self.model2=DQN_with_attention(pretrained_model)
  
        self.optimizer1 = torch.optim.Adam(self.model1.parameters())
        self.optimizer2 = torch.optim.Adam(self.model2.parameters())
    def get_action(self,state,eps=0.2):
        qvals=self.model1.forward(state)
        action= np.argmax(qvals.cpu().detach().numpy())
        
        if(np.random.randn()<eps):
            return self.env.action_space.sample()
    def compute_loss(self,batch_size):
        ## reading from batch and reinitializing the state
        states, actions, rewards, next_states, terminations= self.replay_buffer.sample(batch_size)[0]
        states=torch.FloatTensor(states).to(self.device)
        actions=torch.LongTensor(actions).to(self.device)
        rewards=torch.FloatTensor(rewards).to(self.device)
        next_states= torch.FloatTensor(next_states).to(self.device)
        terminations=torch.FloatTensor(terminations)
        
        ## possible resizing necessary
        
        """
        compute the current-state-Q-values and the next-state-Q-values of both models,
        but use the minimum of the next-state-Q-values to compute the expected Q value
        """
        curr_Q1= self.model1.forward(states)
        curr_Q2= self.model2.forward(states)
        next_Q1= self.model1.forward(next_states)
        next_Q2= self.model2.forward(next_states)
        
        next_Q= torch.min(
            torch.max(next_Q1,1)[0],
            torch.max(next_Q2,1)[0]
        )
        
        next_Q= next_Q.view(next_Q.size(0),1)
        expected_Q= rewards+ (1-dones)*self.gamam*next_Q
        
        
        loss1= F.huber_loss(curr_Q1, expected_Q.detach())
        loss2= F.huber_loss(curr_Q2, expected_Q.detach())
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
        