from agent import *
from DQN import *
from dataReader import *
from env import *

def train(env,agent,max_epochs,max_steps,batch_size):
    """
    env: need to implement step
    agent:
    max_epochs: max training length
    max_steps: max sequence length of our sentence
    batch_size: the batch that we are taking for each training epoch
    """
    epoch_rewards=[]
    for epoch in range(max_epochs):
        #state= env.reset()
        epoch_reward=0
        for step in range(max_steps):
            action= agent.get_action(state)
            next_state,reward,done=env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            epoch_reward+=reward
            
            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)
            if done or step == max_steps-1:
                epoch_rewards.append(epoch_reward)
                print("epoch" + str(epoch) + ": " + str(epoch_reward))
                break
            state= next_state
    return epoch_rewards