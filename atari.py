import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim 
import cv2
import imageio
from collections import namedtuple, deque
import random
random.seed(123)

def to_grayscale(img):
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    img = img.astype(np.uint8)
    return img

def downsample(img):
    resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    x_t = resized_screen[18:102, :]
    return x_t

def preprocess(img):
    return downsample(to_grayscale(img))

def transform_reward(reward):
    return np.sign(reward)

def select_action(state, policy_model, threshold, n_actions, device):
    state = torch.from_numpy(state).type(torch.float32).unsqueeze(dim=0).to(device) #convert numpy -> tensor
    sample = random.random()

    if sample > threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_model(state).max(1)[1].cpu().numpy()[0]
    else:
        return random.randrange(n_actions)

def env_reset(env, no_op_steps):
    frame = env.reset()
    for _ in range(random.randint(1, no_op_steps)):
        frame, _, _, _ = env.step(1)
    frame = preprocess(frame)
    frame_q = deque(maxlen=4)
    for _ in range(4):
        frame_q.append(frame)
    state = np.stack(frame_q, axis=0)
    return state, frame_q

def optimize_model(rm, batch_size, policy_model, target_model, optimizer, device, GAMMA=0.999):
    if len(rm) < 50000:
        return
    transitions = rm.sample(batch_size)
    batch = Transition(*zip(*transitions))
    state_batch = torch.from_numpy(np.stack(batch.state, axis=0)).float().to(device)/255
    next_state_batch = torch.from_numpy(np.stack(batch.next_state, axis=0)).float().to(device)/255
    action_batch = torch.from_numpy(np.stack(batch.action, axis=0)).long().to(device)
    reward_batch = torch.from_numpy(np.stack(batch.reward, axis=0)).float().to(device)
    done_batch = torch.from_numpy(np.stack(batch.done, axis=0).astype(int)).byte().to(device)

    #(N, 4) -> (N, 1) -> (N)
    q_value = policy_model(state_batch).gather(1, action_batch.long().unsqueeze(-1)).squeeze(-1) 
    #(N, 4) -> (N, 1), max:0 value, 1 index
    next_q_value = target_model(next_state_batch).max(1)[0]
    next_q_value[done_batch] = 0.0
    next_q_value = next_q_value.detach()

    expected_q_value = next_q_value * GAMMA + reward_batch
    
    optimizer.zero_grad()
    loss = F.smooth_l1_loss(q_value, expected_q_value)
    loss.backward()
    for param in policy_model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

def play(env, policy_model, device):
    frames = []
    total_reward = 0

    frame = env.reset()
    frame, reward, is_done, _ = env.step(1)
    frame, reward, is_done, _ = env.step(2)

    env.render()

    frames.append(frame)

    frame = preprocess(frame)
    frame_q = deque(maxlen=4)
    for _ in range(4):
        frame_q.append(frame)

    while True:
        state = np.stack(frame_q, axis=0)
        state = torch.from_numpy(state).type(torch.float32).unsqueeze(dim=0).to(device) #convert numpy -> tensor
        action = policy_model(state).max(1)[1].cpu().numpy()[0]
        frame, reward, is_done, _ = env.step(action)
        env.render()
        frames.append(frame)
        total_reward += reward
        frame = preprocess(frame)
        frame_q.append(frame)

        if is_done:
            print(total_reward)
            env.close()
            break

    # imageio.mimsave('/path/to/movie.gif', images)
    # print('Done')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

class Model(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, (8, 8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, (4, 4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, (3, 3), stride=1)
        self.dense = nn.Linear(3136, 512)
        self.out = nn.Linear(512, n_actions)

        for name, p in self.named_parameters():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.kaiming_normal(p, a=0, mode='fan_in')
                                            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 3136)
        x = F.relu(self.dense(x))
        return self.out(x)

class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

def main():
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.02
    EPS_DECAY = 10**5
    TARGET_UPDATE = 10000
    LEARNING_RATE = 0.00025
    LOAD_MODEL = False
    no_op_steps = 30
    # Create a breakout environment
    device = 'cuda:0'
    # env = gym.make('BreakoutDeterministic-v4')
    env = gym.make('Pong-v0')
    n_actions = env.action_space.n
    # Reset it, returns the starting frame
    
    # Render
    # env.render()

    policy_model = Model(n_actions).to(device)
    target_model = Model(n_actions).to(device)
    target_model.eval()

    if LOAD_MODEL:
        policy_model.load_state_dict(torch.load('./checkpoints/BreakoutDeterministic-v4-474.dat'))
        target_model.load_state_dict(policy_model.state_dict())
        print('Loaded!')
        # while True:
        #     play(env, policy_model, device)

    optimizer = optim.RMSprop(policy_model.parameters(), lr=LEARNING_RATE)
    
    rm = ReplayMemory(10**4*4)
    
    steps_done = 0
    total_reward = 0
    total_rewards = []
    best_reward = None
    state, frame_q = env_reset(env, no_op_steps)

    while True:
        # Perform a random action, returns the new frame, reward and whether the game is over
        n_episodes = len(total_rewards)
        steps_done += 1
        if steps_done > 10000:
            epsilon = max(EPS_END, EPS_START - steps_done / EPS_DECAY)
        else:
            epsilon = EPS_START
        
        action = select_action(state, policy_model, epsilon, n_actions, device)
        for _ in range(30): 
            frame, reward, is_done, _ = env.step(action)
            reward = transform_reward(reward)
            total_reward += reward
            if is_done:
                break

        frame = preprocess(frame)
        frame_q.append(frame)

        if len(frame_q) < 4:
            continue
        
        next_state = np.stack(frame_q, axis=0)
        rm.push(state, action, next_state, reward, is_done) 
        state = next_state
        if is_done:
            print('Reward: {}'.format(total_reward))
            # print('EPS:    {}'.format(epsilon))
            state, frame_q = env_reset(env, no_op_steps)
            total_rewards.append(total_reward)
            total_reward = 0
            
            mean_reward = 0
            if n_episodes > 100:
                mean_reward = np.mean(total_rewards[-100:])

            if best_reward is None or mean_reward > best_reward:
                torch.save(policy_model.state_dict(),  "./checkpoints/BreakoutDeterministic-v4-best.dat")
                best_reward = mean_reward
                print('Best reward: {}'.format(best_reward))

            if best_reward > 50 and n_episodes > 10:
                print('Finish!')
                break

        if steps_done % TARGET_UPDATE == 0:
            target_model.load_state_dict(policy_model.state_dict())

        optimize_model(rm, BATCH_SIZE, policy_model, target_model, optimizer, device, GAMMA)
    env.close()

if __name__ == '__main__':
    main()
