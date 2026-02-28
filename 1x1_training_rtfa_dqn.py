import os
import sys
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt

# 检查SUMO_HOME
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary
from generate_random_vehicle_traffic_flow import generate_traffic_flow_for_training

# ==========================================
# 配置参数 (RTFA-DQN 版本)
# ==========================================
class TrainingConfig:
    NET_FILE = "1x1_training.net.xml"
    ROUTE_FILE = "1x1_training.rou.xml"
    SUMO_CFG = "1x1_training.sumocfg"
    MODEL_SAVE_PATH = "model_1x1_rtfa_dqn.pth"
    LOG_DIR = "training_results"  # 指定保存数据的文件夹
    
    USE_GUI = False  # True 则弹出仿真界面，False 则在后台高速运行
    TOTAL_EPISODES =1000
    MAX_STEPS_PER_EPISODE = 10800
    
    TL_ID = "J0"
    YELLOW_TIME = 3
    MIN_GREEN_TIME = 10
    
    PHASES = [
        "grrgGrgrrgGr", "gGrgrrgGrgrr", "grrgrGgrrgrG", "grGgrrgrGgrr",
        "grrgrrgrrgGG", "grrgGGgrrgrr", "grrgrrgGGgrr", "gGGgrrgrrgrr"
    ]

    STATE_DIM = 8
    ACTION_DIM = 8
    BATCH_SIZE = 64
    GAMMA = 0.95
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 0.99
    TARGET_UPDATE = 10
    LEARNING_RATE = 0.00001
    MEMORY_SIZE = 20000

# ==========================================
# 深度强化学习模型
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=TrainingConfig.LEARNING_RATE)
        self.memory = deque(maxlen=TrainingConfig.MEMORY_SIZE)
        self.epsilon = TrainingConfig.EPS_START

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(state_t).argmax().item()

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < TrainingConfig.BATCH_SIZE:
            return None, None
        
        batch = random.sample(self.memory, TrainingConfig.BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)

        s = torch.FloatTensor(np.array(s)).to(self.device)
        a = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(np.array(ns)).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # Q值计算
        current_q = self.policy_net(s).gather(1, a)
        next_q = self.target_net(ns).max(1)[0].unsqueeze(1)
        target_q = r + (TrainingConfig.GAMMA * next_q * (1 - d))

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), current_q.mean().item()

# ==========================================
# 环境封装
# ==========================================
class SumoEnvironment:
    def __init__(self, episode):
        self.episode = episode
        self.tl_id = TrainingConfig.TL_ID
        self.phases = TrainingConfig.PHASES
        self.last_step_queue_lengths = {}

    def reset(self):
        generate_traffic_flow_for_training(self.episode, TrainingConfig.ROUTE_FILE)
        sumo_binary = checkBinary('sumo-gui' if TrainingConfig.USE_GUI else 'sumo')
        traci.start([sumo_binary, "-c", TrainingConfig.SUMO_CFG, "--no-step-log", "true"])
        self.controlled_lanes = traci.trafficlight.getControlledLanes(self.tl_id)
        self.incoming_lanes = list(set(self.controlled_lanes))
        self.last_step_queue_lengths = {lane: 0 for lane in self.incoming_lanes}
        traci.trafficlight.setRedYellowGreenState(self.tl_id, self.phases[0])
        self.current_phase_index = 0
        return self._get_state()

    def _get_state(self):
        state = np.zeros(TrainingConfig.STATE_DIM)
        for i, phase_str in enumerate(self.phases):
            q_sum, count = 0, 0
            for idx, char in enumerate(phase_str):
                if char == 'G':
                    q_sum += traci.lane.getLastStepHaltingNumber(self.controlled_lanes[idx])
                    count += 1
            state[i] = q_sum / count if count > 0 else 0
        return state

    def _compute_reward(self):
        # 获取所有进口道的总排队数
        total_queue = sum([traci.lane.getLastStepHaltingNumber(lane) for lane in self.incoming_lanes])
        
        # 归一化处理：除以车道数，得到平均每条道的排队惩罚
        reward = -(total_queue / len(self.incoming_lanes))
        return reward

    def step(self, action):
        if action != self.current_phase_index:
            yellow = "".join(['y' if c == 'G' else c for c in self.phases[self.current_phase_index]])
            traci.trafficlight.setRedYellowGreenState(self.tl_id, yellow)
            for _ in range(TrainingConfig.YELLOW_TIME): traci.simulationStep()
        
        traci.trafficlight.setRedYellowGreenState(self.tl_id, self.phases[action])
        self.current_phase_index = action
        for _ in range(TrainingConfig.MIN_GREEN_TIME): traci.simulationStep()
        
        return self._get_state(), self._compute_reward(), traci.simulation.getTime() >= TrainingConfig.MAX_STEPS_PER_EPISODE or traci.simulation.getMinExpectedNumber() <= 0

# ==========================================
# 执行训练并保存数据
# ==========================================
def run_training():
    # 创建保存目录
    if not os.path.exists(TrainingConfig.LOG_DIR):
        os.makedirs(TrainingConfig.LOG_DIR)
    
    # 准备CSV文件头
    csv_path = os.path.join(TrainingConfig.LOG_DIR, "training_stats_rtfa.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'Avg_Queue', 'Avg_Loss', 'Avg_Q_Value', 'Epsilon'])

    agent = DQNAgent(TrainingConfig.STATE_DIM, TrainingConfig.ACTION_DIM)
    
    for episode in range(1, TrainingConfig.TOTAL_EPISODES + 1):
        env = SumoEnvironment(episode)
        state = env.reset()
        
        ep_reward, ep_queue, ep_loss, ep_q = 0, [], [], []
        
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.push(state, action, reward, next_state, done)
            
            # 更新模型并记录 Loss 和 Q值
            loss, q_val = agent.update()
            if loss is not None:
                ep_loss.append(loss)
                ep_q.append(q_val)
            
            state = next_state
            ep_reward += reward
            ep_queue.append(np.mean(state))

        # Epsilon 衰减
        agent.epsilon = max(TrainingConfig.EPS_END, agent.epsilon * TrainingConfig.EPS_DECAY)
        
        if episode % TrainingConfig.TARGET_UPDATE == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        
        traci.close()

        # 计算本轮平均指标
        avg_q_len = np.mean(ep_queue)
        avg_loss = np.mean(ep_loss) if ep_loss else 0
        avg_q_val = np.mean(ep_q) if ep_q else 0

        # 保存到CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, ep_reward, avg_q_len, avg_loss, avg_q_val, agent.epsilon])

        print(f"Ep {episode} | Reward: {ep_reward:.1f} | Queue: {avg_q_len:.2f} | Loss: {avg_loss:.4f} | Q: {avg_q_val:.2f}")

        if episode % 20 == 0 or episode == TrainingConfig.TOTAL_EPISODES:
            agent.policy_net.to('cpu') # 转到CPU保存更通用
            torch.save(agent.policy_net.state_dict(), TrainingConfig.MODEL_SAVE_PATH)
            agent.policy_net.to(agent.device)

    print(f"训练完成，所有统计数据已保存至 {TrainingConfig.LOG_DIR} 文件夹。")

if __name__ == "__main__":
    run_training()