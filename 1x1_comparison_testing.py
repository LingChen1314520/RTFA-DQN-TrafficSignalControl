import os
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# 配置参数
# ==========================================
class TestConfig:
    # 仿真文件
    NET_FILE = "1x1_testing.net.xml"
    ROUTE_FILE = "1x1_testing.rou.xml" 
    SUMO_CFG = "1x1_testing.sumocfg"
    
    # 待测试的RL模型路径
    RL_MODELS = {
        "RTFA-DQN": "model_1x1_rtfa_dqn.pth",
        "Standard-DQN": "model_1x1_standard_dqn.pth"
    }
    
    # 测试规模
    START_SEED = 3000      # 测试起始种子
    NUM_EPISODES = 100     # 测试总轮数
    
    RESULT_DIR = "testing_1x1_comparison_results"
    SUMMARY_CSV = "total_performance_report.csv"

    USE_GUI = False
    TL_ID = "J0"
    MAX_STEPS = 10800 
    YELLOW_TIME = 3
    MIN_GREEN_TIME = 10
    MAX_GREEN_TIME = 50 

    # 8 相位定义 (RL使用)
    PHASES_8 = [
        "grrgGrgrrgGr", "gGrgrrgGrgrr", "grrgrGgrrgrG", "grGgrrgrGgrr",
        "grrgrrgrrgGG", "grrgGGgrrgrr", "grrgrrgGGgrr", "gGGgrrgrrgrr"
    ]
    
    # 4 相位定义 (Actuated顺序感应使用)
    PHASES_4 = ["grrgGrgrrgGr", "gGrgrrgGrgrr", "grrgrGgrrgrG", "grGgrrgrGgrr"]
    
    STATE_DIM = 8
    ACTION_DIM = 8

# ==========================================
# 神经网络模型
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

# ==========================================
# 指标收集器 (精简版)
# ==========================================
class MetricTracker:
    def __init__(self):
        self.veh_info = {}
        self.completed_vehs = []
        self.all_queues = []
        
    def update_step(self):
        current_time = traci.simulation.getTime()
        # 统计排队
        lanes = list(set(traci.trafficlight.getControlledLanes(TestConfig.TL_ID)))
        self.all_queues.append(np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes]))
        
        # 统计车辆行程
        vehs = traci.vehicle.getIDList()
        for v in vehs:
            if v not in self.veh_info:
                self.veh_info[v] = {'enter': current_time, 'wait': 0}
            self.veh_info[v]['wait'] = traci.vehicle.getAccumulatedWaitingTime(v)
            self.veh_info[v]['dist'] = traci.vehicle.getDistance(v)
            
        for v in traci.simulation.getArrivedIDList():
            if v in self.veh_info:
                tt = current_time - self.veh_info[v]['enter']
                ds = self.veh_info[v]['dist']
                wt = self.veh_info[v]['wait']
                self.completed_vehs.append([tt, ds/tt if tt>0 else 0, wt])
                del self.veh_info[v]

    def get_results(self):
        if not self.completed_vehs: return 0, 0, 0, 0
        res = np.array(self.completed_vehs)
        return np.mean(res[:,0]), np.mean(res[:,1]), np.mean(self.all_queues), np.mean(res[:,2])

# ==========================================
# 仿真核心逻辑
# ==========================================
def run_simulation(mode, rl_model=None):
    sumo_binary = checkBinary('sumo-gui' if TestConfig.USE_GUI else 'sumo')
    traci.start([sumo_binary, "-c", TestConfig.SUMO_CFG, "--no-step-log", "true"])
    
    tracker = MetricTracker()
    controlled_lanes = traci.trafficlight.getControlledLanes(TestConfig.TL_ID)
    
    step = 0
    current_phase_idx = 0
    green_timer = 0

    while traci.simulation.getMinExpectedNumber() > 0 and step < TestConfig.MAX_STEPS:
        # --- 1. RL 模式 (RTFA-DQN 或 Standard-DQN) ---
        if rl_model is not None:
            if step % (TestConfig.MIN_GREEN_TIME + TestConfig.YELLOW_TIME) == 0:
                # 获取状态
                state = np.zeros(8)
                for i, p in enumerate(TestConfig.PHASES_8):
                    q = sum([traci.lane.getLastStepHaltingNumber(controlled_lanes[idx]) 
                            for idx, c in enumerate(p) if c == 'G'])
                    state[i] = q / p.count('G')
                
                with torch.no_grad():
                    action = rl_model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                
                if action != current_phase_idx:
                    yellow = "".join(['y' if c == 'G' else c for c in TestConfig.PHASES_8[current_phase_idx]])
                    traci.trafficlight.setRedYellowGreenState(TestConfig.TL_ID, yellow)
                    for _ in range(TestConfig.YELLOW_TIME):
                        traci.simulationStep(); tracker.update_step(); step += 1
                
                traci.trafficlight.setRedYellowGreenState(TestConfig.TL_ID, TestConfig.PHASES_8[action])
                current_phase_idx = action
                for _ in range(TestConfig.MIN_GREEN_TIME):
                    traci.simulationStep(); tracker.update_step(); step += 1
            else:
                traci.simulationStep(); tracker.update_step(); step += 1

        # --- 2. Actuated 顺序感应模式 ---
        elif mode == "Actuated":
            # 简化感应逻辑：如果当前相位没车且过了最小绿灯，或达到最大绿灯，则切换
            q_current = sum([traci.lane.getLastStepHaltingNumber(controlled_lanes[idx]) 
                            for idx, c in enumerate(TestConfig.PHASES_4[current_phase_idx]) if c == 'G'])
            
            if (green_timer >= TestConfig.MAX_GREEN_TIME) or (green_timer >= TestConfig.MIN_GREEN_TIME and q_current == 0):
                next_phase = (current_phase_idx + 1) % 4
                yellow = "".join(['y' if c == 'G' else c for c in TestConfig.PHASES_4[current_phase_idx]])
                traci.trafficlight.setRedYellowGreenState(TestConfig.TL_ID, yellow)
                for _ in range(TestConfig.YELLOW_TIME):
                    traci.simulationStep(); tracker.update_step(); step += 1
                
                current_phase_idx = next_phase
                traci.trafficlight.setRedYellowGreenState(TestConfig.TL_ID, TestConfig.PHASES_4[current_phase_idx])
                green_timer = 0
            
            traci.simulationStep(); tracker.update_step(); green_timer += 1; step += 1

        # --- 3. Static 静态模式 ---
        else:
            traci.simulationStep(); tracker.update_step(); step += 1

    metrics = tracker.get_results()
    traci.close()
    return metrics

# ==========================================
# 主循环
# ==========================================
if __name__ == "__main__":
    if not os.path.exists(TestConfig.RESULT_DIR): os.makedirs(TestConfig.RESULT_DIR)
    summary_path = os.path.join(TestConfig.RESULT_DIR, TestConfig.SUMMARY_CSV)
    
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Seed', 'Mode', 'ATT', 'ATS', 'AQL', 'AWT'])

    # 预加载模型
    models = {}
    for name, path in TestConfig.RL_MODELS.items():
        if os.path.exists(path):
            m = DQN(TestConfig.STATE_DIM, TestConfig.ACTION_DIM)
            m.load_state_dict(torch.load(path))
            m.eval()
            models[name] = m

    for ep in range(1, TestConfig.NUM_EPISODES + 1):
        seed = TestConfig.START_SEED + ep
        print(f"\n🚀 Episode {ep}/{TestConfig.NUM_EPISODES} | Seed: {seed}")
        
        # 为本轮生成统一的车流
        generate_traffic_flow_for_training(seed, TestConfig.ROUTE_FILE)

        # 依次运行四种测试
        test_list = [
            ("RTFA-DQN", models.get("RTFA-DQN")),
            ("Standard-DQN", models.get("Standard-DQN")),
            ("Actuated", None),
            ("Static", None)
        ]

        for mode_name, model_obj in test_list:
            if mode_name.endswith("DQN") and model_obj is None:
                continue # 跳过未找到的模型
                
            att, ats, aql, awt = run_simulation(mode_name, model_obj)
            
            with open(summary_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([ep, seed, mode_name, f"{att:.2f}", f"{ats:.2f}", f"{aql:.2f}", f"{awt:.2f}"])
            
            print(f"   [{mode_name:12}] ATT: {att:>7.2f} | AQL: {aql:>6.2f}")

    print(f"\n✅ 测试圆满完成！汇总报告见: {summary_path}")