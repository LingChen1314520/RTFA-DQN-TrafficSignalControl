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
    USE_GUI = False
    NET_FILE = "1x1_testing.net.xml"
    ROUTE_FILE = "1x1_testing.rou.xml" 
    SUMO_CFG = "1x1_testing.sumocfg"
    MODEL_PATH = "model_1x1_dqn.pth"
    RESULT_DIR = "testing_1x1_results"

    TL_ID = "J0"
    MAX_STEPS = 10800 
    YELLOW_TIME = 3
    MIN_GREEN_TIME = 10
    MAX_GREEN_TIME = 50 

    # 8 相位定义
    PHASES = [
        "grrgGrgrrgGr", "gGrgrrgGrgrr", "grrgrGgrrgrG", "grGgrrgrGgrr",
        "grrgrrgrrgGG", "grrgGGgrrgrr", "grrgrrgGGgrr", "gGGgrrgrrgrr"
    ]
    
    # 4 相位定义
    PHASES_4 = [
        "grrgGrgrrgGr", "gGrgrrgGrgrr", "grrgrGgrrgrG", "grGgrrgrGgrr"
    ]
    
    STATE_DIM = 8
    ACTION_DIM = 8

# ==========================================
# 核心指标收集类 (计算 ATT, ATS, AQL, AWT)
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

class MetricTracker:
    def __init__(self):
        self.veh_info = {}
        self.completed_vehs = []
        self.all_queues_history = []
    def update_step(self):
        current_time = traci.simulation.getTime()
        vehs = traci.vehicle.getIDList()
        for v in vehs:
            if v not in self.veh_info:
                self.veh_info[v] = {'enter': current_time, 'wait': traci.vehicle.getAccumulatedWaitingTime(v), 'dist': traci.vehicle.getDistance(v)}
            else:
                self.veh_info[v]['wait'] = traci.vehicle.getAccumulatedWaitingTime(v)
                self.veh_info[v]['dist'] = traci.vehicle.getDistance(v)
        arrived_vehs = traci.simulation.getArrivedIDList()
        for v in arrived_vehs:
            if v in self.veh_info:
                travel_time = current_time - self.veh_info[v]['enter']
                dist = self.veh_info[v]['dist']
                avg_speed = dist / travel_time if travel_time > 0 else 0
                wait_time = self.veh_info[v]['wait']
                self.completed_vehs.append([travel_time, avg_speed, wait_time])
                del self.veh_info[v]
        lanes = list(set(traci.trafficlight.getControlledLanes(TestConfig.TL_ID)))
        self.all_queues_history.append(np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes]) if lanes else 0)
    def calculate_final_metrics(self):
        if not self.completed_vehs: return 0, 0, 0, 0
        arr = np.array(self.completed_vehs)
        return np.mean(arr[:, 0]), np.mean(arr[:, 1]), np.mean(self.all_queues_history), np.mean(arr[:, 2])

# ==========================================
# 工具函数
# ==========================================
def get_state(tl_id, controlled_lanes, use_4_phases=False):
    phases_to_use = TestConfig.PHASES_4 if use_4_phases else TestConfig.PHASES
    state = np.zeros(len(phases_to_use))
    for i, phase_str in enumerate(phases_to_use):
        q_sum, count = 0, 0
        for idx, char in enumerate(phase_str):
            if char == 'G' and idx < len(controlled_lanes):
                q_sum += traci.lane.getLastStepHaltingNumber(controlled_lanes[idx])
                count += 1
        state[i] = q_sum / count if count > 0 else 0
    return state

def collect_metrics():
    controlled_lanes = traci.trafficlight.getControlledLanes(TestConfig.TL_ID)
    queues = [traci.lane.getLastStepHaltingNumber(lane) for lane in list(set(controlled_lanes))]
    speeds = [traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]
    return np.mean(queues) if queues else 0, np.mean(speeds) if speeds else 0

# ==========================================
# 仿真主程序
# ==========================================
def run_test(mode="RL"):
    print(f"--- 正在开始 {mode} 模式测试 ---")
    sumo_binary = checkBinary('sumo-gui' if TestConfig.USE_GUI else 'sumo')
    traci.start([sumo_binary, "-c", TestConfig.SUMO_CFG, "--route-files", TestConfig.ROUTE_FILE, "--no-step-log", "true", "--start", "true"])

    tl_id = TestConfig.TL_ID
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
    tracker = MetricTracker()

    model = None
    if mode == "RL":
        model = DQN(TestConfig.STATE_DIM, TestConfig.ACTION_DIM)
        if os.path.exists(TestConfig.MODEL_PATH):
            model.load_state_dict(torch.load(TestConfig.MODEL_PATH))
            model.eval()

    current_phase_index = 0
    step = 0
    data_log = []
    green_timer = 0 

    while traci.simulation.getMinExpectedNumber() > 0 and step < TestConfig.MAX_STEPS:

        # 1. RL 模式
        if mode == "RL" and step % (TestConfig.MIN_GREEN_TIME + TestConfig.YELLOW_TIME) == 0:
            state = get_state(tl_id, controlled_lanes, use_4_phases=False)
            with torch.no_grad():
                action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
            if action != current_phase_index:
                yellow = "".join(['y' if c == 'G' else c for c in TestConfig.PHASES[current_phase_index]])
                traci.trafficlight.setRedYellowGreenState(tl_id, yellow)
                for _ in range(TestConfig.YELLOW_TIME):
                    traci.simulationStep(); tracker.update_step()
                    q, v = collect_metrics(); data_log.append([step, q, v]); step += 1
            traci.trafficlight.setRedYellowGreenState(tl_id, TestConfig.PHASES[action])
            current_phase_index = action
            for _ in range(TestConfig.MIN_GREEN_TIME):
                traci.simulationStep(); tracker.update_step()
                q, v = collect_metrics(); data_log.append([step, q, v]); step += 1

        # 2. Actuated 顺序感应模式
        elif mode == "Actuated":
            state = get_state(tl_id, controlled_lanes, use_4_phases=True)
            current_q = state[current_phase_index]
            
            # 顺序切换逻辑：
            # 条件 A: 达到最大绿灯时长 -> 必须切
            # 条件 B: 达到最小绿灯时长 且 当前方向已排空 (q=0) -> 可以切
            if (green_timer >= TestConfig.MAX_GREEN_TIME) or \
               (green_timer >= TestConfig.MIN_GREEN_TIME and current_q == 0):
                
                # 严格按照 (0->1->2->3->0) 顺序
                next_phase = (current_phase_index + 1) % 4
                
                # 执行黄灯转换
                yellow = "".join(['y' if c == 'G' else c for c in TestConfig.PHASES_4[current_phase_index]])
                traci.trafficlight.setRedYellowGreenState(tl_id, yellow)
                for _ in range(TestConfig.YELLOW_TIME):
                    traci.simulationStep(); tracker.update_step()
                    q, v = collect_metrics(); data_log.append([step, q, v]); step += 1

                # 切换到下一顺序相位
                current_phase_index = next_phase
                traci.trafficlight.setRedYellowGreenState(tl_id, TestConfig.PHASES_4[current_phase_index])
                green_timer = 0 # 重置绿灯计时

            traci.simulationStep()
            tracker.update_step()
            green_timer += 1
            q, v = collect_metrics(); data_log.append([step, q, v]); step += 1

        # 3. Static 模式
        else:
            traci.simulationStep(); tracker.update_step()
            q, v = collect_metrics(); data_log.append([step, q, v]); step += 1

    traci.close()
    att, ats, aql, awt = tracker.calculate_final_metrics()
    return data_log, (att, ats, aql, awt)

if __name__ == "__main__":
    if not os.path.exists(TestConfig.RESULT_DIR): os.makedirs(TestConfig.RESULT_DIR)
    generate_traffic_flow_for_training(999, TestConfig.ROUTE_FILE)
    
    results = []
    for m in ["RL", "Static", "Actuated"]:
        log, macro = run_test(mode=m)
        results.append((m, log, macro))

    for name, data, macro in results:
        path = os.path.join(TestConfig.RESULT_DIR, f"testing_1x1_{name.lower()}_results.csv")
        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Avg_Queue', 'Avg_Speed'])
            writer.writerows(data)
        print(f"\n[{name} 模式]: ATT: {macro[0]:.2f}, ATS: {macro[1]:.2f}, AQL: {macro[2]:.2f}, AWT: {macro[3]:.2f}")