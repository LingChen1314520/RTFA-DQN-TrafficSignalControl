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

# ==========================================
# 配置参数
# ==========================================
class TestConfig:
    USE_GUI = False  # True 则弹出仿真界面，False 则在后台高速运行
    NET_FILE = "hangzhou_4x4_gudang_18041610_1h.net.xml"
    SUMO_CFG = "hangzhou_4x4_gudang_18041610_1h.sumocfg"
    MODEL_PATH = "model_1x1_dqn.pth"
    RESULT_DIR = "testing_hangzhou_4x4_gudang_18041610_1h_results"
    
    MAX_STEPS = 3600
    YELLOW_TIME = 3
    MIN_GREEN_TIME = 10 
    MAX_GREEN_TIME = 50 
    
    # 8 相位定义
    PHASES = [
        "grrgGrgrrgGr", "gGrgrrgGrgrr", "grrgrGgrrgrG", "grGgrrgrGgrr",
        "grrgrrgrrgGG", "grrgGGggrgrr", "grrgrrgGGgrr", "gGGgrrgrrgrr"
    ]
    
    # 4 相位定义
    PHASES_4 = [
        "grrgGrgrrgGr", "gGrgrrgGrgrr", "grrgrGgrrgrG", "grGgrrgrGgrr"
    ]
    
    STATE_DIM = 8
    ACTION_DIM = 4

# ==========================================
# DQN 模型结构
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
# 核心指标收集类 (计算 ATT, ATS, AQL, AWT)
# ==========================================
class MetricTracker:
    def __init__(self, all_ts):
        self.all_ts = all_ts
        self.veh_info = {}       
        self.completed_vehs = [] 
        self.all_queues_history = []
        
    def update_step(self):
        current_time = traci.simulation.getTime()
        vehs = traci.vehicle.getIDList()
        
        for v in vehs:
            if v not in self.veh_info:
                self.veh_info[v] = {
                    'enter': current_time,
                    'wait': traci.vehicle.getAccumulatedWaitingTime(v),
                    'dist': traci.vehicle.getDistance(v)
                }
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
        
        step_queues = []
        for ts in self.all_ts:
            lanes = list(set(traci.trafficlight.getControlledLanes(ts)))
            step_queues.extend([traci.lane.getLastStepHaltingNumber(l) for l in lanes])
        self.all_queues_history.append(np.mean(step_queues) if step_queues else 0)

    def calculate_final_metrics(self):
        if not self.completed_vehs:
            return 0, 0, 0, 0
        arr = np.array(self.completed_vehs)
        att = np.mean(arr[:, 0])  
        ats = np.mean(arr[:, 1])  
        aql = np.mean(self.all_queues_history) 
        awt = np.mean(arr[:, 2])  
        return att, ats, aql, awt

# ==========================================
# 工具函数
# ==========================================
def get_node_metrics(ts_id):
    """获取单个路口的平均排队和平均速度"""    
    lanes = list(set(traci.trafficlight.getControlledLanes(ts_id)))
    q = np.mean([traci.lane.getLastStepHaltingNumber(l) for l in lanes]) if lanes else 0
    s = np.mean([traci.lane.getLastStepMeanSpeed(l) for l in lanes]) if lanes else 0
    return q, s

def get_node_state(ts_id, use_4_phases=False):
    """根据模式获取排队状态"""
    controlled_lanes = traci.trafficlight.getControlledLanes(ts_id)
    # 根据标志选择使用的相位库
    current_phases = TestConfig.PHASES_4 if use_4_phases else TestConfig.PHASES
    
    state = np.zeros(len(current_phases))
    for i, phase_str in enumerate(current_phases):
        q_sum, count = 0, 0
        for idx, char in enumerate(phase_str):
            if char == 'G' and idx < len(controlled_lanes):
                q_sum += traci.lane.getLastStepHaltingNumber(controlled_lanes[idx])
                count += 1
        state[i] = q_sum / count if count > 0 else 0
    return state

# ==========================================
# 仿真主程序
# ==========================================
def run_test(mode="RL"):
    print(f"\n>>> 启动模式: {mode}")
    sumo_binary = checkBinary('sumo-gui' if TestConfig.USE_GUI else 'sumo')
    traci.start([sumo_binary, "-c", TestConfig.SUMO_CFG, "--no-step-log", "true"])
    
    all_ts = traci.trafficlight.getIDList()
    tracker = MetricTracker(all_ts)
    
    # 状态机管理
    ts_data = {ts: {"last_phase": 0, "timer": 0, "is_yellow": False, "pending_action": 0} for ts in all_ts}
    
    model = None
    if mode == "RL":
        # RL 模式输出维度固定为 8 以适配模型
        model = DQN(TestConfig.STATE_DIM, 8) 
        if os.path.exists(TestConfig.MODEL_PATH):
            model.load_state_dict(torch.load(TestConfig.MODEL_PATH))
            print(f"成功加载模型: {TestConfig.MODEL_PATH}")
        model.eval()
    
    global_log = []
    node_logs = {ts: [] for ts in all_ts}

    for step in range(TestConfig.MAX_STEPS):
        for ts_id in all_ts:
            # 1. RL 控制
            if mode == "RL":
                if ts_data[ts_id]["timer"] <= 0:
                    state = get_node_state(ts_id, use_4_phases=False)
                    with torch.no_grad():
                        action = model(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                    
                    curr_p = ts_data[ts_id]["last_phase"]
                    if action != curr_p:
                        yellow = "".join(['y' if c == 'G' else c for c in TestConfig.PHASES[curr_p]])
                        traci.trafficlight.setRedYellowGreenState(ts_id, yellow)
                        ts_data[ts_id]["timer"] = TestConfig.YELLOW_TIME + TestConfig.MIN_GREEN_TIME
                        ts_data[ts_id]["is_yellow"] = True
                        ts_data[ts_id]["pending_action"] = action
                    else:
                        traci.trafficlight.setRedYellowGreenState(ts_id, TestConfig.PHASES[action])
                        ts_data[ts_id]["timer"] = TestConfig.MIN_GREEN_TIME
                        ts_data[ts_id]["is_yellow"] = False
                
                if ts_data[ts_id]["is_yellow"] and ts_data[ts_id]["timer"] == TestConfig.MIN_GREEN_TIME:
                    traci.trafficlight.setRedYellowGreenState(ts_id, TestConfig.PHASES[ts_data[ts_id]["pending_action"]])
                    ts_data[ts_id]["last_phase"] = ts_data[ts_id]["pending_action"]
                    ts_data[ts_id]["is_yellow"] = False
                
                ts_data[ts_id]["timer"] -= 1

            # 2. Actuated 顺序感应控制
            elif mode == "Actuated":
                curr_p = ts_data[ts_id]["last_phase"]
                if not ts_data[ts_id]["is_yellow"]:
                    ts_data[ts_id]["timer"] += 1
                    state = get_node_state(ts_id, use_4_phases=True)
                    current_q = state[curr_p]
                    
                    # 切换逻辑：必须满足 (无车且过最小绿) 或 (强制最大绿)
                    should_switch = (current_q == 0 and ts_data[ts_id]["timer"] >= TestConfig.MIN_GREEN_TIME) or \
                                   (ts_data[ts_id]["timer"] >= TestConfig.MAX_GREEN_TIME)
                    
                    if should_switch:
                        # 顺序寻找下一个有车相位
                        next_p = curr_p
                        for i in range(1, TestConfig.ACTION_DIM):
                            check_p = (curr_p + i) % TestConfig.ACTION_DIM
                            if state[check_p] > 0:
                                next_p = check_p
                                break
                        # 若都无车，则切换到下一个顺序相位
                        if next_p == curr_p: next_p = (curr_p + 1) % TestConfig.ACTION_DIM

                        # 进入黄灯切换流程
                        yellow = "".join(['y' if c == 'G' else c for c in TestConfig.PHASES_4[curr_p]])
                        traci.trafficlight.setRedYellowGreenState(ts_id, yellow)
                        ts_data[ts_id]["is_yellow"] = True
                        ts_data[ts_id]["timer"] = 0
                        ts_data[ts_id]["pending_action"] = next_p
                else:
                    ts_data[ts_id]["timer"] += 1
                    if ts_data[ts_id]["timer"] >= TestConfig.YELLOW_TIME:
                        # 黄灯结束，切换至新绿灯
                        new_p = ts_data[ts_id]["pending_action"]
                        traci.trafficlight.setRedYellowGreenState(ts_id, TestConfig.PHASES_4[new_p])
                        ts_data[ts_id]["last_phase"] = new_p
                        ts_data[ts_id]["is_yellow"] = False
                        ts_data[ts_id]["timer"] = 0
            # 3. Static 模式直接由 SUMO 配置文件驱动
        traci.simulationStep()
        tracker.update_step()
        
        # 统计收集
        all_vehs = traci.vehicle.getIDList()
        g_s = np.mean([traci.vehicle.getSpeed(v) for v in all_vehs]) if all_vehs else 0
        step_node_queues = []
        for ts in all_ts:
            nq, ns = get_node_metrics(ts)
            node_logs[ts].append([step, nq, ns])
            step_node_queues.append(nq)
        
        global_log.append([step, np.mean(step_node_queues) if step_node_queues else 0, g_s])

    att, ats, aql, awt = tracker.calculate_final_metrics()
    traci.close()
    return global_log, node_logs, (att, ats, aql, awt)

if __name__ == "__main__":
    if not os.path.exists(TestConfig.RESULT_DIR):
        os.makedirs(TestConfig.RESULT_DIR)
        
    evaluation_summary = []
    modes = ["RL", "Static", "Actuated"] #

    for m in modes:
        g_log, n_logs, macro = run_test(mode=m)
        evaluation_summary.append((m, macro))
        
        with open(os.path.join(TestConfig.RESULT_DIR, f"{m}_global_metrics.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'Avg_Queue', 'Avg_Speed'])
            writer.writerows(g_log)
            
        node_dir = os.path.join(TestConfig.RESULT_DIR, f"{m}_nodes_data")
        if not os.path.exists(node_dir): os.makedirs(node_dir)
        for ts, data in n_logs.items():
            with open(os.path.join(node_dir, f"node_{ts}.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Step', 'Queue_Length', 'Mean_Speed'])
                writer.writerows(data)

    print("\n" + "="*50)
    print("杭州古荡路网 (4x4) 仿真实验最终指标汇总")
    print("="*50)
    print(f"{'Mode':<10} | {'ATT (s)':<10} | {'ATS (m/s)':<10} | {'AQL (veh)':<10} | {'AWT (s)':<10}")
    print("-" * 60)
    for m, macro in evaluation_summary:
        print(f"{m:<10} | {macro[0]:<10.2f} | {macro[1]:<10.2f} | {macro[2]:<10.2f} | {macro[3]:<10.2f}")
    print("="*50)