import random
from typing import List, Tuple, Dict, Union
import numpy as np


# ========== 配置参数 ==========
class TrafficConfig:
    """交通流量配置类"""
    
    # 路口结构：顺序严格：直 -> 左 -> 右
    ROUTES = {
        "E1": [("E1", "E7"), ("E1", "E8"), ("E1", "E6")],
        "E2": [("E2", "E8"), ("E2", "E5"), ("E2", "E7")],
        "E3": [("E3", "E5"), ("E3", "E6"), ("E3", "E8")],
        "E4": [("E4", "E6"), ("E4", "E7"), ("E4", "E5")],
    }
    
    # 基础进口道比例（入口权重）
    BASE_ENTRY_RATIO = {
        "E1": 0.25,
        "E3": 0.25,
        "E2": 0.25,
        "E4": 0.25
    }
    
    # 基础转向比例（直行，左转，右转）
    BASE_TURN_RATIO = {
        "E1": [0.45, 0.30, 0.25],
        "E2": [0.45, 0.30, 0.25],
        "E3": [0.45, 0.30, 0.25],
        "E4": [0.45, 0.30, 0.25]
    }
    
    # 转向标签映射
    TURN_LABELS = ["S", "L", "R"]
    TURN_NAMES = ["直行", "左转", "右转"]
    
    # 三个时段总流量配置：基于3600的饱和度百分比
    TIME_PERIODS = [
        {"start": 0, "end": 3600, "total": 1080, "sat_level": 30, "name": "低流"},
        {"start": 3600, "end": 7200, "total": 2160, "sat_level": 60, "name": "中流"},
        {"start": 7200, "end": 10800, "total": 2880, "sat_level": 80, "name": "高流"}
    ]
    
    # 随机扰动参数 - 进口道比例扰动范围调整为±100%，其他保持±20%
    RANDOM_CONFIG = {
        "entry_ratio": {
            "noise_range": 2.0,    # 进口道比例扰动范围 (±100%)
            "min_value": 0.05,     # 最小比例值
        },
        "turn_ratio": {
            "noise_range": 0.20,   # 转向比例扰动范围 (±20%)
            "min_value": 0.08,     # 最小转向比例
        },
        "period_flow": {
            "noise_range": 0.20,   # 时段总流量扰动范围 (±20%)
            "min_multiplier": 0.80,  # 最小流量乘数
            "max_multiplier": 1.20,  # 最大流量乘数
        },
        "distribution": {
            "use_normal": True,    # 使用正态分布
            "bias_strength": 0.3,  # 偏向性强度
            "period_correlation": True,  # 时段间相关性
        }
    }


class RandomDisturbance:
    """随机扰动生成器"""
    
    def __init__(self, config: TrafficConfig = None):
        self.config = config or TrafficConfig()
        # 保存状态以实现时段间的相关性
        self.previous_disturbances = {
            "entry": None,
            "turn": None,
            "flow": None
        }
    
    def add_entry_ratio_disturbance(self, base_ratio: Dict[str, float], 
                                    period_idx: int = 0,
                                    correlation: bool = True) -> Dict[str, float]:
        """
        为进口道比例添加随机扰动（±100%范围）
        
        Args:
            base_ratio: 基础进口道比例
            period_idx: 时段索引，用于时段相关性
            correlation: 是否与上一时段相关
            
        Returns:
            添加扰动后的进口道比例
        """
        noise_range = self.config.RANDOM_CONFIG["entry_ratio"]["noise_range"]
        min_value = self.config.RANDOM_CONFIG["entry_ratio"]["min_value"]
        use_normal = self.config.RANDOM_CONFIG["distribution"]["use_normal"]
        bias_strength = self.config.RANDOM_CONFIG["distribution"]["bias_strength"]
        
        disturbed_ratio = {}
        
        # 检查是否需要相关性
        if correlation and self.previous_disturbances["entry"] is not None:
            base_noise = self.previous_disturbances["entry"]
            # 在上一时段扰动基础上进一步扰动
            correlation_strength = 0.6  # 相关性强度
        else:
            base_noise = {k: 0 for k in base_ratio.keys()}
            correlation_strength = 0
        
        for entrance, ratio in base_ratio.items():
            if use_normal:
                # 使用正态分布，更集中在中间值
                noise_std = noise_range / 3  # 使±100%范围覆盖约99%的值
                new_noise = random.normalvariate(0, noise_std)
                
                # 添加偏向性（例如早晚高峰特定方向流量增加）
                if period_idx == 2:  # 高峰时段
                    if entrance in ["E1", "E3"]:  # 假设主路入口更繁忙
                        new_noise += random.uniform(0, noise_range * bias_strength)
            else:
                # 均匀分布
                new_noise = random.uniform(-noise_range, noise_range)
            
            # 结合相关性
            total_noise = (1 - correlation_strength) * new_noise + correlation_strength * base_noise.get(entrance, 0)
            
            # 计算扰动值
            disturbed_value = ratio * (1 + total_noise)
            
            # 确保最小值
            disturbed_ratio[entrance] = max(min_value, disturbed_value)
        
        # 归一化
        total = sum(disturbed_ratio.values())
        normalized = {k: v/total for k, v in disturbed_ratio.items()}
        
        # 保存当前扰动状态用于下一时段
        self.previous_disturbances["entry"] = normalized
        
        return normalized
    
    def add_turn_ratio_disturbance(self, base_turn_ratio: Union[List[float], Dict[str, List[float]]],
                                   period_idx: int = 0,
                                   correlation: bool = True) -> Union[List[float], Dict[str, List[float]]]:
        """
        为转向比例添加随机扰动（±20%范围）
        
        Args:
            base_turn_ratio: 基础转向比例
            period_idx: 时段索引
            correlation: 是否与上一时段相关
            
        Returns:
            添加扰动后的转向比例
        """
        noise_range = self.config.RANDOM_CONFIG["turn_ratio"]["noise_range"]
        min_value = self.config.RANDOM_CONFIG["turn_ratio"]["min_value"]
        use_normal = self.config.RANDOM_CONFIG["distribution"]["use_normal"]
        
        # 如果是通用比例（所有进口道相同）
        if isinstance(base_turn_ratio, list):
            return self._disturb_single_turn_ratio(
                base_turn_ratio, period_idx, noise_range, min_value, use_normal, correlation
            )
        
        # 如果是各进口道独立比例
        elif isinstance(base_turn_ratio, dict):
            disturbed_ratios = {}
            for entrance, ratio_list in base_turn_ratio.items():
                # 为每个进口道添加不同的扰动
                disturbed_ratios[entrance] = self._disturb_single_turn_ratio(
                    ratio_list, period_idx, noise_range, min_value, use_normal, correlation
                )
            return disturbed_ratios
        
        return base_turn_ratio
    
    def _disturb_single_turn_ratio(self, base_ratio: List[float], 
                                   period_idx: int,
                                   noise_range: float,
                                   min_value: float,
                                   use_normal: bool,
                                   correlation: bool) -> List[float]:
        """为单个转向比例列表添加扰动"""
        disturbed = []
        
        # 检查相关性
        if correlation and self.previous_disturbances["turn"] is not None:
            base_noise = self.previous_disturbances["turn"]
        else:
            base_noise = [0] * len(base_ratio)
        
        for i, ratio in enumerate(base_ratio):
            # 根据不同转向类型使用不同的扰动策略
            turn_type = ["straight", "left", "right"][i]
            
            if use_normal:
                # 正态分布扰动
                noise_std = noise_range / 3
                base_noise_value = random.normalvariate(0, noise_std)
                
                # 根据不同转向类型调整扰动
                if turn_type == "straight":
                    # 直行扰动相对较小
                    base_noise_value *= 0.8
                elif turn_type == "left":
                    # 左转扰动较大
                    base_noise_value *= 1.2
                # 右转保持正常扰动
            else:
                base_noise_value = random.uniform(-noise_range, noise_range)
            
            # 添加时段特性
            if period_idx == 0:  # 早高峰
                if turn_type == "straight":
                    base_noise_value += random.uniform(0, noise_range * 0.1)
            elif period_idx == 2:  # 晚高峰
                if turn_type == "left":
                    base_noise_value += random.uniform(0, noise_range * 0.15)
            
            # 结合相关性
            correlation_strength = 0.5 if correlation else 0
            total_noise = (1 - correlation_strength) * base_noise_value + correlation_strength * base_noise[i] if i < len(base_noise) else base_noise_value
            
            disturbed_value = max(min_value, ratio + total_noise)
            disturbed.append(disturbed_value)
        
        # 归一化
        total = sum(disturbed)
        normalized = [v/total for v in disturbed]
        
        # 保存扰动状态
        self.previous_disturbances["turn"] = normalized
        
        return normalized
    
    def add_period_flow_disturbance(self, period_config: Dict,
                                    period_idx: int = 0,
                                    correlation: bool = True) -> Dict:
        """
        为时段总流量添加扰动（±20%范围）
        
        Args:
            period_config: 时段配置
            period_idx: 时段索引
            correlation: 是否与上一时段相关
            
        Returns:
            添加扰动后的时段配置
        """
        noise_range = self.config.RANDOM_CONFIG["period_flow"]["noise_range"]
        min_multiplier = self.config.RANDOM_CONFIG["period_flow"]["min_multiplier"]
        max_multiplier = self.config.RANDOM_CONFIG["period_flow"]["max_multiplier"]
        use_normal = self.config.RANDOM_CONFIG["distribution"]["use_normal"]
        
        modified_config = period_config.copy()
        
        # 检查相关性
        if correlation and self.previous_disturbances["flow"] is not None:
            base_noise = self.previous_disturbances["flow"]
        else:
            base_noise = 0
        
        if use_normal:
            # 正态分布扰动
            noise_std = noise_range / 3
            new_noise = random.normalvariate(0, noise_std)
        else:
            new_noise = random.uniform(-noise_range, noise_range)
        
        # 添加时段特性
        if period_idx == 0:  # 早高峰，流量可能更高
            new_noise += random.uniform(0, noise_range * 0.1)
        elif period_idx == 2:  # 晚高峰，流量波动更大
            new_noise += random.uniform(-noise_range * 0.05, noise_range * 0.1)
        
        # 结合相关性
        correlation_strength = 0.4 if correlation else 0
        total_noise = (1 - correlation_strength) * new_noise + correlation_strength * base_noise
        
        # 计算流量乘数
        flow_multiplier = 1 + total_noise
        flow_multiplier = max(min_multiplier, min(max_multiplier, flow_multiplier))
        
        # 应用扰动
        original_flow = period_config["total"]
        disturbed_flow = int(original_flow * flow_multiplier)
        
        modified_config["total"] = disturbed_flow
        modified_config["original_total"] = original_flow
        modified_config["multiplier"] = flow_multiplier
        modified_config["noise"] = total_noise
        
        # 保存扰动状态
        self.previous_disturbances["flow"] = total_noise
        
        return modified_config


class TrafficFlowGenerator:
    """交通流量生成器"""
    
    def __init__(self, config: TrafficConfig = None):
        self.config = config or TrafficConfig()
        self.randomizer = RandomDisturbance(config)
        
        # 记录生成的流量统计
        self.generated_stats = {
            "periods": [],
            "entry_ratios": [],
            "turn_ratios": [],
            "detailed_stats": []
        }
    
    def generate_flow_xml(self, output_filename: str = "1x1_training.rou.xml", 
                          seed: int = None, verbose: bool = False) -> Dict:
        """
        生成带随机扰动的SUMO流量XML文件
        
        Args:
            output_filename: 输出文件名
            seed: 随机种子，用于复现结果
            verbose: 是否打印详细统计信息
            
        Returns:
            生成的流量统计信息字典
        """
        
        # 设置随机种子
        if seed is not None:
            random.seed(seed)
            if verbose:
                print(f"📊 使用随机种子: {seed}")
        
        # 重置统计
        self.generated_stats = {
            "periods": [],
            "entry_ratios": [],
            "turn_ratios": [],
            "detailed_stats": []
        }
        
        # 重置随机器状态
        self.randomizer.previous_disturbances = {
            "entry": None,
            "turn": None,
            "flow": None
        }
        
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '',
            '<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
            '        xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">',
            ''
        ]
        
        # 添加文件头注释
        lines.append('    <!-- =========================================================== -->')
        lines.append('    <!-- 交通流量配置文件 (进口道比例扰动±100%，转向和流量扰动±20%) -->')
        lines.append('    <!-- 扰动类型：时段总流量、进口道比例、转向比例 -->')
        lines.append('    <!-- 随机种子：{} -->'.format(seed if seed is not None else "默认"))
        lines.append('    <!-- =========================================================== -->')
        lines.append('')
        
        # 为每个时段生成流量
        for period_idx, period in enumerate(self.config.TIME_PERIODS):
            # 为时段总流量添加扰动
            disturbed_period = self.randomizer.add_period_flow_disturbance(
                period,
                period_idx=period_idx,
                correlation=self.config.RANDOM_CONFIG["distribution"]["period_correlation"]
            )
            
            begin = disturbed_period["start"]
            end = disturbed_period["end"]
            total = disturbed_period["total"]
            sat_level = disturbed_period["sat_level"]
            level_name = disturbed_period["name"]
            original_total = disturbed_period["original_total"]
            multiplier = disturbed_period["multiplier"]
            
            # 保存统计信息
            period_stat = {
                "period": period_idx + 1,
                "name": level_name,
                "original_flow": original_total,
                "disturbed_flow": total,
                "multiplier": multiplier,
                "change_percent": ((total - original_total) / original_total) * 100
            }
            self.generated_stats["periods"].append(period_stat)
            
            # 时段注释
            lines.append('    <!-- =========================================================== -->')
            lines.append(f'    <!-- 第{period_idx+1}小时：{level_name}时段 (扰动配置：进口道±100%，其他±20%) -->')
            lines.append('    <!-- =========================================================== -->')
            lines.append(f'    <!-- 基础流量：{original_total} veh/h ({sat_level}%饱和度) -->')
            lines.append(f'    <!-- 扰动后流量：{total} veh/h (乘数：{multiplier:.3f}) -->')
            lines.append(f'    <!-- 变化：{((total - original_total) / original_total) * 100:+.1f}% -->')
            lines.append("")
            
            # 为进口道比例添加扰动（±100%范围）
            disturbed_entry_ratio = self.randomizer.add_entry_ratio_disturbance(
                self.config.BASE_ENTRY_RATIO,
                period_idx=period_idx,
                correlation=self.config.RANDOM_CONFIG["distribution"]["period_correlation"]
            )
            
            # 保存进口道比例统计
            self.generated_stats["entry_ratios"].append({
                "period": period_idx + 1,
                "ratios": disturbed_entry_ratio.copy()
            })
            
            # 为转向比例添加扰动（±20%范围）
            disturbed_turn_ratio = self.randomizer.add_turn_ratio_disturbance(
                self.config.BASE_TURN_RATIO,
                period_idx=period_idx,
                correlation=self.config.RANDOM_CONFIG["distribution"]["period_correlation"]
            )
            
            # 保存转向比例统计
            self.generated_stats["turn_ratios"].append({
                "period": period_idx + 1,
                "ratios": disturbed_turn_ratio.copy()
            })
            
            # 详细统计
            detailed_period_stats = {
                "period": period_idx + 1,
                "total_flow": total,
                "entries": {}
            }
            
            # 生成流量XML
            total_generated = 0
            entry_generated = {}
            
            for entrance, entry_ratio in disturbed_entry_ratio.items():
                entrance_total = int(total * entry_ratio)
                total_generated += entrance_total
                entry_generated[entrance] = entrance_total
                
                lines.append(f'    <!-- {entrance}入口：{entrance_total} veh/h ({entry_ratio*100:.1f}%) -->')
                
                # 获取该进口道的转向比例
                if isinstance(disturbed_turn_ratio, dict):
                    turn_ratio_for_entrance = disturbed_turn_ratio[entrance]
                else:
                    turn_ratio_for_entrance = disturbed_turn_ratio
                
                # 详细统计
                detailed_entry_stats = {
                    "total": entrance_total,
                    "ratio": entry_ratio,
                    "turns": {}
                }
                
                # 为每个转向生成流量
                for i, (frm, to) in enumerate(self.config.ROUTES[entrance]):
                    turn_flow = int(entrance_total * turn_ratio_for_entrance[i])
                    turn_type = self.config.TURN_LABELS[i]
                    turn_name = self.config.TURN_NAMES[i]
                    base_turn_ratio = self.config.BASE_TURN_RATIO["E1"][i] if isinstance(self.config.BASE_TURN_RATIO, dict) else self.config.BASE_TURN_RATIO[i]
                    turn_change = ((turn_ratio_for_entrance[i] - base_turn_ratio) / base_turn_ratio) * 100
                    
                    flow_line = (f'    <flow id="h{period_idx+1}_{entrance}_{turn_type}" '
                               f'begin="{begin}" end="{end}" '
                               f'from="{frm}" to="{to}" '
                               f'vehsPerHour="{turn_flow}" />  '
                               f'<!-- {turn_name}：{turn_ratio_for_entrance[i]*100:.1f}% (基准：{base_turn_ratio*100:.1f}%，变化：{turn_change:+.1f}%) -->')
                    lines.append(flow_line)
                    
                    # 记录详细统计
                    detailed_entry_stats["turns"][turn_name] = {
                        "flow": turn_flow,
                        "ratio": turn_ratio_for_entrance[i],
                        "base_ratio": base_turn_ratio,
                        "change": turn_change
                    }
                
                lines.append("")
                detailed_period_stats["entries"][entrance] = detailed_entry_stats
            
            # 添加流量总和验证
            flow_diff = total - total_generated
            lines.append(f'    <!-- 流量验证 -->')
            lines.append(f'    <!-- 目标流量：{total} veh/h -->')
            lines.append(f'    <!-- 生成流量：{total_generated} veh/h -->')
            if flow_diff != 0:
                lines.append(f'    <!-- 差值：{flow_diff} veh/h (因四舍五入产生) -->')
            lines.append("")
            
            # 保存详细统计
            self.generated_stats["detailed_stats"].append(detailed_period_stats)
        
        lines.append("</routes>")
        
        # 写入文件
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        if verbose:
            print(f"✅ SUMO流量文件已生成: {output_filename}")
            self._print_statistics(seed)
        
        return self.generated_stats
    
    def _print_statistics(self, seed: int = None) -> None:
        """打印生成的流量统计信息"""
        print("\n" + "=" * 80)
        print(f"交通流量生成统计信息 (随机种子: {seed if seed is not None else '默认'})")
        print("=" * 80)
        
        # 打印配置摘要
        print("\n📊 扰动配置摘要:")
        print("-" * 40)
        print(f"  进口道比例扰动范围: ±{self.config.RANDOM_CONFIG['entry_ratio']['noise_range']*100:.0f}%")
        print(f"  转向比例扰动范围: ±{self.config.RANDOM_CONFIG['turn_ratio']['noise_range']*100:.0f}%")
        print(f"  时段流量扰动范围: ±{self.config.RANDOM_CONFIG['period_flow']['noise_range']*100:.0f}%")
        print(f"  使用正态分布: {'是' if self.config.RANDOM_CONFIG['distribution']['use_normal'] else '否'}")
        print(f"  时段相关性: {'是' if self.config.RANDOM_CONFIG['distribution']['period_correlation'] else '否'}")
        
        # 打印时段流量扰动
        print("\n📈 时段总流量扰动 (±20%):")
        print("-" * 50)
        print("  时段 | 基准流量 | 扰动后流量 | 乘数   | 变化百分比")
        print("  ----|----------|------------|--------|------------")
        for stat in self.generated_stats["periods"]:
            print(f"  第{stat['period']}小时 | {stat['original_flow']:>8} | {stat['disturbed_flow']:>10} | {stat['multiplier']:>6.3f} | {stat['change_percent']:>+8.1f}%")
        
        # 打印进口道比例扰动
        print("\n🚗 进口道比例扰动 (±100%):")
        print("-" * 50)
        
        for period_idx, entry_stat in enumerate(self.generated_stats["entry_ratios"]):
            print(f"\n  📍 时段{entry_stat['period']}:")
            print("     入口 | 基准比例 | 扰动后比例 | 变化百分比")
            print("     ----|----------|------------|------------")
            
            for entrance, base_ratio in self.config.BASE_ENTRY_RATIO.items():
                disturbed_ratio = entry_stat["ratios"][entrance]
                change = ((disturbed_ratio - base_ratio) / base_ratio) * 100
                print(f"     {entrance} | {base_ratio*100:>7.1f}% | {disturbed_ratio*100:>10.1f}% | {change:>+10.1f}%")
        
        # 打印转向比例扰动
        print("\n🔄 转向比例扰动 (±20%):")
        print("-" * 50)
        
        # 获取基础转向比例
        base_turn = self.config.BASE_TURN_RATIO["E1"]
        
        for period_idx, turn_stat in enumerate(self.generated_stats["turn_ratios"]):
            print(f"\n  ⏱️ 时段{turn_stat['period']}:")
            
            if isinstance(turn_stat["ratios"], dict):
                # 各进口道独立比例
                for entrance, disturbed_ratio in turn_stat["ratios"].items():
                    print(f"\n    🚦 {entrance}入口:")
                    print("        转向 | 基准比例 | 扰动后比例 | 变化百分比")
                    print("        ----|----------|------------|------------")
                    
                    for i, (turn_name, base_ratio) in enumerate(zip(self.config.TURN_NAMES, base_turn)):
                        change = ((disturbed_ratio[i] - base_ratio) / base_ratio) * 100
                        print(f"        {turn_name} | {base_ratio*100:>7.1f}% | {disturbed_ratio[i]*100:>10.1f}% | {change:>+10.1f}%")
            else:
                # 统一比例
                disturbed_ratio = turn_stat["ratios"]
                print("        转向 | 基准比例 | 扰动后比例 | 变化百分比")
                print("        ----|----------|------------|------------")
                
                for i, (turn_name, base_ratio) in enumerate(zip(self.config.TURN_NAMES, base_turn)):
                    change = ((disturbed_ratio[i] - base_ratio) / base_ratio) * 100
                    print(f"        {turn_name} | {base_ratio*100:>7.1f}% | {disturbed_ratio[i]*100:>10.1f}% | {change:>+10.1f}%")
        
        print("\n" + "=" * 80)


# ========== 核心封装函数 ==========

def generate_traffic_flow_for_training(episode: int, output_file: str = "1x1_training.rou.xml", 
                                       verbose: bool = False) -> Dict:
    """
    为强化学习训练生成交通流文件
    
    Args:
        episode: 当前训练轮次（用作随机种子）
        output_file: 输出文件路径
        verbose: 是否打印详细信息
        
    Returns:
        生成的流量统计信息字典
        
    使用示例：
        # 在强化学习训练循环中
        for episode in range(total_episodes):
            # 为当前轮次生成交通流文件
            stats = generate_traffic_flow_for_training(episode, "1x1_training.rou.xml")
            
            # 运行SUMO仿真
            # ... 你的仿真代码 ...
    """
    if verbose:
        print(f"🚦 开始为第 {episode} 轮训练生成交通流量...")
        print(f"📝 使用随机种子: {episode}")
        print("📊 扰动范围配置:")
        print("   - 进口道比例: ±100% (大幅扰动)")
        print("   - 转向比例: ±20% (适度扰动)")
        print("   - 时段流量: ±20% (适度扰动)")
    
    # 创建配置实例
    config = TrafficConfig()
    
    # 创建生成器
    generator = TrafficFlowGenerator(config)
    
    # 生成流量文件，使用episode作为随机种子
    stats = generator.generate_flow_xml(output_file, seed=episode, verbose=verbose)
    
    if verbose:
        print(f"\n✅ 第 {episode} 轮流量生成完成!")
        print(f"📁 输出文件: {output_file}")
    
    return stats