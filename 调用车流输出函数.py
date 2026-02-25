"""
简单测试代码 - 直接调用流量生成函数
"""

# 方法1: 直接导入并使用
from generate_random_vehicle_traffic_flow import generate_traffic_flow_for_training

# 定义总训练轮次
total_episodes = 5

print("🚦 开始测试交通流量生成...")
print(f"将生成 {total_episodes} 轮不同的流量")

for episode in range(total_episodes):
    print(f"\n📋 正在生成第 {episode} 轮流量...")
    
    # 调用流量生成函数
    stats = generate_traffic_flow_for_training(
        episode=episode,
        output_file="调用车流输出函数.rou.xml",
        verbose=False  # 设为True可以看到详细信息
    )
    
    print(f"✅ 第 {episode} 轮流量生成完成!")

print("\n🎯 所有测试完成!")
print("\n💡 提示: 在真正的训练循环中，将 verbose=False/True  以提高效率")