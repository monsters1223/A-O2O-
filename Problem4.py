"""
问题四：外卖冲击下的空间博弈与"云餐厅"选址决策

核心内容：
1. Hotelling双寡头博弈模型拓展
2. 考虑租金、平台抽成、距离摩擦
3. 双渠道竞争：线下堂食 vs 线上外卖
4. 纳什均衡分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==================== 参数设定 ====================
# 文汇路参数
STREET_LENGTH = 2500  # 文汇路长度（米）

# 成本参数
cost_params = {
    'base_cost': 8,          # 单位产品基础成本（元）
    'delivery_cost': 3,      # 配送成本（元/单）
    'platform_fee': 0.18,    # 平台抽成比例
}

# 租金参数（元/月）
rent_params = {
    'prime_location': 25000,   # 主街黄金地段租金（如1/2, 1/4, 3/4位置）
    'normal_location': 15000,  # 主街普通位置租金
    'back_alley': 5000,        # 背街云餐厅租金
}

# 消费者参数
consumer_params = {
    'density': 3,            # 消费者密度（人/米）- 实际有效客流约7500人/天
    'time_cost': 0.05,       # 时间成本（元/分钟）
    'walk_speed': 60,        # 步行速度（米/分钟）
    'price_sensitivity': 2.0, # 价格敏感度
    'offline_ratio': 0.4,    # 偏好线下消费者比例
    'online_ratio': 0.6,     # 偏好线上消费者比例
    'conversion_rate': 0.15, # 实际购买转化率（15%）
}

# 巨头参数（已占据1/2位置）
giant_params = {
    'location': 0.5,         # 位置（街道比例）
    'offline_price': 18,     # 线下价格
    'online_price': 20,      # 线上价格
    'type': 'traditional'    # 传统堂食巨头
}

# ==================== 效用函数 ====================
def calculate_utility(consumer_pos, shop_pos, price_offline, price_online, channel):
    """
    计算消费者效用
    
    参数：
    - consumer_pos: 消费者位置（0-1）
    - shop_pos: 店铺位置（0-1）
    - price_offline: 线下价格
    - price_online: 线上价格
    - channel: 'offline' 或 'online'
    
    返回：
    - utility: 效用值
    """
    if channel == 'offline':
        # 线下效用 = 基础效用 - 价格 - 距离成本
        distance = abs(consumer_pos - shop_pos) * STREET_LENGTH
        time_cost = distance / consumer_params['walk_speed'] * consumer_params['time_cost']
        utility = 100 - price_offline - time_cost
    else:
        # 线上效用 = 基础效用 - 价格 - 配送成本
        utility = 95 - price_online - cost_params['delivery_cost']
    
    return utility

def calculate_demand(shop_pos, price_offline, price_online, 
                    giant_pos, giant_offline, giant_online):
    """
    计算店铺需求（考虑竞争）
    
    参数：
    - shop_pos: 创业者店铺位置
    - price_offline: 创业者线下价格
    - price_online: 创业者线上价格
    - giant_pos: 巨头位置
    - giant_offline: 巨头线下价格
    - giant_online: 巨头线上价格
    
    返回：
    - demand_offline: 线下需求
    - demand_online: 线上需求
    """
    # 计算消费者选择边界
    # 线下边界：消费者对两家店线下效用无差异的位置
    # U1_off = U2_off => 100 - p1 - t*d1 = 100 - p2 - t*d2
    
    t = consumer_params['time_cost'] * STREET_LENGTH / consumer_params['walk_speed']
    
    # 线下市场份额（Hotelling模型）
    if abs(price_offline - giant_offline) >= t * abs(shop_pos - giant_pos):
        # 某一方完全占领市场
        if price_offline < giant_offline - t * abs(shop_pos - giant_pos):
            offline_share = 1.0
        else:
            offline_share = 0.0
    else:
        # 市场分割
        # 边界位置 x = (p2 - p1)/(2t) + (a + b)/2
        boundary = (giant_offline - price_offline) / (2 * t) + (shop_pos + giant_pos) / 2
        boundary = max(0, min(1, boundary))
        
        if shop_pos < giant_pos:
            offline_share = boundary
        else:
            offline_share = 1 - boundary
    
    # 线上市场份额（价格竞争为主）
    price_ratio = price_online / giant_online
    online_share = 1 / (1 + price_ratio ** consumer_params['price_sensitivity'])
    
    # 总消费者数量（日均客流）
    total_consumers = STREET_LENGTH * consumer_params['density']
    
    # 线下需求（仅考虑愿意到店的消费者比例，乘以转化率）
    offline_demand = offline_share * total_consumers * consumer_params['offline_ratio'] * consumer_params['conversion_rate']
    
    # 线上需求（乘以转化率）
    online_demand = online_share * total_consumers * consumer_params['online_ratio'] * consumer_params['conversion_rate']
    
    return offline_demand, online_demand

# ==================== 利润函数 ====================
def calculate_profit(shop_pos, price_offline, price_online, 
                    giant_pos, giant_offline, giant_online,
                    location_type='normal'):
    """
    计算利润
    
    参数：
    - shop_pos: 位置（0-1）
    - price_offline: 线下价格
    - price_online: 线上价格
    - location_type: 位置类型 ('prime', 'normal', 'back_alley')
    
    返回：
    - profit: 日利润
    """
    # 计算需求
    offline_demand, online_demand = calculate_demand(
        shop_pos, price_offline, price_online,
        giant_pos, giant_offline, giant_online
    )
    
    # 收入
    revenue_offline = offline_demand * price_offline
    revenue_online = online_demand * price_online
    
    # 变动成本
    cost_offline = offline_demand * cost_params['base_cost']
    cost_online = online_demand * (cost_params['base_cost'] + cost_params['delivery_cost'])
    
    # 平台抽成
    platform_fee = revenue_online * cost_params['platform_fee']
    
    # 租金（日租金）
    if location_type == 'prime':
        daily_rent = rent_params['prime_location'] / 30
    elif location_type == 'back_alley':
        daily_rent = rent_params['back_alley'] / 30
    else:
        daily_rent = rent_params['normal_location'] / 30
    
    # 利润
    profit = (revenue_offline + revenue_online) - (cost_offline + cost_online) - platform_fee - daily_rent
    
    return profit

# ==================== 博弈模型 ====================
def entrepreneur_profit(x, giant_strategy):
    """
    创业者利润函数（作为博弈参与者）
    
    参数：
    - x: [位置, 线下价格, 线上价格]
    - giant_strategy: [位置, 线下价格, 线上价格]
    
    返回：
    - profit: 利润
    """
    shop_pos, price_offline, price_online = x
    giant_pos, giant_offline, giant_online = giant_strategy
    
    # 判断位置类型
    if abs(shop_pos - 0.5) < 0.05 or abs(shop_pos - 0.25) < 0.05 or abs(shop_pos - 0.75) < 0.05:
        location_type = 'prime'
    elif shop_pos < 0.1 or shop_pos > 0.9:
        location_type = 'back_alley'
    else:
        location_type = 'normal'
    
    return calculate_profit(shop_pos, price_offline, price_online,
                           giant_pos, giant_offline, giant_online,
                           location_type)

def giant_profit(x, entrepreneur_strategy):
    """
    巨头利润函数
    
    参数：
    - x: [位置, 线下价格, 线上价格]
    - entrepreneur_strategy: [位置, 线下价格, 线上价格]
    
    返回：
    - profit: 利润
    """
    giant_pos, giant_offline, giant_online = x
    shop_pos, price_offline, price_online = entrepreneur_strategy
    
    return calculate_profit(giant_pos, giant_offline, giant_online,
                           shop_pos, price_offline, price_online,
                           'prime')  # 巨头在黄金位置

# ==================== 纳什均衡求解 ====================
def find_nash_equilibrium(max_iter=100, tol=1e-3):
    """
    寻找纳什均衡
    
    返回：
    - entrepreneur_strategy: 创业者策略
    - giant_strategy: 巨头策略
    """
    # 初始策略
    # 巨头固定在1/2位置
    giant_pos = 0.5
    giant_offline = 18
    giant_online = 20
    
    # 创业者初始策略（尝试在1/4位置）
    shop_pos = 0.25
    price_offline = 17
    price_online = 19
    
    for iteration in range(max_iter):
        # 创业者最优反应
        def entrepreneur_objective(x):
            return -entrepreneur_profit(x, [giant_pos, giant_offline, giant_online])
        
        result_entrepreneur = minimize(
            entrepreneur_objective,
            [shop_pos, price_offline, price_online],
            bounds=[(0, 1), (10, 30), (12, 35)],
            method='L-BFGS-B'
        )
        
        new_shop_pos, new_price_offline, new_price_online = result_entrepreneur.x
        
        # 巨头最优反应（位置固定，仅调整价格）
        def giant_objective(x):
            return -giant_profit([giant_pos, x[0], x[1]], [new_shop_pos, new_price_offline, new_price_online])
        
        result_giant = minimize(
            giant_objective,
            [giant_offline, giant_online],
            bounds=[(10, 30), (12, 35)],
            method='L-BFGS-B'
        )
        
        new_giant_offline, new_giant_online = result_giant.x
        
        # 检查收敛
        pos_diff = abs(new_shop_pos - shop_pos)
        price_diff = abs(new_price_offline - price_offline) + abs(new_price_online - price_online)
        giant_price_diff = abs(new_giant_offline - giant_offline) + abs(new_giant_online - giant_online)
        
        # 更新策略
        shop_pos, price_offline, price_online = new_shop_pos, new_price_offline, new_price_online
        giant_offline, giant_online = new_giant_offline, new_giant_online
        
        if pos_diff < tol and price_diff < tol and giant_price_diff < tol:
            print(f"纳什均衡在第{iteration+1}次迭代收敛")
            break
    
    return {
        'entrepreneur': {
            'position': shop_pos,
            'price_offline': price_offline,
            'price_online': price_online,
            'profit': entrepreneur_profit([shop_pos, price_offline, price_online], [giant_pos, giant_offline, giant_online])
        },
        'giant': {
            'position': giant_pos,
            'price_offline': giant_offline,
            'price_online': giant_online,
            'profit': giant_profit([giant_pos, giant_offline, giant_online], [shop_pos, price_offline, price_online])
        }
    }

# ==================== 不同选址策略对比 ====================
def compare_strategies():
    """
    对比不同选址策略的利润
    """
    strategies = [
        {'name': '主街1/4位置', 'position': 0.25, 'type': 'prime'},
        {'name': '主街3/4位置', 'position': 0.75, 'type': 'prime'},
        {'name': '主街普通位置', 'position': 0.35, 'type': 'normal'},
        {'name': '背街云餐厅', 'position': 0.05, 'type': 'back_alley'},
    ]
    
    results = []
    
    for strategy in strategies:
        # 优化价格
        def objective(x):
            return -calculate_profit(
                strategy['position'], x[0], x[1],
                giant_params['location'], giant_params['offline_price'], giant_params['online_price'],
                strategy['type']
            )
        
        result = minimize(objective, [17, 19], bounds=[(10, 30), (12, 35)], method='L-BFGS-B')
        opt_offline, opt_online = result.x
        
        profit = calculate_profit(
            strategy['position'], opt_offline, opt_online,
            giant_params['location'], giant_params['offline_price'], giant_params['online_price'],
            strategy['type']
        )
        
        results.append({
            'strategy': strategy['name'],
            'position': strategy['position'],
            'location_type': strategy['type'],
            'optimal_offline_price': opt_offline,
            'optimal_online_price': opt_online,
            'profit': profit,
            'online_ratio': opt_online / (opt_offline + opt_online)
        })
    
    return pd.DataFrame(results)

# ==================== 主程序 ====================
if __name__ == '__main__':
    print("="*60)
    print("问题四：空间博弈与云餐厅选址决策模型")
    print("="*60)
    
    # 1. 不同选址策略对比
    print("\n1. 不同选址策略对比")
    print("-"*60)
    strategy_df = compare_strategies()
    
    print("\n策略对比结果：")
    print(strategy_df.to_string(index=False))
    
    # 2. 纳什均衡分析
    print("\n2. 纳什均衡分析")
    print("-"*60)
    equilibrium = find_nash_equilibrium()
    
    print("\n纳什均衡策略：")
    print(f"\n创业者策略：")
    print(f"  位置：{equilibrium['entrepreneur']['position']*100:.1f}%（{equilibrium['entrepreneur']['position']*STREET_LENGTH:.0f}米处）")
    print(f"  线下价格：{equilibrium['entrepreneur']['price_offline']:.2f}元")
    print(f"  线上价格：{equilibrium['entrepreneur']['price_online']:.2f}元")
    print(f"  日利润：{equilibrium['entrepreneur']['profit']:.2f}元")
    
    print(f"\n巨头策略：")
    print(f"  位置：{equilibrium['giant']['position']*100:.1f}%（黄金地段）")
    print(f"  线下价格：{equilibrium['giant']['price_offline']:.2f}元")
    print(f"  线上价格：{equilibrium['giant']['price_online']:.2f}元")
    print(f"  日利润：{equilibrium['giant']['profit']:.2f}元")
    
    # 3. 可视化
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 策略对比柱状图
    axes[0].bar(strategy_df['strategy'], strategy_df['profit'], color=['#3498DB', '#3498DB', '#95A5A6', '#E74C3C'])
    axes[0].set_title('不同选址策略的日利润对比', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('选址策略', fontsize=12)
    axes[0].set_ylabel('日利润（元）', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 线上价格占比
    axes[1].bar(strategy_df['strategy'], strategy_df['online_ratio'], color=['#3498DB', '#3498DB', '#95A5A6', '#E74C3C'])
    axes[1].set_title('不同策略的线上渠道价格占比', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('选址策略', fontsize=12)
    axes[1].set_ylabel('线上价格占比', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:/files/管理运筹学_work/A题建模过程/Problem4_results.png', dpi=300, bbox_inches='tight')
    print("\n分析图表已保存为: Problem4_results.png")
    
    # 4. 管理建议
    print("\n3. 管理建议")
    print("-"*60)
    print("基于模型分析，创业者最优策略建议：")
    
    # 找出最优策略
    best_strategy = strategy_df.loc[strategy_df['profit'].idxmax()]
    
    print(f"\n最优策略：{best_strategy['strategy']}")
    print(f"预期日利润：{best_strategy['profit']:.2f}元")
    
    if best_strategy['location_type'] == 'back_alley':
        print("\n建议选择云餐厅模式，原因：")
        print("  ① 租金成本仅为主街的20%，显著降低固定成本")
        print("  ② 避开与巨头的线下直接竞争")
        print("  ③ 专注线上渠道，发挥外卖平台的空间突破优势")
        print("  ④ 线上价格可略低于巨头，吸引价格敏感型消费者")
    else:
        print("\n建议选择主街选址，原因：")
        print("  ① 可获得线下客流，双渠道协同")
        print("  ② 地理位置优势带来的品牌曝光")
        print("  ③ 可与巨头形成差异化定位")