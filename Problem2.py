"""
问题二：O2O双渠道下的"定价与满减"组合优化模型

核心内容：
1. 多产品定价（3款单品 + 2款套餐）
2. 双渠道差异化定价（线上 vs 线下）
3. 满减促销规则建模
4. 平台抽成成本考虑
5. 利润最大化优化
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ==================== 参数设定 ====================
# 产品信息（3款单品 + 2款套餐）
products = pd.DataFrame({
    'product_id': ['P1', 'P2', 'P3', 'C1', 'C2'],
    'name': ['经典奶茶', '水果茶', '轻食沙拉', '套餐A(茶+小食)', '套餐B(沙拉+茶)'],
    'cost': [6, 8, 12, 14, 18],  # 基础成本（元）
    'is_set': [False, False, False, True, True]  # 是否为套餐
})

# 渠道参数
channels = {
    'offline': {
        'price_elasticity': -2.0,  # 价格弹性（线下消费者对价格不太敏感）
        'base_demand': 200,  # 基础需求量
        'cost_multiplier': 1.0  # 线下无额外成本
    },
    'online': {
        'price_elasticity': -3.5,  # 价格弹性（线上消费者更敏感）
        'base_demand': 150,  # 基础需求量
        'cost_multiplier': 1.25,  # 线上额外成本（包装+配送）
        'platform_fee_rate': 0.18  # 平台抽成比例（18%）
    }
}

# 满减规则参数
rebate_rules = [
    {'threshold': 20, 'discount': 2},   # 满20减2
    {'threshold': 40, 'discount': 5},   # 满40减5
    {'threshold': 60, 'discount': 8}    # 满60减8
]

# 交叉价格弹性（不同产品之间的影响）
cross_elasticity = np.array([
    [1.0, -0.2, -0.1, -0.3, -0.1],
    [-0.2, 1.0, -0.15, -0.1, -0.3],
    [-0.1, -0.15, 1.0, -0.2, -0.4],
    [-0.3, -0.1, -0.2, 1.0, -0.25],
    [-0.1, -0.3, -0.4, -0.25, 1.0]
])

# ==================== 需求函数 ====================
def calculate_demand(prices_offline, prices_online, channel):
    """
    计算各产品在特定渠道的需求量
    
    参数：
    - prices_offline: 线下价格数组
    - prices_online: 线上价格数组
    - channel: 'offline' 或 'online'
    """
    params = channels[channel]
    
    n_products = len(products)
    demand = np.zeros(n_products)
    
    # 定义参考价格（成本的2倍作为基准）
    ref_prices = products['cost'] * 2.0
    
    for i in range(n_products):
        # 当前渠道价格
        price = prices_offline[i] if channel == 'offline' else prices_online[i]
        
        # 价格与参考价格的比值
        price_ratio = price / ref_prices[i]
        
        # 需求 = 基础需求 * (参考价格/当前价格)^价格弹性
        # 这是一个更标准的需求函数形式
        base_demand = params['base_demand'] * (0.6 if products['is_set'][i] else 0.4)
        demand[i] = base_demand * (1.0 / price_ratio) ** abs(params['price_elasticity'])
    
    # 确保需求非负
    demand = np.maximum(demand, 0)
    
    return demand

# ==================== 满减优惠计算 ====================
def calculate_rebate(order_value):
    """
    根据订单金额计算满减优惠
    
    参数：
    - order_value: 订单金额（元）
    
    返回：
    - rebate: 优惠金额（元）
    """
    rebate = 0
    for rule in sorted(rebate_rules, key=lambda x: x['threshold'], reverse=True):
        if order_value >= rule['threshold']:
            rebate = rule['discount']
            break
    return rebate

def calculate_average_rebate(prices, demand):
    """
    计算平均每单满减优惠
    
    参数：
    - prices: 价格数组
    - demand: 需求量数组
    
    返回：
    - avg_rebate: 平均每单优惠金额
    """
    # 假设订单组合遵循一定的概率分布
    # 简化：根据产品组合概率计算期望订单金额
    total_demand = np.sum(demand)
    if total_demand == 0:
        return 0
    
    # 计算期望订单金额（考虑常见购买组合）
    expected_order_value = 0
    
    # 单品购买概率（60%）
    for i in range(3):
        prob = demand[i] / total_demand * 0.6
        expected_order_value += prob * prices[i]
    
    # 双品组合概率（30%）
    for i in range(5):
        for j in range(i+1, 5):
            prob = (demand[i] * demand[j]) / (total_demand ** 2) * 0.3
            expected_order_value += prob * (prices[i] + prices[j])
    
    # 套餐购买概率（10%）
    for i in range(3, 5):
        prob = demand[i] / total_demand * 0.1
        expected_order_value += prob * prices[i]
    
    # 计算期望满减
    avg_rebate = calculate_rebate(expected_order_value)
    
    return avg_rebate

# ==================== 利润函数 ====================
def calculate_profit(prices_offline, prices_online, platform_fee_rate=0.18):
    """
    计算总利润
    
    参数：
    - prices_offline: 线下价格数组
    - prices_online: 线上价格数组
    - platform_fee_rate: 平台抽成比例
    
    返回：
    - total_profit: 总利润（元）
    """
    # 计算线下需求和利润
    demand_offline = calculate_demand(prices_offline, prices_online, 'offline')
    revenue_offline = np.sum(prices_offline * demand_offline)
    cost_offline = np.sum(products['cost'] * demand_offline)
    profit_offline = revenue_offline - cost_offline
    
    # 计算线上需求和利润
    demand_online = calculate_demand(prices_offline, prices_online, 'online')
    avg_rebate = calculate_average_rebate(prices_online, demand_online)
    
    # 线上收入（扣除满减后）
    revenue_online_before_rebate = np.sum(prices_online * demand_online)
    rebate_total = avg_rebate * np.sum(demand_online) / 2  # 平均每单优惠
    revenue_online = revenue_online_before_rebate - rebate_total
    
    # 线上成本（含平台抽成）
    cost_online = np.sum(products['cost'] * channels['online']['cost_multiplier'] * demand_online)
    platform_fee = revenue_online_before_rebate * platform_fee_rate
    
    profit_online = revenue_online - cost_online - platform_fee
    
    total_profit = profit_offline + profit_online
    
    return total_profit, {
        'offline': {'profit': profit_offline, 'demand': demand_offline, 'revenue': revenue_offline},
        'online': {'profit': profit_online, 'demand': demand_online, 'revenue': revenue_online, 'rebate': rebate_total},
        'total': total_profit
    }

# ==================== 优化模型 ====================
def objective_function(x):
    """
    目标函数：最大化利润（转为最小化负利润）
    
    参数：
    - x: 决策变量数组，包含线下价格和线上价格
         [p1_offline, p2_offline, p3_offline, c1_offline, c2_offline,
          p1_online, p2_online, p3_online, c1_online, c2_online]
    """
    n = len(products)
    prices_offline = x[:n]
    prices_online = x[n:]
    
    profit, _ = calculate_profit(prices_offline, prices_online)
    
    return -profit  # 最小化负利润 = 最大化利润

def constraint_profit_margin(x):
    """
    约束：每款产品的利润率不低于20%
    """
    n = len(products)
    prices_offline = x[:n]
    prices_online = x[n:]
    
    # 线下利润率 >= 20%
    margin_offline = (prices_offline - products['cost']) / products['cost']
    # 线上利润率 >= 15%（考虑额外成本）
    margin_online = (prices_online - products['cost'] * channels['online']['cost_multiplier']) / (products['cost'] * channels['online']['cost_multiplier'])
    
    # 确保所有利润率满足要求
    min_margin = min(np.min(margin_offline), np.min(margin_online))
    
    return min_margin - 0.15  # >= 0

def constraint_price_difference(x):
    """
    约束：同款产品线上线下价格差异不超过3元
    """
    n = len(products)
    prices_offline = x[:n]
    prices_online = x[n:]
    
    diff = np.abs(prices_offline - prices_online)
    
    return 3 - np.max(diff)  # >= 0

def constraint_reasonable_prices(x):
    """
    约束：价格在合理范围内（成本的1.3倍到3倍）
    """
    n = len(products)
    prices_offline = x[:n]
    prices_online = x[n:]
    
    all_prices = np.concatenate([prices_offline, prices_online])
    costs = np.concatenate([products['cost'], products['cost']])
    
    lower_bound = costs * 1.3
    upper_bound = costs * 3.0
    
    # 检查所有价格在范围内
    lower_ok = np.min(all_prices - lower_bound)
    upper_ok = np.min(upper_bound - all_prices)
    
    return min(lower_ok, upper_ok)

# ==================== 执行优化 ====================
def solve_optimization(platform_fee_rate=0.18):
    """
    求解优化问题
    
    参数：
    - platform_fee_rate: 平台抽成比例
    
    返回：
    - result: 优化结果
    """
    n = len(products)
    
    # 初始价格（成本的1.8倍）
    initial_prices = np.tile(products['cost'] * 1.8, 2)
    
    # 约束条件
    constraints = [
        {'type': 'ineq', 'fun': constraint_profit_margin},
        {'type': 'ineq', 'fun': constraint_price_difference},
        {'type': 'ineq', 'fun': constraint_reasonable_prices}
    ]
    
    # 边界条件
    bounds = []
    for _ in range(2):  # 线下和线上
        for cost in products['cost']:
            bounds.append((cost * 1.3, cost * 3.0))
    
    # 定义带平台抽成的目标函数
    def objective_with_fee(x):
        prices_offline = x[:n]
        prices_online = x[n:]
        profit, _ = calculate_profit(prices_offline, prices_online, platform_fee_rate)
        return -profit
    
    # 求解
    result = minimize(objective_with_fee, initial_prices, method='SLSQP',
                     constraints=constraints, bounds=bounds,
                     options={'maxiter': 1000, 'disp': True})
    
    return result

# ==================== 满减方案优化 ====================
def optimize_rebate_strategy(prices_online):
    """
    优化满减方案，计算最优满减门槛和优惠力度
    
    参数：
    - prices_online: 线上价格数组
    
    返回：
    - best_rebate: 最优满减方案
    - lift_ratio: 引流提升比例
    """
    # 可能的满减方案
    possible_rebates = [
        [{'threshold': 20, 'discount': 2}, {'threshold': 40, 'discount': 5}],
        [{'threshold': 20, 'discount': 3}, {'threshold': 45, 'discount': 6}],
        [{'threshold': 25, 'discount': 3}, {'threshold': 50, 'discount': 8}],
        [{'threshold': 18, 'discount': 2}, {'threshold': 38, 'discount': 5}, {'threshold': 58, 'discount': 9}],
    ]
    
    best_rebate = None
    best_lift = 0
    
    for rebate in possible_rebates:
        # 计算引流效应（满减带来的需求提升）
        avg_order_value = np.mean(prices_online)
        lift_ratio = 1.0
        
        # 满减力度越大，引流效果越好
        total_discount = sum(r['discount'] for r in rebate)
        lift_ratio = 1.0 + total_discount * 0.05  # 每元优惠带来5%的需求提升
        
        # 门槛越低，引流效果越好
        min_threshold = min(r['threshold'] for r in rebate)
        lift_ratio *= (1 + (20 - min_threshold) * 0.01)
        
        if lift_ratio > best_lift:
            best_lift = lift_ratio
            best_rebate = rebate
    
    return best_rebate, best_lift

# ==================== 抽成敏感性分析 ====================
def analyze_fee_sensitivity():
    """
    深入分析平台抽成上升时的动态调整策略
    """
    fee_rates = np.linspace(0.10, 0.35, 26)  # 10%到35%
    results = []
    
    for fee_rate in fee_rates:
        result = solve_optimization(fee_rate)
        
        if result.success:
            n = len(products)
            prices_offline = result.x[:n]
            prices_online = result.x[n:]
            profit, details = calculate_profit(prices_offline, prices_online, fee_rate)
            
            # 计算最优满减方案
            best_rebate, lift_ratio = optimize_rebate_strategy(prices_online)
            
            results.append({
                'fee_rate': fee_rate,
                'prices_offline': prices_offline,
                'prices_online': prices_online,
                'profit': profit,
                'price_diff': np.mean(np.abs(prices_online - prices_offline)),
                'rebate_threshold': min(r['threshold'] for r in best_rebate),
                'rebate_discount': sum(r['discount'] for r in best_rebate),
                'demand_lift': lift_ratio
            })
    
    return pd.DataFrame(results)

# ==================== 主程序 ====================
if __name__ == '__main__':
    print("="*60)
    print("问题二：O2O双渠道定价与满减组合优化模型")
    print("="*60)
    
    # 执行优化（默认18%抽成）
    print("\n1. 求解最优定价策略（平台抽成18%）")
    print("-"*60)
    result = solve_optimization()
    
    if result.success:
        n = len(products)
        optimal_offline = result.x[:n]
        optimal_online = result.x[n:]
        max_profit, details = calculate_profit(optimal_offline, optimal_online)
        
        # 输出结果
        print("\n最优线下价格：")
        for i, (name, price) in enumerate(zip(products['name'], optimal_offline)):
            print(f"  {name}: {price:.2f}元")
        
        print("\n最优线上价格：")
        for i, (name, price) in enumerate(zip(products['name'], optimal_online)):
            print(f"  {name}: {price:.2f}元")
        
        print(f"\n日总利润期望值：{max_profit:.2f}元")
        print(f"  - 线下利润：{details['offline']['profit']:.2f}元")
        print(f"  - 线上利润：{details['online']['profit']:.2f}元")
        print(f"  - 线上满减优惠：{details['online']['rebate']:.2f}元")
        
        # 满减活动方案
        print("\n2. 最优满减活动方案")
        print("-"*60)
        best_rebate, lift_ratio = optimize_rebate_strategy(optimal_online)
        print("\n推荐满减方案：")
        for rebate in best_rebate:
            print(f"  满{rebate['threshold']}元减{rebate['discount']}元")
        print(f"\n预期引流提升比例：{(lift_ratio-1)*100:.1f}%")
        
        # 分析平台抽成敏感性
        print("\n3. 平台抽成敏感性分析")
        print("-"*60)
        sensitivity_df = analyze_fee_sensitivity()
        
        print("\n平台抽成对利润和策略的影响：")
        print(f"{'抽成比例':<12} {'日利润':<12} {'平均价差':<10} {'满减门槛':<10} {'优惠力度':<10}")
        print("-" * 60)
        for _, row in sensitivity_df.iloc[::5].iterrows():
            print(f"{row['fee_rate']*100:<11}% {row['profit']:<12.1f} {row['price_diff']:<10.2f} {row['rebate_threshold']:<10} {row['rebate_discount']:<10}")
        
        # 可视化
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # 利润变化
        axes[0].plot(sensitivity_df['fee_rate']*100, sensitivity_df['profit'], 'b-o', linewidth=2)
        axes[0].set_title('平台抽成比例对总利润的影响', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('平台抽成比例（%）', fontsize=12)
        axes[0].set_ylabel('日利润（元）', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 价差变化
        axes[1].plot(sensitivity_df['fee_rate']*100, sensitivity_df['price_diff'], 'r-s', linewidth=2)
        axes[1].set_title('平台抽成比例对双渠道价差的影响', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('平台抽成比例（%）', fontsize=12)
        axes[1].set_ylabel('平均价差（元）', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        # 满减策略变化
        axes[2].plot(sensitivity_df['fee_rate']*100, sensitivity_df['rebate_threshold'], 'g-^', linewidth=2, label='满减门槛')
        axes[2].plot(sensitivity_df['fee_rate']*100, sensitivity_df['rebate_discount'], 'm-v', linewidth=2, label='优惠力度')
        axes[2].set_title('平台抽成比例对满减策略的影响', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('平台抽成比例（%）', fontsize=12)
        axes[2].set_ylabel('满减参数', fontsize=12)
        axes[2].legend(fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('d:/files/管理运筹学_work/A题建模过程/Problem2_results.png', dpi=300, bbox_inches='tight')
        print("\n分析图表已保存为: Problem2_results.png")
        
        # 输出管理建议
        print("\n4. 管理建议")
        print("-"*60)
        print("当外卖平台抽成比例上升时，商家应动态调整策略：")
        print("")
        print("【价差调整策略】")
        print("  - 抽成15%-20%：维持较小价差（1-2元），保持价格竞争力")
        print("  - 抽成20%-25%：适度扩大价差（2-3元），转嫁部分成本")
        print("  - 抽成>25%：最大化价差（接近3元上限），保护利润")
        print("")
        print("【满减门槛调整策略】")
        print("  - 抽成15%-20%：低门槛高优惠（满20减3），吸引订单")
        print("  - 抽成20%-25%：中等门槛中等优惠（满25减3）")
        print("  - 抽成>25%：高门槛低优惠（满30减3），减少利润侵蚀")
        print("")
        print("【综合建议】")
        print("  ① 提高线上价格，维持线下价格稳定")
        print("  ② 根据抽成水平动态调整满减门槛和优惠力度")
        print("  ③ 加大线下促销力度，引导消费者到店消费")
        print("  ④ 优化产品组合，提高高毛利产品占比")
        
    else:
        print("优化求解失败！")
        print(result.message)