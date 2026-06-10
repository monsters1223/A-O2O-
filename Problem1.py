"""
问题一：文汇路大学生O2O双渠道消费选择机理与市场份额模拟
基于Logit离散选择模型
"""

import numpy as np
import pandas as pd
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==================== 参数设定 ====================
# 文汇路参数
STREET_LENGTH = 2500  # 文汇路长度（米）
NUM_CONSUMERS = 10000  # 消费者数量
NUM_SHOPS = 5  # 店铺数量

# 店铺位置（随机生成）
np.random.seed(42)
shop_positions = np.sort(np.random.choice(range(100, STREET_LENGTH-100), NUM_SHOPS, replace=False))

# 店铺定位参数
shop_info = pd.DataFrame({
    'shop_id': range(1, NUM_SHOPS+1),
    'position': shop_positions,
    'price_offline': [28, 22, 32, 25, 30],  # 线下价格
    'price_online': [25, 19, 28, 22, 26],   # 线上价格
    'experience': [4.2, 3.5, 4.8, 4.0, 4.5],  # 体验评分(1-5)
    'delivery_fee': [2, 1.5, 3, 2, 2.5],     # 配送费
    'discount': [0.1, 0.15, 0.05, 0.12, 0.08]  # 折扣力度
})

# 效用函数参数
beta = {'intercept': 2.0, 'price': -0.15, 'time_cost': -0.08, 'experience': 0.6}
gamma = {'intercept': 1.5, 'price': -0.12, 'delivery_time': -0.06,
         'delivery_fee': -0.8, 'discount': 5.0}

# ==================== 生成消费者数据 ====================
consumer_positions = np.random.uniform(0, STREET_LENGTH, NUM_CONSUMERS)

# ==================== 计算效用函数 ====================
def calculate_offline_utility(consumer_pos, shop_info, beta):
    """计算线下渠道效用"""
    utilities = []
    for _, shop in shop_info.iterrows():
        # 计算到店时间成本（步行时间+等待时间）
        walk_time = abs(consumer_pos - shop['position']) / 50  # 步行速度50米/分钟
        wait_time = np.random.uniform(5, 15)  # 随机等待时间
        time_cost = walk_time + wait_time

        # 计算效用
        V = (beta['intercept'] +
             beta['price'] * shop['price_offline'] +
             beta['time_cost'] * time_cost +
             beta['experience'] * shop['experience'])

        # 添加随机误差（Gumbel分布）
        epsilon = gumbel_r.rvs()
        U = V + epsilon

        utilities.append(U)

    return utilities

def calculate_online_utility(consumer_pos, shop_info, gamma):
    """计算线上渠道效用"""
    utilities = []
    for _, shop in shop_info.iterrows():
        # 计算配送时间
        delivery_time = abs(consumer_pos - shop['position']) / 200 + np.random.uniform(5, 10)

        # 计算效用
        V = (gamma['intercept'] +
             gamma['price'] * shop['price_online'] +
             gamma['delivery_time'] * delivery_time +
             gamma['delivery_fee'] * shop['delivery_fee'] +
             gamma['discount'] * shop['discount'])

        # 添加随机误差（Gumbel分布）
        epsilon = gumbel_r.rvs()
        U = V + epsilon

        utilities.append(U)

    return utilities

# ==================== 模拟消费者选择 ====================
def simulate_choices(consumer_positions, shop_info, beta, gamma):
    """模拟所有消费者的选择"""
    results = []

    for pos in consumer_positions:
        # 计算线下效用
        offline_utils = calculate_offline_utility(pos, shop_info, beta)
        # 计算线上效用
        online_utils = calculate_online_utility(pos, shop_info, gamma)

        # 整合所有选项（线下+线上）
        all_utils = {f'线下_{i+1}': offline_utils[i] for i in range(NUM_SHOPS)}
        all_utils.update({f'线上_{i+1}': online_utils[i] for i in range(NUM_SHOPS)})

        # 选择效用最大的选项
        choice = max(all_utils, key=all_utils.get)

        results.append({
            'consumer_pos': pos,
            'choice': choice,
            'channel': '线下' if '线下' in choice else '线上',
            'shop_id': int(choice.split('_')[1])
        })

    return pd.DataFrame(results)

# 执行模拟
print("开始模拟消费者选择...")
choices_df = simulate_choices(consumer_positions, shop_info, beta, gamma)
print("模拟完成！\n")

# ==================== 计算市场份额 ====================
# 总体市场份额
market_share = choices_df.groupby(['shop_id', 'channel']).size().unstack(fill_value=0)
market_share['合计'] = market_share.sum(axis=1)
market_share = market_share / NUM_CONSUMERS * 100  # 转换为百分比

# 输出结果
print("="*60)
print("各店铺市场份额（%）")
print("="*60)
print(market_share.round(2))

# 渠道占比
channel_share = choices_df['channel'].value_counts() / NUM_CONSUMERS * 100
print("\n" + "="*60)
print("渠道占比（%）")
print("="*60)
print(channel_share.round(2))

# ==================== 详细分析 ====================
print("\n" + "="*60)
print("店铺详细信息")
print("="*60)
print(shop_info.to_string(index=False))

# ==================== 可视化结果 ====================
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 市场份额堆叠柱状图
bars = market_share[['线下', '线上']].plot(kind='bar', stacked=True, ax=axes[0, 0],
                                     color=['#4A90D9', '#E74C3C'], edgecolor='white', linewidth=1, width=0.7)
axes[0, 0].set_title('各店铺线上线下市场份额', fontsize=16, fontweight='bold', pad=20)
axes[0, 0].set_xlabel('店铺编号', fontsize=13, labelpad=10)
axes[0, 0].set_ylabel('市场份额（%）', fontsize=13, labelpad=10)
axes[0, 0].legend(title='渠道', fontsize=12, loc='upper right', bbox_to_anchor=(1.05, 1))
axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')
axes[0, 0].tick_params(axis='both', labelsize=11)
axes[0, 0].set_ylim(0, 35)

# 添加堆叠柱状图的标签（智能判断颜色）
for i, (index, row) in enumerate(market_share.iterrows()):
    offline = row['线下']
    online = row['线上']
    total = row['合计']

    # 线下标签：柱子足够高时白字在内部，否则黑字在外部
    if offline >= 4:  # 柱子高度>=4%时用白字
        axes[0, 0].text(i, offline/2, f'{offline:.1f}%', ha='center', fontsize=10,
                        fontweight='bold', color='white')
    else:  # 柱子太短时用黑字标注在右侧
        axes[0, 0].text(i + 0.15, offline, f'{offline:.1f}%', ha='left', fontsize=10,
                        fontweight='bold', color='black', va='center')

    # 线上标签
    if online >= 4:
        axes[0, 0].text(i, offline + online/2, f'{online:.1f}%', ha='center', fontsize=10,
                        fontweight='bold', color='white')
    else:
        axes[0, 0].text(i + 0.15, offline + online, f'{online:.1f}%', ha='left', fontsize=10,
                        fontweight='bold', color='black', va='center')

    # 合计标签（在柱子顶部）
    axes[0, 0].text(i, total + 0.8, f'合计 {total:.1f}%', ha='center', fontsize=11, fontweight='bold')

# 2. 渠道占比饼图
channel_share.plot(kind='pie', ax=axes[0, 1], 
                   colors=['#4A90D9', '#E74C3C'],
                   autopct='%1.1f%%', startangle=90, 
                   fontsize=12, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0, 1].set_title('线上线下渠道占比', fontsize=16, fontweight='bold', pad=20)
axes[0, 1].set_ylabel('')

# 3. 空间分布可视化 - 优化版本（店铺融入消费者群体中）
# 消费者分布在y轴中下部
consumer_y = np.random.uniform(-0.4, 0.1, NUM_CONSUMERS)  # 消费者分布在下方
axes[1, 0].scatter(consumer_positions, consumer_y, alpha=0.5, s=15, c='#3498DB', marker='o',
                   label='消费者', edgecolor='white', linewidth=0.3)

# 店铺也分布在文汇路上（与消费者在同一y轴区域，更真实）
# 5家店铺都位于街道两侧，y值有微小差异
shop_y = [0.15, 0.20, 0.15, 0.18, 0.15]  # 店铺都位于街道附近
axes[1, 0].scatter(shop_positions, shop_y, s=350, c='#E74C3C', marker='^',
                   label='店铺', edgecolor='white', linewidth=2, zorder=10)

# 添加店铺编号标签（位于店铺上方）
for i, pos in enumerate(shop_positions):
    axes[1, 0].annotate(f'店铺{i+1}', xy=(pos, shop_y[i]), xytext=(pos, shop_y[i] + 0.12),
                        ha='center', fontsize=12, fontweight='bold', color='black',
                        arrowprops=dict(arrowstyle='-', color='gray', lw=0.8, alpha=0.6))

# 添加门店定位标注线
for i, pos in enumerate(shop_positions):
    axes[1, 0].axvline(x=pos, ymin=0, ymax=0.6, color='gray', linestyle=':', alpha=0.3, zorder=1)

axes[1, 0].set_title('文汇路消费者与店铺空间分布', fontsize=16, fontweight='bold', pad=20)
axes[1, 0].set_xlabel('距离文汇路起点位置（米）', fontsize=13, labelpad=10)
axes[1, 0].set_ylabel('垂直位置（示意）', fontsize=13, labelpad=10)
axes[1, 0].set_yticks([])
axes[1, 0].legend(fontsize=12, loc='upper right')
axes[1, 0].set_xlim(-50, STREET_LENGTH + 50)
axes[1, 0].set_ylim(-0.5, 0.5)
axes[1, 0].grid(axis='x', alpha=0.3, linestyle='--')

# 添加文汇路标识线（街道中心线）
axes[1, 0].axhline(y=-0.1, color='#7F8C8D', linestyle='-', linewidth=4, alpha=0.5, zorder=2)
axes[1, 0].text(STREET_LENGTH/2, -0.45, '文汇路（2.5公里）', ha='center', fontsize=12,
                color='#7F8C8D', fontweight='bold')

# 4. 价格与市场份额关系
shop_total_share = market_share['合计']
scatter = axes[1, 1].scatter(shop_info['price_offline'], shop_total_share, 
                             s=300, alpha=0.7, c='#27AE60', marker='s', edgecolor='white', linewidth=2)
axes[1, 1].set_title('线下价格与市场份额关系', fontsize=16, fontweight='bold', pad=20)
axes[1, 1].set_xlabel('线下价格（元）', fontsize=13, labelpad=10)
axes[1, 1].set_ylabel('市场份额（%）', fontsize=13, labelpad=10)
axes[1, 1].grid(True, alpha=0.3, linestyle='--')
axes[1, 1].tick_params(axis='both', labelsize=11)
axes[1, 1].set_xlim(18, 36)
axes[1, 1].set_ylim(10, 32)

# 添加店铺标签
for i, shop in shop_info.iterrows():
    shop_idx = int(shop['shop_id']) - 1
    if shop_idx < len(shop_total_share):
        axes[1, 1].annotate(f"店铺{shop['shop_id']}",
                           (shop['price_offline'], shop_total_share.iloc[shop_idx]),
                           fontsize=11, fontweight='bold', ha='center', va='bottom')

# 添加趋势线
z = np.polyfit(shop_info['price_offline'], shop_total_share, 1)
p = np.poly1d(z)
axes[1, 1].plot(shop_info['price_offline'], p(shop_info['price_offline']), 
                "r--", alpha=0.6, label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')
axes[1, 1].legend(fontsize=10)

# 整体布局调整
plt.tight_layout(pad=3)
plt.savefig('d:/files/管理运筹学_work/A题建模过程/Problem1_results.png', dpi=300, bbox_inches='tight')
print("\n图表已保存为: Problem1_results.png")
# plt.show()  # 注释掉plt.show()以避免在某些环境中无法显示窗口的问题

# ==================== 模型性能评估 ====================
print("\n" + "="*60)
print("模型性能评估")
print("="*60)

# 计算选择集中度（Herfindahl指数）
total_share = market_share['合计'] / 100
hh_index = np.sum(total_share ** 2)
print(f"市场集中度(HHI): {hh_index:.4f}")
print(f"  - HHI < 0.25: 竞争充分")
print(f"  - 0.25 <= HHI < 0.5: 竞争适中")
print(f"  - HHI >= 0.5: 竞争不足")

# 计算渠道转换率
conversion_rate = choices_df.groupby('shop_id')['channel'].value_counts(normalize=True).unstack(fill_value=0)
print(f"\n各店铺线上转化率:")
for shop_id in range(1, NUM_SHOPS+1):
    if '线上' in conversion_rate.columns:
        online_rate = conversion_rate.loc[shop_id, '线上'] * 100
        print(f"  店铺{shop_id}: {online_rate:.2f}%")

print("\n" + "="*60)
print("模型运行完成！")
print("="*60)