import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib
from matplotlib.patches import Patch
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 设置字体为 Times New Roman，大小为 17
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 17
plt.rcParams['axes.labelsize'] = 17
plt.rcParams['axes.titlesize'] = 17
plt.rcParams['xtick.labelsize'] = 17
plt.rcParams['ytick.labelsize'] = 17
plt.rcParams['legend.fontsize'] = 17
plt.rcParams['axes.unicode_minus'] = False


# 新的数据集
data_sets = [
    np.array([5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50]),
    np.array([5000, 2775, 1540, 854, 474, 263, 146, 81, 45, 25]),
    np.array([5000, 2506, 1256, 629, 315, 158, 79, 39, 19, 10]),
    np.array([5000, 2320, 1077, 500, 232, 107, 50, 23, 10, 5])
]

# 创建x轴数据
x = np.arange(1, len(data_sets[0])+1)
# 定义x轴的标签
x_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 创建图形和主视图
fig, ax = plt.subplots(figsize=(7, 6))

# 绘制多条曲线
labels = ['$\mathcal{T} = 100$', '$\mathcal{T} = 50$', '$\mathcal{T} = 20$', '$\mathcal{T} = 10$']
for data, label in zip(data_sets, labels):
    ax.plot(x, data, marker='o', markersize=6, label=label)

# 设置x轴的刻度标签
plt.xticks(x, x_labels, rotation=-25)  # 或者使用 ax.set_xticklabels(x_labels)，并根据需要调整旋转角度

# 使用科学记数法显示 y 轴（不改变坐标类型）
ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


# 设置标题和轴标签
ax.set_title('Cifar10 with diffrent $\mathcal{T}$')
# ax.set_xlabel('Class label')
ax.set_ylabel('Number of Train Samples')

# # 添加图例
# ax.legend(prop={'size': 12})

# 显示网格
ax.grid(True)

# 定义您想展示的尾部数据点的数量
n_points = 3  # 将此值更改为任何您想要显示的数据点数

# 计算最小值和最大值时使用 n_points
min_val = min([np.min(data[-n_points:]) for data in data_sets]) - 5
max_val = max([np.max(data[-n_points:]) for data in data_sets]) + 5

# 创建次级视图，聚焦于尾部数据
ax_inset = fig.add_axes([0.5, 0.4, 0.35, 0.4]) # [left, bottom, width, height]

# 设置全局字体和大小
ax_inset.xaxis.set_tick_params(labelsize=17)
ax_inset.yaxis.set_tick_params(labelsize=17)
ax_inset.xaxis.set_tick_params(which='both', width=1)
ax_inset.yaxis.set_tick_params(which='both', width=1)

# 确保使用相同的字体
for tick in ax_inset.get_xticklabels():
    tick.set_fontname("Times New Roman")
for tick in ax_inset.get_yticklabels():
    tick.set_fontname("Times New Roman")

# 绘制每个数据集的最后 n_points 个点到次级视图中
for data in data_sets:
    ax_inset.plot(x[-n_points:], data[-n_points:], marker='o', markersize=6) 

ax_inset.set_xticks(x[-n_points:], x_labels[-n_points:], rotation=-15)

# 使用科学记数法显示 y 轴（不改变坐标类型）
ax_inset.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax_inset.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# 设置次级视图的范围
ax_inset.set_xlim(x[-n_points:].min() - 0.5, x[-n_points:].max() + 0.5)
ax_inset.set_ylim(min_val, max_val)

# 如果需要，可以添加网格线
ax_inset.grid(True)

# 在设置图例的地方进行如下更改：
legend = ax.legend(prop={'size': 12}, bbox_to_anchor=(0.52, -0.15), loc='upper center', ncol=4)
# 调整子图参数以便为图例留出空间
plt.subplots_adjust(bottom=0.2) # 根据需要调整这个值

# 确保在调用 plt.tight_layout() 前完成上述操作，或者如果使用了 tight_layout，
# 可能需要手动调整图例和图表间的距离，因为 tight_layout 可能会覆盖之前的布局调整。
# plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 此处 rect 参数调整主图的有效区域，避开底部和顶部的一些区域用于标题或图例。

# 展示图形或保存图片的部分保持不变
plt.savefig('custom_dataset_comparison_with_inset.png', dpi=300, bbox_inches='tight')
plt.savefig('custom_dataset_comparison_with_inset.pdf', dpi=300, bbox_inches='tight')
plt.savefig('custom_dataset_comparison_with_inset.svg', dpi=300, bbox_inches='tight')
plt.show()