# lambda参数	1.00E+00	1.00E-01	1.00E-02	1.00E-03	1.00E-04	1.00E-08
# SDF error	0.084 	0.072 	0.070 	0.055 	0.051 	0.051 
# compression-rate	59.95 	55.48 	30.10 	9.48 	4.85 	3.46 
# plot compression-rate vs sdf-error for different lambda values
import matplotlib.pyplot as plt
import numpy as np

# 数据
lambda_values = [1.00E+00, 1.00E-01, 1.00E-02, 1.00E-03, 1.00E-04, 1.00E-08]
sdf_error = [0.084, 0.072, 0.070, 0.061, 0.051, 0.051]
compression_rate = [59.95, 55.48, 30.10, 9.48, 4.85, 3.46]

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制曲线和散点
plt.plot(sdf_error, compression_rate, 'o-', linewidth=2, markersize=8, 
         color='#3b82f6', label='λ parameters', markerfacecolor='#3b82f6')

# 在每个点旁边标注lambda值
for i, lambda_val in enumerate(lambda_values):
    plt.annotate(f'λ={lambda_val:.0e}', 
                xy=(sdf_error[i], compression_rate[i]),
                xytext=(10, 5), textcoords='offset points',
                fontsize=9, color='#1f2937',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor='gray', alpha=0.8))

# 设置标签和标题
plt.xlabel('SDF Error', fontsize=12, fontweight='bold')
plt.ylabel('Compression Rate', fontsize=12, fontweight='bold')
plt.title('Compression Rate vs SDF Error\nTrade-off for Different λ Parameters', 
          fontsize=14, fontweight='bold', pad=20)

# 设置网格
plt.grid(True, linestyle='--', alpha=0.3)

# 设置坐标轴范围
plt.xlim(0.04, 0.09)
plt.ylim(0, 65)

# 添加图例
plt.legend(loc='lower right', fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()