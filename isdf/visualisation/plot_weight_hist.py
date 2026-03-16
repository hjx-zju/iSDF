#!/usr/bin/env python3
"""
plot_weight_hist.py

绘制 SDFMap 与 EntropySDFMap 的第二层、第三层权重直方图对比。

用法示例:
  python isdf/eval/figs/plot_weight_hist.py \
    --sdfmap-ckpt /path/to/sdfmap_checkpoint.pth \
    --entropy-ckpt /path/to/entropy_checkpoint.pth \
    --out /path/to/output_hist.png

说明:
- 从两个 checkpoint 文件中加载模型权重。
- 提取第二层（mid1）和第三层（cat_layer）的权重张量。
- 绘制四个子图（2x2 grid）：
    - 左上：SDFMap 第二层权重直方图
    - 右上：EntropySDFMap 第二层权重直方图
    - 左下：SDFMap 第三层权重直方图
    - 右下：EntropySDFMap 第三层权重直方图
- 支持自动推断层名（对 EntropySDFMap，层名带 entropy_bottleneck 或 ema_w/ema_b）。
"""
import argparse
from re import X
from tkinter import font
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_checkpoint_weights(ckpt_path: Path):
    """
    加载 checkpoint 并返回 state_dict。
    支持直接 state_dict 或包含 'model_state_dict' 键的字典。
    """
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    # 尝试提取 state_dict（常见格式：直接 dict 或包含 model_state_dict）
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
    else:
        raise ValueError(f"Unknown checkpoint format: {type(ckpt)}")
    
    return state_dict

def extract_layer_weights(state_dict, layer_name_patterns):
    """
    从 state_dict 中提取匹配给定模式的层权重。
    layer_name_patterns: list of str patterns（例如 ['mid1', 'cat_layer']）
    返回 dict: {pattern: tensor}
    """
    results = {}
    for pattern in layer_name_patterns:
        matched = []
        for k, v in state_dict.items():
            # 匹配包含 pattern 且包含 'weight' 的键
            if pattern in k and 'weight' in k:
                matched.append((k, v))
        
        if not matched:
            print(f"Warning: No weights found for pattern '{pattern}'")
            results[pattern] = None
        else:
            # 如果有多个匹配，合并或选择主权重
            # 通常第一个匹配的是主权重，或 ema_w 是编码后的权重
            # 优先选择不含 'entropy_bottleneck' 的（即原始权重或 ema_w）
            main_weight = None
            for k, v in matched:
                if 'entropy_bottleneck' not in k:
                    main_weight = v
                    print(f"  Found '{pattern}' weight in key: {k}, shape: {v.shape}")
                    break
            if main_weight is None:
                # fallback: 取第一个
                main_weight = matched[0][1]
                print(f"  Found '{pattern}' weight in key: {matched[0][0]}, shape: {main_weight.shape}")
            
            results[pattern] = main_weight
    
    return results

def plot_histograms(sdfmap_weights, entropy_weights, layer_names, out_path: Path, bins=50, xlim=None):
    """
    为每一层在同一子图中叠加绘制 SDFMap 与 EntropySDFMap 的权重直方图。
    如果传入 xlim (tuple/list of 2 floats)，将强制使用该范围；否则自动使用对称范围。
    """
    fig, axes = plt.subplots(1, len(layer_names), figsize=(14, 6))
    if len(layer_names) == 1:
        axes = [axes]
    # fig.suptitle("Weight Histograms (overlay): SDFMap vs EntropySDFMap", fontsize=16, fontweight='bold')

    colors = {'sdf': 'steelblue', 'ent': 'darkorange'}
    alphas = {'sdf': 0.5, 'ent': 0.5}

    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        sdf_w = sdfmap_weights.get(layer_name)
        ent_w = entropy_weights.get(layer_name)

        sdf_arr = sdf_w.detach().cpu().numpy().flatten() if sdf_w is not None else None
        ent_arr = ent_w.detach().cpu().numpy().flatten() if ent_w is not None else None

        # 计算自动对称 xlim（仅在未传入 xlim 时使用）
        if xlim is None:
            max_abs = 0.0
            for arr in (sdf_arr, ent_arr):
                if arr is not None and arr.size > 0:
                    cur = max(abs(arr.min()), abs(arr.max()))
                    if cur > max_abs:
                        max_abs = cur
            if max_abs == 0:
                max_abs = 1e-6
            use_xlim = (-max_abs, max_abs)
        else:
            use_xlim = (float(xlim[0]), float(xlim[1]))
# 准备 weights（使柱高为概率 counts/N）
        sdf_weights = None
        ent_weights = None
        if sdf_arr is not None and sdf_arr.size > 0:
            sdf_weights = np.ones_like(sdf_arr) / float(sdf_arr.size)
        if ent_arr is not None and ent_arr.size > 0:
            ent_weights = np.ones_like(ent_arr) / float(ent_arr.size)
        # 绘制（使用 density=True 便于形状比较）
        plotted_any = False
        if sdf_arr is not None and sdf_arr.size > 0:
            ax.hist(sdf_arr, bins=bins, color=colors['sdf'], alpha=alphas['sdf'],
                     density=False,weights=sdf_weights, label='iSDF')
            plotted_any = True
            mean_s, std_s = np.mean(sdf_arr), np.std(sdf_arr)
        else:
            mean_s = std_s = None

        if ent_arr is not None and ent_arr.size > 0:
            ax.hist(ent_arr, bins=bins, color=colors['ent'], alpha=alphas['ent'],
                     density=False,weights=ent_weights, label='Ours')
            plotted_any = True
            mean_e, std_e = np.mean(ent_arr), np.std(ent_arr)
        else:
            mean_e = std_e = None

        # ax.set_title(f"Layer {i+2}", fontsize=12, fontweight='bold')
        # ax.set_xlabel("Weight value",fontsize=12, fontweight='bold')
        # ax.set_ylabel("Frequency" if plotted_any else "",fontsize=12, fontweight='bold')
        ax.set_title(f"第{i+2}层", fontsize=20, fontweight='bold')
        ax.set_xlabel("权重值",fontsize=20, fontweight='bold')
        ax.set_ylabel("权重值频率" if plotted_any else "",fontsize=20, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=15)
        # 使用指定或自动计算到的 xlim（比如 [-0.5, 0.5]）
        ax.set_xlim(use_xlim)

        if plotted_any:
            ax.legend(fontsize=15, loc='upper right',frameon=True,shadow=True)
            # stats_lines = []
            # if mean_s is not None:
            #     stats_lines.append(f"SDF μ={mean_s:.3e}, σ={std_s:.1e}")
            # if mean_e is not None:
            #     stats_lines.append(f"ENT μ={mean_e:.3e}, σ={std_e:.1e}")
            # if stats_lines:
            #     ax.text(0.98, 0.95, "\n".join(stats_lines),
            #             ha='right', va='top', transform=ax.transAxes,
            #             fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

        if not plotted_any:
            ax.text(0.5, 0.5, f"No data for {layer_name}",
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{layer_name} (N/A)", fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out_path), dpi=200)
    plt.close()
    print(f"Saved overlay histogram plot to: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Plot weight histograms for SDFMap and EntropySDFMap checkpoints")
    parser.add_argument("--sdfmap-ckpt", type=str, required=False, help="Path to SDFMap checkpoint (.pth)",default="/home/hjx/iSDF/results/iSDF/12-07-25_18-09-10/checkpoints/step_130.000.pth")
    parser.add_argument("--entropy-ckpt", type=str, required=False, help="Path to EntropySDFMap checkpoint (.pth)",default="/home/hjx/iSDF/results/iSDF/12-07-25_18-49-20_entropy/checkpoints/step_130.000.pth")
    parser.add_argument("--out", type=str, default="weight_histograms_cn.png", help="Output image path")
    parser.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    parser.add_argument("--xlim", type=float, nargs=2, default=[-0.5, 0.5],
                        help="X axis limits: provide two floats (min max). Default: -0.5 0.5")
    args = parser.parse_args()
    
    sdfmap_ckpt = Path(args.sdfmap_ckpt)
    entropy_ckpt = Path(args.entropy_ckpt)
    out_path = Path(args.out)
    
    print("Loading SDFMap checkpoint...")
    sdfmap_state = load_checkpoint_weights(sdfmap_ckpt)
    print(f"  Loaded {len(sdfmap_state)} keys from {sdfmap_ckpt.name}")
    
    print("Loading EntropySDFMap checkpoint...")
    entropy_state = load_checkpoint_weights(entropy_ckpt)
    print(f"  Loaded {len(entropy_state)} keys from {entropy_ckpt.name}")
    
    # 定义要提取的层名称模式（第二层：mid1，第三层：cat_layer）
    layer_patterns = ['mid1', 'cat_layer']
    
    print("\nExtracting weights from SDFMap...")
    sdfmap_weights = extract_layer_weights(sdfmap_state, layer_patterns)
    
    print("\nExtracting weights from EntropySDFMap...")
    entropy_weights = extract_layer_weights(entropy_state, layer_patterns)
    
    print("\nGenerating histogram plots...")
    plot_histograms(sdfmap_weights, entropy_weights, layer_patterns, out_path, bins=args.bins, xlim=args.xlim)

if __name__ == "__main__":
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] 
    main()