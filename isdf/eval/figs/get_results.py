#!/usr/bin/env python3
"""
extract_last_metrics -> get_results.py

遍历 base-dir 的直接子目录（每个子目录为一个实验），读取每个实验目录下的 vox_res.json，
选取时间戳最大的条目（按 key 的 float 值比较），从该条目的 rays -> vis 中提取三个标量：
  - av_l1 (float)
  - l1_chomp_costs[chomp_ix] (float, chomp_ix=2)
  - av_cossim[cossim_ix] (float, cossim_ix=1)

将结果保留三位小数并输出为 CSV：
- 第一行：实验名（每列一个实验）
- 第二行：av_l1
- 第三行：l1_chomp_costs（单值）
- 第四行：av_cossim（单值）
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# 固定索引（参考 all_seq.py）
CHOMP_IX = 2
COSSIM_IX = 1

def fmt_val(v: Any) -> str:
    """格式化值为字符串，数字保留3位小数；缺失返回空字符串"""
    if v is None:
        return ""
    try:
        return f"{float(v):.3f}"
    except Exception:
        return str(v)

def extract_from_vox_res_single(vox_res_path: Path, chomp_ix: int, cossim_ix: int) -> Dict[str, Optional[float]]:
    """
    返回 dict 包含 keys: av_l1, l1_chomp_cost, av_cossim 和 error（若有）
    av_l1: float or None
    l1_chomp_cost: float or None (attempt to index chomp_ix)
    av_cossim: float or None (attempt to index cossim_ix)
    """
    if not vox_res_path.exists():
        return {"av_l1": None, "l1_chomp_cost": None, "av_cossim": None, "error": "missing file"}
    try:
        with open(vox_res_path, "r") as f:
            res = json.load(f)
    except Exception as e:
        return {"av_l1": None, "l1_chomp_cost": None, "av_cossim": None, "error": f"json load error: {e}"}

    if not res:
        return {"av_l1": None, "l1_chomp_cost": None, "av_cossim": None, "error": "empty json"}

    # 选择最后时间键（按数值最大）
    try:
        last_key = max(res.keys(), key=lambda k: float(k))
    except Exception:
        last_key = list(res.keys())[-1]

    entry = res.get(last_key, {})
    rays = entry.get("rays", {})
    vis = rays.get("vis", {})

    av_l1 = vis.get("av_l1", None)

    l1_chomp_cost = None
    try:
        l1_list = vis.get("l1_chomp_costs", None)
        if isinstance(l1_list, (list, tuple)) and len(l1_list) > chomp_ix:
            l1_chomp_cost = float(l1_list[chomp_ix])
    except Exception:
        l1_chomp_cost = None

    av_cossim = None
    try:
        cossim_list = vis.get("av_cossim", None)
        if isinstance(cossim_list, (list, tuple)) and len(cossim_list) > cossim_ix:
            av_cossim = float(cossim_list[cossim_ix])
    except Exception:
        av_cossim = None

    return {
        "av_l1": av_l1,
        "l1_chomp_cost": l1_chomp_cost,
        "av_cossim": av_cossim,
        "error": None
    }

def main(base_dir: str, out_csv: str):
    base = Path(base_dir)
    if not base.exists():
        raise SystemExit(f"Base dir not found: {base_dir}")

    # 只遍历直接子目录，按字母排序以保证稳定列顺序
    exps = sorted([p for p in base.iterdir() if p.is_dir()])

    if len(exps) == 0:
        raise SystemExit(f"No experiment subdirectories found under {base_dir}")

    names: List[str] = []
    av_l1_row: List[str] = []
    chomp_row: List[str] = []
    cossim_row: List[str] = []
    errors: List[str] = []

    for d in exps:
        names.append(d.name)
        vox_res = d / "vox_res.json"
        out = extract_from_vox_res_single(vox_res, CHOMP_IX, COSSIM_IX)
        if out.get("error"):
            errors.append(f"{d.name}: {out['error']}")
        else:
            errors.append("")

        av_l1_row.append(fmt_val(out.get("av_l1")))
        chomp_row.append(fmt_val(out.get("l1_chomp_cost")))
        cossim_row.append(fmt_val(out.get("av_cossim")))

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # header: experiment names
        f.write(",".join([f'"{n}"' for n in names]) + "\n")
        # av_l1 row
        f.write(",".join(av_l1_row) + "\n")
        # l1_chomp_cost row
        f.write(",".join(chomp_row) + "\n")
        # av_cossim row
        f.write(",".join(cossim_row) + "\n")

    print(f"Wrote CSV to: {out_path}")
    if any(e for e in errors):
        print("Some experiments had errors or missing fields:")
        for i, e in enumerate(errors):
            if e:
                print(f"  {names[i]}: {e}")
    else:
        print("All experiments processed without error.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract last-time rays.vis single-index metrics from vox_res.json into CSV")
    parser.add_argument("--base-dir", type=str, default="/home/hjx/iSDF/results/iSDF/exp1", help="Base directory that contains experiment subfolders")
    parser.add_argument("--out", type=str, default="/home/hjx/iSDF/results/iSDF/exp1/last_metrics.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args.base_dir, args.out)