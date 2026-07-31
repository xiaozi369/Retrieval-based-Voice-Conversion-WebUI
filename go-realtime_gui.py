"""
RVC 实时变声启动器 — GPU 检测 + 启动 Realtime GUI
"""

import os
import sys
import subprocess

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

python = sys.executable
base_dir = os.path.dirname(os.path.abspath(__file__))

# === 检测并选择 GPU ===
try:
    import torch

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("未检测到可用 GPU，将使用 CPU。")
    elif n_gpus == 1:
        name = torch.cuda.get_device_name(0)
        print(f"检测到 1 块显卡: [{0}] {name}")
    else:
        print(f"检测到 {n_gpus} 块显卡:")
        for i in range(n_gpus):
            name = torch.cuda.get_device_name(i)
            print(f"  [{i}] {name}")
        while True:
            try:
                choice = input("请选择要使用的显卡序号（回车默认为 0）: ").strip()
                if choice == "":
                    choice = "0"
                idx = int(choice)
                if 0 <= idx < n_gpus:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
                    print(f"已选择显卡 [{idx}] {torch.cuda.get_device_name(idx)}")
                    break
                else:
                    print(f"序号超出范围，请输入 0~{n_gpus - 1}")
            except ValueError:
                print("请输入有效数字")
except ImportError:
    print("torch 未安装，将使用 CPU。")
except Exception as e:
    print(f"检测 GPU 时出错: {e}")

# === 启动 realtime_gui.py ===
subprocess.run(
    [python, "-s", "realtime_gui.py"],
    cwd=base_dir,
)
