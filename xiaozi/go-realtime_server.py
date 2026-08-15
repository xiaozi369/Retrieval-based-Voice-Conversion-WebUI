"""
RVC 控制服务启动器 — GPU 检测 + 启动 realtime_server.py
"""

import os
import sys
import subprocess

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 本文件位于 xiaozi/ 子目录,项目根目录为上溯一级
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

python = sys.executable
base_dir = _ROOT_DIR

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

# === 写入 RVC 根目录路径引导 ===
site_packages = os.path.join(base_dir, "venv", "Lib", "site-packages")
os.makedirs(site_packages, exist_ok=True)
root_anchor = os.path.join(site_packages, "rvc_root_path.pth")
with open(root_anchor, "w", encoding="utf-8") as f:
    f.write(
        "import sys, os as _o; _r = %r; sys.path.insert(0, _r); "
        "[__import__(_m) for _m in ('infer', 'configs', 'i18n', 'tools', 'train')]\n"
        % base_dir
    )


# === 启动 realtime_server.py(同目录,位于 xiaozi/ 子目录) ===
print("\033[1;32m[INFO]\033[0m 正在启动控制服务 realtime_server.py...")
subprocess.run(
    [python, "-s", os.path.join(os.path.dirname(os.path.abspath(__file__)), "realtime_server.py")],
    cwd=base_dir,
)
