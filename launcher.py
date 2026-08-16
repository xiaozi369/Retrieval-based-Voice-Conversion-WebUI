"""
RVC 启动菜单 — 支持方向键选择
"""

import os
import sys
import subprocess
import msvcrt
import webbrowser

# ANSI 颜色常量
RESET = "\033[0m"
HIGHLIGHT = "\033[1;36m"
GRAY = "\033[2m"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_EXECUTABLE = sys.executable
GIT_BIN_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", ".xiaoziya", "PortableGit", "bin")
)
FFMPEG_BIN_DIR = os.path.normpath(
    os.path.join(SCRIPT_DIR, "..", "..", ".xiaoziya", "ffmpeg", "bin")
)

# 把 PortableGit / ffmpeg 的 bin 目录加入本次进程的 PATH，
for bin_dir in (GIT_BIN_DIR, FFMPEG_BIN_DIR):
    if os.path.isdir(bin_dir):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

OPTIONS = [
    {
        "name": "启动 Web 界面",
        "file": "go-webui.py",
        "description": "启动 Gradio WebUI，训练推理一体界面",
    },
    {
        "name": "启动实时变声",
        "exe": "rvc-晓.exe",
        "description": "启动实时变声器界面，低延迟",
    },
    {
        "name": "下载模型",
        "url": "https://pan.quark.cn/s/cbd4b92e20e2",
        "description": "前往夸克网盘下载模型文件",
    },
]


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def show_menu(selected_index):
    clear_screen()
    print("=" * 50)
    print(f"   RVC  启动选项 | {HIGHLIGHT}B站晓子ya{RESET}")
    print("=" * 50)
    print()

    for i, option in enumerate(OPTIONS):
        if i == selected_index:
            print(f"  → [{i+1}] {HIGHLIGHT}{option['name']}{RESET}")
        else:
            print(f"    [{i+1}] {option['name']}")
        if "description" in option:
            print(f"       {option['description']}")
        if i < len(OPTIONS) - 1:
            print()

    print()
    print("使用 ↑ ↓ 键选择，Enter 键确认")
    print("=" * 50)


def select_option():
    selected_index = 0
    show_menu(selected_index)

    while True:
        key = msvcrt.getch()

        if key == b"\xe0" or key == b"\x00":
            key = msvcrt.getch()
            if key == b"H":
                selected_index = (selected_index - 1) % len(OPTIONS)
                show_menu(selected_index)
            elif key == b"P":
                selected_index = (selected_index + 1) % len(OPTIONS)
                show_menu(selected_index)
        elif key == b"\r":
            return selected_index
        elif key in [str(i).encode() for i in range(1, len(OPTIONS) + 1)]:
            selected_index = int(key.decode()) - 1
            return selected_index
        elif key == b"\x1b":
            print("\n已取消启动")
            sys.exit(0)


def download_exe(target):
    """从 Gitee 下载 rvc-晓.exe 到目标路径（不存在时由启动分支调用）。"""
    url = (
        "https://gitee.com/xiaozi22333/rvc-xiao/releases/download/"
        "v0.1.1/rvc-%E6%99%93.exe"
    )
    print("未找到程序，正在下载 RVC-晓启动器 ...")
    try:
        import urllib.request

        def _report(block_count, block_size, total_size):
            downloaded = block_count * block_size
            if total_size > 0:
                percent = min(100, int(downloaded * 100 / total_size))
                print(
                    f"\r下载进度: {percent}% "
                    f"({downloaded // 1024 // 1024}MB / {total_size // 1024 // 1024}MB)",
                    end="",
                )
            else:
                print(f"\r已下载: {downloaded // 1024 // 1024}MB", end="")

        urllib.request.urlretrieve(url, target, reporthook=_report)
        print()
        print("✓ 下载完成")
        return True
    except Exception as e:
        print(f"\n⚠ 下载失败: {e}")
        return False


def git_pull():
    """自动拉取更新，自动暂存用户的本地改动"""
    if not os.path.isdir(GIT_BIN_DIR):
        return

    print("正在检查更新...")
    try:
        result = subprocess.run(
            ["git", "pull", "--rebase", "--autostash"],
            capture_output=True, text=True, timeout=30, cwd=SCRIPT_DIR,
        )
        if result.returncode == 0:
            out = result.stdout.strip()
            if out and "Already up to date" not in out:
                print(out)
                print("✓ 更新完成")
            else:
                print("✓ 已是最新版本")
        else:
            err = result.stderr.strip()
            if "CONFLICT" in err:
                print("⚠ 合并冲突，请手动解决后重试")
            else:
                print(f"⚠ 更新跳过: {err.splitlines()[-1]}")
    except subprocess.TimeoutExpired:
        print("⚠ 更新超时，跳过")
    except Exception as e:
        print(f"⚠ 更新失败: {e}")


def main():
    git_pull()
    try:
        while True:
            selected_index = select_option()
            option = OPTIONS[selected_index]

            print(f"\n您选择了：{option['name']}")

            if "url" in option:
                print(f"正在打开链接：{option['url']}")
                webbrowser.open(option["url"])
                print()
                continue  # 返回菜单
            elif "exe" in option:
                target = os.path.join(SCRIPT_DIR, option["exe"])
                if not os.path.exists(target):
                    if not download_exe(target):
                        break  # 下载失败则退出
                subprocess.run([target], cwd=SCRIPT_DIR)
                break  # 启动 exe 后退出
            else:
                target = os.path.join(SCRIPT_DIR, option["file"])
                subprocess.run([PYTHON_EXECUTABLE, "-s", target], cwd=SCRIPT_DIR)
                break  # 启动服务后退出

    except KeyboardInterrupt:
        print("\n\n程序已中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        input("\n按任意键退出...")
        sys.exit(0)


if __name__ == "__main__":
    main()
