#!/usr/bin/env python3
"""
启动脚本
"""
import subprocess
import sys
import os

def main():
    # 检查是否在虚拟环境中
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("请先激活虚拟环境：")
        print("source venv/bin/activate")
        return

    # 启动Flask应用
    try:
        subprocess.run([sys.executable, 'app.py'], check=True)
    except KeyboardInterrupt:
        print("\n应用已停止")
    except subprocess.CalledProcessError as e:
        print(f"启动失败: {e}")

if __name__ == '__main__':
    main()



