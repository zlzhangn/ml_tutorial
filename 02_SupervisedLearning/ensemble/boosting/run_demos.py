# -*- coding: utf-8 -*-
"""
Boosting 演示快速导航与说明

这个脚本提供了一个交互式菜单来运行各个 Boosting 演示脚本。
"""

import os
import subprocess
import sys

def print_banner():
    """打印欢迎横幅"""
    print("\n" + "="*70)
    print(" "*20 + "Boosting 集成学习演示")
    print("="*70)
    print("\n本文件夹包含四种主要 Boosting 方法的演示脚本：")
    print("  1. AdaBoost - 基础 Boosting 算法")
    print("  2. Gradient Boosting - 梯度提升方法")
    print("  3. XGBoost - 优化的梯度提升（可选）")
    print("  4. LightGBM - 轻量级快速梯度提升（可选）")
    print("  5. 综合对比 - 四种方法的全面对比")
    print("\n" + "="*70)


def print_menu():
    """打印菜单选项"""
    print("\n请选择要运行的演示：")
    print("-" * 70)
    print("1. AdaBoost 演示")
    print("   展示基础的自适应 Boosting 方法")
    print("   ")
    print("2. Gradient Boosting 演示")
    print("   展示梯度提升方法在分类和回归中的应用")
    print("   ")
    print("3. XGBoost 演示")
    print("   展示优化的梯度提升方法（需要安装 xgboost）")
    print("   ")
    print("4. LightGBM 演示")
    print("   展示轻量级快速梯度提升方法（需要安装 lightgbm）")
    print("   ")
    print("5. 综合对比")
    print("   展示所有 Boosting 方法的全面对比")
    print("   ")
    print("6. 查看 README（说明文档）")
    print("   查看详细的文档说明")
    print("   ")
    print("0. 退出")
    print("-" * 70)


def run_demo(script_name):
    """运行指定的演示脚本"""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"\n✗ 错误：找不到文件 {script_name}")
        return
    
    print(f"\n正在运行 {script_name}...")
    print("="*70)
    
    try:
        if sys.platform.startswith('win'):
            # Windows 系统
            subprocess.run([sys.executable, script_path], check=True)
        else:
            # Linux/Mac 系统
            subprocess.run([sys.executable, script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 演示运行失败，错误代码：{e.returncode}")
    except Exception as e:
        print(f"\n✗ 运行失败：{str(e)}")


def view_readme():
    """查看 README 文件"""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    
    if not os.path.exists(readme_path):
        print("\n✗ 错误：找不到 README.md 文件")
        return
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("\n" + "="*70)
        print("README - Boosting 演示说明")
        print("="*70 + "\n")
        print(content)
    except Exception as e:
        print(f"\n✗ 读取 README 失败：{str(e)}")


def check_dependencies():
    """检查依赖库的安装情况"""
    print("\n检查依赖库...")
    print("-" * 70)
    
    dependencies = {
        'numpy': '基础数值计算',
        'pandas': '数据处理',
        'matplotlib': '绘图库',
        'seaborn': '统计绘图',
        'scikit-learn': 'Boosting 基础库'
    }
    
    optional = {
        'xgboost': 'XGBoost 库',
        'lightgbm': 'LightGBM 库'
    }
    
    print("必需库：")
    for lib, desc in dependencies.items():
        try:
            __import__(lib)
            print(f"  ✓ {lib:<15} ({desc})")
        except ImportError:
            print(f"  ✗ {lib:<15} ({desc}) - 未安装")
    
    print("\n可选库：")
    for lib, desc in optional.items():
        try:
            __import__(lib)
            print(f"  ✓ {lib:<15} ({desc})")
        except ImportError:
            print(f"  ✗ {lib:<15} ({desc}) - 未安装（演示可用，但功能受限）")
    
    print("\n安装缺失的库：")
    print("  pip install numpy pandas matplotlib seaborn scikit-learn")
    print("  pip install xgboost lightgbm  # 可选，用于完整演示")
    print("-" * 70)


def main():
    """主函数"""
    print_banner()
    check_dependencies()
    
    while True:
        print_menu()
        choice = input("\n请输入选项 (0-6)：").strip()
        
        if choice == '1':
            run_demo('1_adaboost_demo.py')
        elif choice == '2':
            run_demo('2_gradient_boosting_demo.py')
        elif choice == '3':
            run_demo('3_xgboost_demo.py')
        elif choice == '4':
            run_demo('4_lightgbm_demo.py')
        elif choice == '5':
            run_demo('5_comprehensive_comparison.py')
        elif choice == '6':
            view_readme()
        elif choice == '0':
            print("\n感谢使用 Boosting 演示程序！")
            print("="*70 + "\n")
            break
        else:
            print("\n✗ 无效的选项，请重新输入")
        
        input("\n按 Enter 键继续...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n演示被中断")
        sys.exit(0)
