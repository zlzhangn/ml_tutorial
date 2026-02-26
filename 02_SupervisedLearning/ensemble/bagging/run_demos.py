#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════╗
║                 Bagging 集成学习演示菜单                      ║
║              Interactive Menu for Bagging Methods             ║
╚════════════════════════════════════════════════════════════════╝

功能说明：
1. 交互式菜单选择要运行的演示
2. 自动依赖检查
3. 运行相应的演示脚本
4. 显示运行结果

使用方法：
python run_demos.py

需要的库：
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 颜色和格式设置
# ─────────────────────────────────────────────────────────────

class Colors:
    """颜色定义类"""
    HEADER = '\033[95m'      # 紫色 - 标题
    OKBLUE = '\033[94m'      # 蓝色 - 提示
    OKCYAN = '\033[96m'      # 青色
    OKGREEN = '\033[92m'     # 绿色 - 成功
    WARNING = '\033[93m'     # 黄色 - 警告
    FAIL = '\033[91m'        # 红色 - 错误
    ENDC = '\033[0m'         # 结束
    BOLD = '\033[1m'         # 加粗
    UNDERLINE = '\033[4m'    # 下划线


# ─────────────────────────────────────────────────────────────
# 依赖检查函数
# ─────────────────────────────────────────────────────────────

def check_dependencies():
    """
    检查是否安装了所有必需的依赖包
    
    需要的包：
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scikit-learn
    """
    required_packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'scikit-learn'
    }
    
    missing_packages = []
    
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}检查依赖包...{Colors.ENDC}\n")
    
    for package, display_name in required_packages.items():
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing_packages.append(package)
            print(f"{Colors.FAIL}✗ {display_name} - 未安装{Colors.ENDC}")
        else:
            print(f"{Colors.OKGREEN}✓ {display_name} - 已安装{Colors.ENDC}")
    
    if missing_packages:
        print(f"\n{Colors.WARNING}缺少以下包：{', '.join(missing_packages)}{Colors.ENDC}")
        print(f"\n{Colors.BOLD}安装缺失包：{Colors.ENDC}")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"\n{Colors.OKGREEN}所有依赖包已安装！{Colors.ENDC}")
        return True


# ─────────────────────────────────────────────────────────────
# 脚本信息定义
# ─────────────────────────────────────────────────────────────

SCRIPTS = {
    1: {
        'name': 'Bagging 基础演示',
        'file': '1_bagging_demo.py',
        'description': '学习 Bagging 的核心概念和基本用法',
        'features': [
            '- Bagging 分类器演示',
            '- 估计器数量影响分析',
            '- 回归问题演示',
            '- 基础分类器对比'
        ],
        'time': '5-10分钟',
        'level': '初级'
    },
    2: {
        'name': 'Random Forest 演示',
        'file': '2_random_forest_demo.py',
        'description': '学习最常用的集成方法 - 随机森林',
        'features': [
            '- Random Forest 分类和回归',
            '- OOB 评分演示',
            '- 特征重要性分析',
            '- max_features 参数对比'
        ],
        'time': '8-12分钟',
        'level': '中级'
    },
    3: {
        'name': 'Extra Trees 演示',
        'file': '3_extra_trees_demo.py',
        'description': '学习快速的集成方法 - 极端随机树',
        'features': [
            '- Extra Trees 分类和回归',
            '- 速度对比（RF vs ET）',
            '- 大数据集处理',
            '- 分割策略对比'
        ],
        'time': '8-12分钟',
        'level': '中级'
    },
    4: {
        'name': '全面对比演示',
        'file': '4_bagging_comparison.py',
        'description': '对比所有四种 Bagging 方法的性能',
        'features': [
            '- Bagging vs Random Forest vs Extra Trees vs Pasting',
            '- 性能指标对比（准确率、精确率、召回率等）',
            '- 速度 vs 准确率权衡分析',
            '- 多维对比（雷达图）'
        ],
        'time': '10-15分钟',
        'level': '高级'
    },
    5: {
        'name': '查看文档',
        'file': 'README.md',
        'description': '查看项目的详细文档',
        'features': [
            '- 完整的方法介绍',
            '- 参数调优指南',
            '- 常见问题解答',
            '- 性能对比表'
        ],
        'time': '20-30分钟阅读',
        'level': '各级'
    },
    6: {
        'name': '快速开始指南',
        'file': 'QUICKSTART.md',
        'description': '快速开始和常用代码示例',
        'features': [
            '- 5分钟快速开始',
            '- 常用参数速查',
            '- 代码示例',
            '- 故障排查'
        ],
        'time': '10-15分钟',
        'level': '初级'
    }
}


# ─────────────────────────────────────────────────────────────
# UI 函数 - 菜单显示和交互
# ─────────────────────────────────────────────────────────────

def print_header():
    """打印项目标题"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║         🤖 Bagging 集成学习 - 演示和学习菜单 🤖               ║")
    print("║                                                                ║")
    print("║     探索 Bagging, Random Forest, Extra Trees, Pasting         ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")


def print_menu():
    """打印菜单选项"""
    print(f"{Colors.BOLD}{Colors.OKCYAN}📋 请选择要运行的演示：{Colors.ENDC}\n")
    
    for num, script_info in SCRIPTS.items():
        level_color = {
            '初级': Colors.OKGREEN,
            '中级': Colors.WARNING,
            '高级': Colors.FAIL
        }.get(script_info['level'], Colors.OKBLUE)
        
        print(f"{Colors.BOLD}{num}. {script_info['name']}{Colors.ENDC}")
        print(f"   📝 {script_info['description']}")
        print(f"   ⏱️  耗时：{script_info['time']}")
        print(f"   📊 等级：{level_color}{script_info['level']}{Colors.ENDC}")
        print()
    
    print(f"{Colors.BOLD}0. 退出程序{Colors.ENDC}")
    print(f"{Colors.BOLD}a. 运行所有演示脚本（1-4）{Colors.ENDC}")
    print()


def print_script_details(script_num):
    """打印脚本的详细信息"""
    if script_num not in SCRIPTS:
        return
    
    script = SCRIPTS[script_num]
    
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{script['name']}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKBLUE}{'='*60}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}📝 描述：{Colors.ENDC}{script['description']}\n")
    
    print(f"{Colors.BOLD}✨ 包含内容：{Colors.ENDC}")
    for feature in script['features']:
        print(f"  {feature}")
    
    print(f"\n{Colors.BOLD}⏱️  预计耗时：{Colors.ENDC}{script['time']}")
    print(f"{Colors.BOLD}📊 难度等级：{Colors.ENDC}{script['level']}\n")


def open_file(filename):
    """用默认程序打开文件（文档）"""
    file_path = Path(filename)
    
    if not file_path.exists():
        print(f"{Colors.FAIL}❌ 文件未找到：{filename}{Colors.ENDC}")
        return
    
    try:
        import platform
        import subprocess
        
        system = platform.system()
        
        if system == 'Darwin':  # macOS
            subprocess.Popen(['open', str(file_path)])
        elif system == 'Windows':
            os.startfile(str(file_path))
        else:  # Linux
            subprocess.Popen(['xdg-open', str(file_path)])
        
        print(f"\n{Colors.OKGREEN}✓ 已用默认程序打开文件：{filename}{Colors.ENDC}")
        print(f"  (如果没有自动打开，请手动打开：{file_path})\n")
    
    except Exception as e:
        print(f"\n{Colors.WARNING}⚠️  无法自动打开文件{Colors.ENDC}")
        print(f"  请手动打开：{file_path}\n")


def run_script(script_num):
    """运行指定的脚本"""
    if script_num not in SCRIPTS:
        print(f"{Colors.FAIL}❌ 无效的选择{Colors.ENDC}")
        return
    
    script = SCRIPTS[script_num]
    filename = script['file']
    
    # 文档类型的特殊处理
    if filename.endswith('.md'):
        print_script_details(script_num)
        print(f"{Colors.OKBLUE}💡 提示：可以在文本编辑器中打开以获得更好的阅读体验{Colors.ENDC}\n")
        
        open_choice = input(f"{Colors.BOLD}是否尝试打开文件？(y/n)：{Colors.ENDC} ").lower()
        if open_choice == 'y':
            open_file(filename)
        return
    
    # 检查脚本文件是否存在
    if not os.path.exists(filename):
        print(f"\n{Colors.FAIL}❌ 脚本文件未找到：{filename}{Colors.ENDC}")
        print(f"  请确保在正确的目录中运行此程序\n")
        return
    
    print_script_details(script_num)
    
    # 运行脚本
    print(f"{Colors.BOLD}{Colors.OKGREEN}▶️  运行脚本：{filename}{Colors.ENDC}\n")
    print(f"{Colors.OKBLUE}{'='*60}{Colors.ENDC}\n")
    
    try:
        result = subprocess.run([sys.executable, filename], check=True)
        
        print(f"\n{Colors.OKBLUE}{'='*60}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}✓ 脚本运行完成！{Colors.ENDC}")
        print(f"\n{Colors.BOLD}生成的图表已保存到 {Colors.WARNING}images/{Colors.ENDC} 目录\n")
    
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.OKBLUE}{'='*60}{Colors.ENDC}")
        print(f"{Colors.FAIL}❌ 脚本运行出错{Colors.ENDC}")
        print(f"  错误代码：{e.returncode}\n")
    
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ 发生错误：{e}{Colors.ENDC}\n")


def run_all_scripts():
    """顺序运行所有演示脚本"""
    print(f"\n{Colors.BOLD}{Colors.OKCYAN}运行所有演示脚本...{Colors.ENDC}\n")
    
    for num in range(1, 5):  # 运行脚本 1-4
        print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.OKGREEN}执行脚本 {num}/4 - {SCRIPTS[num]['name']}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.OKGREEN}{'='*60}{Colors.ENDC}\n")
        
        run_script(num)
        
        if num < 4:
            print(f"\n{Colors.OKBLUE}准备运行下一个脚本...{Colors.ENDC}")
            try:
                import time
                time.sleep(2)
            except:
                pass
    
    print(f"\n{Colors.BOLD}{Colors.OKGREEN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✅ 所有脚本运行完成！{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.OKGREEN}{'='*60}{Colors.ENDC}\n")


# ─────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────

def main():
    """主程序循环"""
    
    # 检查依赖
    if not check_dependencies():
        print(f"\n{Colors.BOLD}{Colors.WARNING}请先安装缺失的依赖包后再运行此程序。{Colors.ENDC}\n")
        input("按 Enter 键退出...")
        return
    
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input(f"{Colors.BOLD}{Colors.OKCYAN}请输入选择 (0-6, a)：{Colors.ENDC} ").strip().lower()
            
            if choice == '0':
                print(f"\n{Colors.OKGREEN}👋 感谢使用 Bagging 演示菜单！再见！{Colors.ENDC}\n")
                break
            
            elif choice == 'a':
                run_all_scripts()
                input(f"\n{Colors.BOLD}按 Enter 键返回菜单...{Colors.ENDC}")
            
            elif choice.isdigit() and 1 <= int(choice) <= 6:
                run_script(int(choice))
                input(f"\n{Colors.BOLD}按 Enter 键返回菜单...{Colors.ENDC}")
            
            else:
                print(f"{Colors.FAIL}❌ 无效的输入，请重试{Colors.ENDC}\n")
                input(f"{Colors.BOLD}按 Enter 键继续...{Colors.ENDC}")
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}⚠️  用户中断程序{Colors.ENDC}\n")
            break
        except Exception as e:
            print(f"{Colors.FAIL}❌ 发生错误：{e}{Colors.ENDC}\n")
            input(f"{Colors.BOLD}按 Enter 键继续...{Colors.ENDC}")


if __name__ == '__main__':
    main()
