#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
╔════════════════════════════════════════════════════════════════╗
║                  Stacking 演示菜单 - 交互式                   ║
║              Interactive Menu for Stacking Methods            ║
╚════════════════════════════════════════════════════════════════╝

功能：提供交互式菜单来运行各个演示脚本

使用方法：
python run_demos.py
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path

# 颜色定义
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    OKCYAN = '\033[96m'


# 脚本信息
SCRIPTS = {
    1: {
        'name': 'Stacking 基础演示',
        'file': '1_stacking_basic.py',
        'description': '学习 Stacking 的核心概念和基本用法',
        'features': [
            '- Stacking 分类器演示',
            '- Stacking 回归器演示',
            '- 基础学习器组合对比',
            '- Stacking vs 其他集成方法'
        ],
        'time': '5-10 分钟',
        'level': '初级'
    },
    2: {
        'name': 'Blending 演示',
        'file': '2_blending_demo.py',
        'description': '学习 Blending - Stacking 的简化版本',
        'features': [
            '- Blending 分类器（手动实现）',
            '- Blending 回归器',
            '- Blending vs Stacking 对比',
            '- 理解元特征生成过程'
        ],
        'time': '5-10 分钟',
        'level': '中级'
    },
    3: {
        'name': '多层 Stacking 演示',
        'file': '3_multilevel_stacking.py',
        'description': '学习多层集成 - 堆叠多个元学习器',
        'features': [
            '- 两层 Stacking',
            '- 三层 Stacking',
            '- 不同层数的性能对比',
            '- 复杂度分析'
        ],
        'time': '8-12 分钟',
        'level': '高级'
    },
    4: {
        'name': '综合对比分析',
        'file': '4_stacking_comprehensive.py',
        'description': '全面对比 Stacking 与其他集成方法',
        'features': [
            '- 不同元学习器的影响',
            '- 基础学习器多样性的重要性',
            '- 6 种集成方法全面对比',
            '- 性能与速度权衡分析'
        ],
        'time': '10-15 分钟',
        'level': '高级'
    },
    5: {
        'name': '查看文档',
        'file': 'README.md',
        'description': '查看项目的详细文档',
        'features': [
            '- Stacking 的完整说明',
            '- 参数调优指南',
            '- 最佳实践',
            '- 常见问题解答'
        ],
        'time': '20-30 分钟阅读',
        'level': '各级'
    },
    6: {
        'name': '快速开始指南',
        'file': 'QUICKSTART.md',
        'description': '快速开始和常用代码示例',
        'features': [
            '- 5 分钟快速开始',
            '- 常用代码示例',
            '- 参数速查表',
            '- 故障排查'
        ],
        'time': '10-15 分钟',
        'level': '初级'
    }
}


def check_dependencies():
    """检查依赖包"""
    required = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    print(f"\n{Colors.OKBLUE}检查依赖包...{Colors.ENDC}\n")
    
    for pkg, name in required.items():
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
            print(f"{Colors.FAIL}✗ {name} - 未安装{Colors.ENDC}")
        else:
            print(f"{Colors.OKGREEN}✓ {name} - 已安装{Colors.ENDC}")
    
    if missing:
        print(f"\n{Colors.WARNING}缺少包: {', '.join(missing)}{Colors.ENDC}")
        print(f"安装命令: pip install {' '.join(missing)}")
        return False
    else:
        print(f"\n{Colors.OKGREEN}所有依赖已安装！{Colors.ENDC}")
        return True


def print_header():
    """打印标题"""
    print(f"\n{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║                                                                ║")
    print("║            🤖 Stacking 集成学习 - 演示菜单 🤖                  ║")
    print("║                                                                ║")
    print("║              探索 Stacking, Blending, 多层集成                 ║")
    print("║                                                                ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}\n")


def print_menu():
    """打印菜单"""
    print(f"{Colors.BOLD}{Colors.OKCYAN}📋 请选择要运行的演示：{Colors.ENDC}\n")
    
    for num, info in SCRIPTS.items():
        level_colors = {
            '初级': Colors.OKGREEN,
            '中级': Colors.WARNING,
            '高级': Colors.FAIL
        }.get(info['level'], Colors.OKBLUE)
        
        print(f"{Colors.BOLD}{num}. {info['name']}{Colors.ENDC}")
        print(f"   📝 {info['description']}")
        print(f"   ⏱️  {info['time']}")
        print(f"   📊 {level_colors}{info['level']}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}0. 退出{Colors.ENDC}")
    print(f"{Colors.BOLD}a. 运行所有演示脚本 (1-4){Colors.ENDC}\n")


def print_script_details(num):
    """打印脚本详情"""
    if num not in SCRIPTS:
        return
    
    info = SCRIPTS[num]
    print(f"\n{Colors.OKBLUE}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{info['name']}{Colors.ENDC}")
    print(f"{Colors.OKBLUE}{'='*60}{Colors.ENDC}\n")
    
    print(f"{Colors.BOLD}📝 描述：{Colors.ENDC}{info['description']}\n")
    print(f"{Colors.BOLD}✨ 包含内容：{Colors.ENDC}")
    for feature in info['features']:
        print(f"  {feature}")
    
    print(f"\n{Colors.BOLD}⏱️  预计耗时：{Colors.ENDC}{info['time']}")
    print(f"{Colors.BOLD}📊 难度等级：{Colors.ENDC}{info['level']}\n")


def run_script(num):
    """运行脚本"""
    if num not in SCRIPTS:
        print(f"{Colors.FAIL}❌ 无效选择{Colors.ENDC}")
        return
    
    info = SCRIPTS[num]
    filename = info['file']
    
    # 文档处理
    if filename.endswith('.md'):
        print_script_details(num)
        print(f"{Colors.OKBLUE}💡 提示：可以用文本编辑器打开获得更好的阅读体验{Colors.ENDC}\n")
        input(f"{Colors.BOLD}按 Enter 键返回菜单...{Colors.ENDC}")
        return
    
    # 检查文件存在
    if not os.path.exists(filename):
        print(f"{Colors.FAIL}❌ 文件未找到：{filename}{Colors.ENDC}\n")
        return
    
    print_script_details(num)
    print(f"{Colors.BOLD}{Colors.OKGREEN}▶️  运行脚本：{filename}{Colors.ENDC}\n")
    print(f"{Colors.OKBLUE}{'='*60}{Colors.ENDC}\n")
    
    try:
        result = subprocess.run([sys.executable, filename], check=True)
        
        print(f"\n{Colors.OKBLUE}{'='*60}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}✓ 脚本运行完成！{Colors.ENDC}")
        print(f"\n{Colors.BOLD}生成的图表已保存到 {Colors.WARNING}./images/{Colors.ENDC} 目录\n")
    
    except subprocess.CalledProcessError as e:
        print(f"\n{Colors.FAIL}❌ 脚本运行出错{Colors.ENDC}")
        print(f"  错误代码: {e.returncode}\n")
    except Exception as e:
        print(f"{Colors.FAIL}❌ 发生错误: {e}{Colors.ENDC}\n")
    
    input(f"{Colors.BOLD}按 Enter 键返回菜单...{Colors.ENDC}")


def run_all_scripts():
    """运行所有脚本"""
    print(f"\n{Colors.OKCYAN}运行所有演示脚本...{Colors.ENDC}\n")
    
    for num in range(1, 5):
        info = SCRIPTS[num]
        filename = info['file']
        
        if filename.endswith('.md'):
            continue
        
        print(f"{Colors.OKGREEN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}执行脚本 {num}/4 - {info['name']}{Colors.ENDC}")
        print(f"{Colors.OKGREEN}{'='*60}{Colors.ENDC}\n")
        
        try:
            subprocess.run([sys.executable, filename], check=True)
        except Exception as e:
            print(f"{Colors.FAIL}❌ 脚本运行出错: {e}{Colors.ENDC}\n")
        
        if num < 4:
            print(f"\n{Colors.OKBLUE}准备运行下一个脚本...{Colors.ENDC}")
            import time
            time.sleep(1)
    
    print(f"\n{Colors.OKGREEN}{'='*60}{Colors.ENDC}")
    print(f"{Colors.OKGREEN}✅ 所有脚本运行完成！{Colors.ENDC}")
    print(f"{Colors.OKGREEN}{'='*60}{Colors.ENDC}\n")
    
    input(f"{Colors.BOLD}按 Enter 键返回菜单...{Colors.ENDC}")


def main():
    """主程序"""
    # 检查依赖
    if not check_dependencies():
        print(f"\n{Colors.WARNING}请先安装缺失的依赖包{Colors.ENDC}\n")
        return
    
    while True:
        print_header()
        print_menu()
        
        try:
            choice = input(f"{Colors.BOLD}{Colors.OKCYAN}请输入选择 (0-6, a)：{Colors.ENDC} ").strip().lower()
            
            if choice == '0':
                print(f"\n{Colors.OKGREEN}👋 感谢使用 Stacking 菜单！再见！{Colors.ENDC}\n")
                break
            
            elif choice == 'a':
                run_all_scripts()
            
            elif choice.isdigit() and 1 <= int(choice) <= 6:
                run_script(int(choice))
            
            else:
                print(f"\n{Colors.FAIL}❌ 无效输入{Colors.ENDC}\n")
                input(f"{Colors.BOLD}按 Enter 键继续...{Colors.ENDC}")
        
        except KeyboardInterrupt:
            print(f"\n\n{Colors.WARNING}⚠️  用户中断{Colors.ENDC}\n")
            break
        except Exception as e:
            print(f"{Colors.FAIL}❌ 发生错误: {e}{Colors.ENDC}\n")
            input(f"{Colors.BOLD}按 Enter 键继续...{Colors.ENDC}")


if __name__ == '__main__':
    main()
