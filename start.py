#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
药物-靶点相互作用预测项目启动脚本
一键运行完整流程
"""

import os
import subprocess
import time

def print_header(message):
    """打印带有格式的标题"""
    border = "=" * 80
    print(f"\n{border}")
    print(f"{message.center(80)}")
    print(f"{border}\n")

def check_environment():
    """检查环境"""
    print_header("检查环境")
    
    # 检查必要目录
    for directory in ["data", "configs", "logs", "models", "runs"]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")
    
    # 检查数据目录
    for directory in ["data/raw", "data/raw/drugbank", "data/processed"]:
        path = os.path.normpath(directory)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            print(f"创建目录: {path}")
    
    # 检查配置文件
    config_path = os.path.join("configs", "dti_config.yaml")
    if not os.path.exists(config_path):
        print(f"错误: 缺少配置文件 {config_path}")
        return False
    
    print("环境检查完成！")
    return True

def check_data_files():
    """检查数据文件"""
    print_header("检查数据文件")
    
    # 检查必要的数据文件
    molecular_properties = os.path.join("data", "raw", "molecular_properties.csv")
    drug_target_interactions = os.path.join("data", "raw", "drugbank", "drug_target_interactions.csv")
    
    if not os.path.exists(molecular_properties):
        print(f"警告: 缺少分子性质数据文件 {molecular_properties}")
        print("请确保将分子性质数据文件放置在正确位置")
        return False
    
    if not os.path.exists(drug_target_interactions):
        print(f"警告: 缺少药物-靶点相互作用数据文件 {drug_target_interactions}")
        print("请确保将药物-靶点相互作用数据文件放置在正确位置")
        return False
    
    print(f"分子性质数据文件: {molecular_properties} [已找到]")
    print(f"药物-靶点相互作用数据文件: {drug_target_interactions} [已找到]")
    print("数据文件检查完成！")
    return True

def run_command(command, description):
    """运行命令并打印输出"""
    print_header(description)
    
    print(f"执行命令: {command}")
    start_time = time.time()
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # 实时打印输出
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    end_time = time.time()
    
    if process.returncode == 0:
        print(f"\n命令成功完成! 耗时: {end_time - start_time:.2f} 秒")
        return True
    else:
        print(f"\n命令失败，退出代码: {process.returncode}")
        return False

def main():
    """主函数"""
    print_header("药物-靶点相互作用预测项目")
    print("这个脚本将自动运行完整的药物-靶点相互作用预测流程")
    
    # 检查环境
    if not check_environment():
        return
    
    # 检查数据文件
    if not check_data_files():
        user_input = input("是否继续运行？(y/n): ")
        if user_input.lower() != 'y':
            return
    
    # 数据处理
    if not run_command("python src/process_drug_target_interactions.py", "数据处理"):
        print("数据处理失败，停止流程")
        return
    
    # 模型训练
    if not run_command("python src/train_dti.py", "模型训练"):
        print("模型训练失败，停止流程")
        return
    
    print_header("流程完成")
    print("药物-靶点相互作用预测流程已完成！")
    print("训练结果保存在 models/ 目录")
    print("可以使用 TensorBoard 查看训练过程: tensorboard --logdir runs")

if __name__ == "__main__":
    main() 