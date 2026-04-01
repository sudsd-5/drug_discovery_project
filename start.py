#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
药物-靶点相互作用预测项目启动脚本
一键运行完整流程
"""

import os
import subprocess
import time
import sys
import shutil

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
    for directory in ["data", "configs", "logs", "models", "runs", "data/raw", "data/raw/drugbank", "data/processed"]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"创建目录: {directory}")
    
    # 检查配置文件
    config_path = os.path.join("configs", "dti_config.yaml")
    if not os.path.exists(config_path):
        if os.path.exists("dti_config.yaml"):
            print(f"将 dti_config.yaml 复制到 {config_path}")
            shutil.copy("dti_config.yaml", config_path)
        else:
            print(f"错误: 缺少配置文件 {config_path}")
            return False
    
    print("环境检查完成！")
    return True

def create_dummy_data():
    """创建模拟数据用于调试"""
    print_header("创建模拟数据")
    
    molecular_properties = os.path.join("data", "raw", "molecular_properties.csv")
    drug_target_interactions = os.path.join("data", "raw", "drugbank", "drug_target_interactions.csv")
    
    if not os.path.exists(molecular_properties):
        with open(molecular_properties, 'w') as f:
            f.write("id,smiles,logP,MW\n")
            f.write("1,CCO,0.5,46.07\n")
            f.write("2,c1ccccc1,2.1,78.11\n")
            f.write("3,CC(=O)Oc1ccccc1C(=O)O,1.19,180.16\n")
            f.write("4,CN1C=NC2=C1C(=O)N(C(=O)N2C)C,-0.07,194.19\n")
            f.write("5,C1=CC=C(C=C1)CC(C(=O)O)N,1.4,165.19\n")
        print(f"已创建: {molecular_properties}")

    if not os.path.exists(drug_target_interactions):
        with open(drug_target_interactions, 'w') as f:
            f.write("smiles,target_sequence,label\n")
            f.write("CCO,MAAAAAAAAAA,inhibitor\n")
            f.write("c1ccccc1,MBBBBBBBBBB,binder\n")
            f.write("CC(=O)Oc1ccccc1C(=O)O,MCCCCCCCCCC,ligand\n")
            f.write("CN1C=NC2=C1C(=O)N(C(=O)N2C)C,MDDDDDDDDDD,none\n")
            f.write("C1=CC=C(C=C1)CC(C(=O)O)N,MEEEEEEEEEE,inhibitor\n")
        print(f"已创建: {drug_target_interactions}")

def check_data_files():
    """检查数据文件"""
    print_header("检查数据文件")
    
    # 检查必要的数据文件
    molecular_properties = os.path.join("data", "raw", "molecular_properties.csv")
    drug_target_interactions = os.path.join("data", "raw", "drugbank", "drug_target_interactions.csv")
    
    if not os.path.exists(molecular_properties) or not os.path.exists(drug_target_interactions):
        create_dummy_data()
    
    print(f"分子性质数据文件: {molecular_properties} [已找到]")
    print(f"药物-靶点相互作用数据文件: {drug_target_interactions} [已找到]")
    print("数据文件检查完成！")
    return True

def run_command(command, description):
    """运行命令并打印输出"""
    print_header(description)
    
    # 使用当前的 python 解释器
    full_command = f"{sys.executable} {command}"
    print(f"执行命令: {full_command}")
    
    start_time = time.time()
    process = subprocess.Popen(
        full_command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # 实时打印输出
    if process.stdout:
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
    check_data_files()
    
    # 数据处理
    if not run_command("process_drug_target_interactions.py", "数据处理"):
        print("数据处理失败，停止流程")
        return
    
    # 模型训练
    if not run_command("train_dti.py", "模型训练"):
        print("模型训练失败，停止流程")
        return
    
    print_header("流程完成")
    print("药物-靶点相互作用预测流程已完成！")
    print("训练结果保存在 models/ 目录")
    print("可以使用 TensorBoard 查看训练过程: tensorboard --logdir runs")

if __name__ == "__main__":
    main()
