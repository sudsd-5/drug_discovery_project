import os
import subprocess

def run_md_simulation(pdb_file, output_dir, gpu_id=0):
    """运行GROMACS分子动力学模拟"""
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)

    # 生成拓扑文件
    subprocess.run(['gmx', 'pdb2gmx', '-f', pdb_file, '-o', 'aspirin.gro', '-water', 'spce', '-ff', 'oplsaa'], check=True)

    # 能量最小化
    with open('minim.mdp', 'w') as f:
        f.write("integrator = steep\nnsteps = 5000\n")
    subprocess.run(['gmx', 'grompp', '-f', 'minim.mdp', '-c', 'aspirin.gro', '-p', 'topol.top', '-o', 'em.tpr'], check=True)
    subprocess.run(['gmx', 'mdrun', '-v', '-deffnm', 'em', '-gpu_id', str(gpu_id)], check=True)

    # MD模拟
    with open('md.mdp', 'w') as f:
        f.write("integrator = md\nnsteps = 10000\ndt = 0.002\nnstenergy = 100\n")
    subprocess.run(['gmx', 'grompp', '-f', 'md.mdp', '-c', 'em.gro', '-p', 'topol.top', '-o', 'md.tpr'], check=True)
    subprocess.run(['gmx', 'mdrun', '-v', '-deffnm', 'md', '-gpu_id', str(gpu_id)], check=True)

if __name__ == "__main__":
    run_md_simulation('/root/workspace/data/aspirin.pdb', '/root/workspace/output/md')