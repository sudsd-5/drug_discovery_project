# 保存为 E:\AI\drug_discovery_project\1.py
import sqlite3
import pandas as pd
import random
import os

# 连接数据库
conn = sqlite3.connect(r"E:\AI\drug_discovery_project\data\raw\chembl\chembl_34_sqlite\chembl_34.db")

# 查询正样本（IC50活性），提取 20,000 条，包含 target_sequence
query = """
SELECT DISTINCT
    md.chembl_id AS chembl_id,
    cs.canonical_smiles AS smiles,
    td.chembl_id AS target_id,
    csq.sequence AS target_sequence,
    act.standard_value AS standard_value
FROM molecule_dictionary md
JOIN compound_structures cs ON md.molregno = cs.molregno
JOIN activities act ON md.molregno = act.molregno
JOIN assays a ON act.assay_id = a.assay_id
JOIN target_dictionary td ON a.tid = td.tid
JOIN target_components tc ON td.tid = tc.tid
JOIN component_sequences csq ON tc.component_id = csq.component_id
WHERE act.standard_type = 'IC50'
AND cs.canonical_smiles IS NOT NULL
AND act.standard_value IS NOT NULL
AND csq.sequence IS NOT NULL
LIMIT 20000
"""
df_positive = pd.read_sql_query(query, conn)
df_positive["label"] = 1

# 生成负样本，20,000 条
drugs = df_positive["chembl_id"].unique().tolist()
targets = df_positive["target_id"].unique().tolist()
sequences = dict(zip(df_positive["target_id"], df_positive["target_sequence"]))  # 目标序列映射
positive_pairs = set(zip(df_positive["chembl_id"], df_positive["target_id"]))
negative_data = []
for _ in range(20000):
    drug = random.choice(drugs)
    target = random.choice(targets)
    if (drug, target) not in positive_pairs:
        negative_data.append({
            "chembl_id": drug,
            "smiles": df_positive[df_positive["chembl_id"] == drug]["smiles"].iloc[0],
            "target_id": target,
            "target_sequence": sequences[target],
            "standard_value": "N/A",
            "label": 0
        })

# 合并数据
df_negative = pd.DataFrame(negative_data)
df = pd.concat([df_positive, df_negative], ignore_index=True)

# 确保目录存在
output_dir = r"E:\AI\drug_discovery_project\data\raw\chembl"
os.makedirs(output_dir, exist_ok=True)

# 保存到绝对路径
output_path = r"E:\AI\drug_discovery_project\data\raw\chembl\molecular_properties.csv"
df.to_csv(output_path, index=False)
print(f"Generated {len(df)} samples, saved to {output_path}")
conn.close()