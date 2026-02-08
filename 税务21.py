import numpy as np
import pandas as pd
import networkx as nx
import json
import os
import time
from scipy import sparse
from datetime import datetime

# ================= 配置与路径 =================
# 所有产出文件将保存在桌面，方便您后续查看和处理
DESKTOP_PATH = r"C:\Users\wlw\Desktop"
os.makedirs(DESKTOP_PATH, exist_ok=True)

# 实验规模设定：20,000个企业节点，每个节点具有50维特征
N_AGENTS = 20000
N_FEATURES = 50
TIMESTEPS = 12  # 模拟过去12个月的财务序列

print(f"[*] 启动 LegiAgent 数据工厂 - 目标规模: {N_AGENTS} 节点")

class LegiDataGenerator:
    """
    KDD 级别数据集生成器：模拟现实世界供应链的复杂性与法律知识图谱的逻辑关联。
    """
    
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.graph = None
        self.features_df = None
        self.legal_dag = None

    def generate_supply_chain_topology(self):
        """
        构建无标度网络 (Scale-Free Network)：
        现实中的供应链节点度数服从幂律分布，少数核心企业连接大量供应商。
        使用 Barabasi-Albert 模型生成，以确保网络具有“富者愈富”的特征。
        """
        print("[1/4] 正在构建无标度供应链拓扑 (Barabasi-Albert Graph)...")
        # m=3 表示每个新加入节点连接3个已有节点，模拟稳定的供应关系
        self.graph = nx.barabasi_albert_graph(self.n_nodes, m=3)
        
        # 将邻接矩阵转为稀疏格式存储（模拟大规模图计算场景）
        adj_sparse = nx.adjacency_matrix(self.graph)
        sparse_path = os.path.join(DESKTOP_PATH, "kdd_exp_adj.npz")
        sparse.save_npz(sparse_path, adj_sparse)
        print(f"    - 拓扑连接完成，保存至: {sparse_path}")

    def generate_high_dimensional_features(self):
        """
        生成高维特征向量：
        包含基础属性、动态财务时间序列以及隐藏的合规倾向。
        """
        print("[2/4] 正在生成 50 维高维特征矩阵与财务序列...")
        data = []
        for i in range(self.n_nodes):
            # 1. 基础属性
            sic_code = np.random.choice([10, 20, 30, 40, 50, 70]) * 100  # 模拟SIC行业代码
            reg_capital = np.random.lognormal(mean=10, sigma=2)  # 注册资本符合对数正态分布
            age = np.random.randint(1, 30)
            
            # 2. 隐藏变量：合规倾向 (Compliance Propensity)
            # 使用 Beta 分布模拟，大多数企业合规，少数极不合规
            compliance_propensity = np.random.beta(a=8, b=2)
            
            # 3. 动态财务时间序列 (12个月)
            # 资产负债表与利润表核心指标：收入、利润、负债
            # 每一个月的数据都作为特征维度
            monthly_revenue = np.random.normal(loc=1000, scale=200, size=TIMESTEPS).cumsum()
            monthly_debt = monthly_revenue * np.random.uniform(0.3, 0.8, size=TIMESTEPS)
            
            # 拼凑成 50 维向量
            # 前12维：收入序列；次12维：负债序列；其余：基础属性与噪声特征
            node_features = np.concatenate([
                monthly_revenue, 
                monthly_debt,
                [sic_code, reg_capital, age, compliance_propensity],
                np.random.randn(N_FEATURES - (TIMESTEPS*2 + 4)) # 补充噪声模拟复杂环境
            ])
            data.append(node_features)

        columns = [f"feat_{i}" for i in range(N_FEATURES)]
        self.features_df = pd.DataFrame(data, columns=columns)
        self.features_df['node_id'] = range(self.n_nodes)
        
        # 保存为 Parquet 格式 (大数据标准存储)
        parquet_path = os.path.join(DESKTOP_PATH, "kdd_exp_features.parquet")
        self.features_df.to_parquet(parquet_path, index=False)
        print(f"    - 特征数据生成完成，保存至: {parquet_path}")

    def generate_legal_knowledge_graph(self):
        """
        构建法律知识图谱 (DAG)：
        模拟 500 条法律条文之间的引用、互斥关系。
        """
        print("[3/4] 正在构建法律逻辑 DAG (Directed Acyclic Graph)...")
        legal_dag = nx.gnp_random_graph(500, 0.005, directed=True)
        # 强制转换为 DAG（只保留上三角连接）
        legal_dag = nx.DiGraph([(u, v) for (u, v) in legal_dag.edges() if u < v])
        
        legal_data = {}
        for node in legal_dag.nodes():
            legal_data[node] = {
                "id": f"LAW_{node}",
                "logic_predicate": f"IF tax_avoidance > {np.random.uniform(0.1, 0.5):.2f} THEN penalty=1",
                "references": list(legal_dag.successors(node)),
                "sector_scope": int(np.random.choice([1000, 2000, 3000]))
            }
        
        kg_path = os.path.join(DESKTOP_PATH, "kdd_exp_legal_kg.json")
        with open(kg_path, 'w', encoding='utf-8') as f:
            json.dump(legal_data, f, indent=4)
        print(f"    - 法律知识图谱生成完成，保存至: {kg_path}")

    def expand_data_to_gb(self):
        """
        数据扩展：通过增加冗余特征描述和精细化时间序列，
        将文件体积扩展至接近 1GB，以模拟工业级实验环境。
        """
        print("[4/4] 正在执行数据扩展以模拟 GB 级大数据量...")
        # 复制现有特征并增加冗余维度（模拟未处理的原始审计日志）
        expanded_df = pd.concat([self.features_df] * 10, ignore_index=True)
        # 模拟高频交易日志属性（每家企业增加 100 维历史记录）
        extra_data = np.random.randn(len(expanded_df), 100)
        extra_cols = [f"audit_log_{i}" for i in range(100)]
        expanded_df = pd.concat([expanded_df, pd.DataFrame(extra_data, columns=extra_cols)], axis=1)
        
        big_data_path = os.path.join(DESKTOP_PATH, "kdd_exp_big_data_raw.parquet")
        expanded_df.to_parquet(big_data_path, index=False)
        print(f"    - [警告] 大规模原始数据已生成: {big_data_path}")
        print(f"    - 文件大小约: {os.path.getsize(big_data_path) / (1024**2):.2f} MB")

if __name__ == "__main__":
    start = time.time()
    generator = LegiDataGenerator(N_AGENTS)
    generator.generate_supply_chain_topology()
    generator.generate_high_dimensional_features()
    generator.generate_legal_knowledge_graph()
    generator.expand_data_to_gb()
    
    print("\n" + "="*50)
    print(f"实验第一阶段成功完成！")
    print(f"总耗时: {time.time() - start:.2f} 秒")
    print("您现在桌面上拥有了一个具有复杂拓扑关联、高维特征和法律逻辑的‘重型’数据集。")
    print("="*50)