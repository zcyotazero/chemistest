import streamlit as st
import math
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict

# ==========================================
# 页面全局配置
# ==========================================
st.set_page_config(page_title="Chemical Lab Pro | 高分子仿真引擎", layout="wide", page_icon="🔬")

# ==========================================
# 核心高分子材料物理化学数据库
# ==========================================
@dataclass
class PolymerMaterial:
    name: str
    type_class: str
    density: float
    hardness: float
    tensile: float
    friction: float
    tr10: float
    surface_energy: float

class MaterialDatabase:
    def __init__(self):
        self.db = {
            "FVMQ_HighPhenyl": PolymerMaterial("高苯基低氟共聚物 (FVMQ)", "Elastomer", 1.28, 45, 8.5, 0.22, -55.0, 22.0),
            "VMQ_Standard": PolymerMaterial("聚二甲基硅氧烷 (VMQ)", "BaseMatrix", 1.15, 60, 6.5, 0.65, -40.0, 28.0),
            "EPDM_Standard": PolymerMaterial("三元乙丙二烯单体 (EPDM)", "BaseMatrix", 1.05, 65, 10.0, 0.70, -30.0, 32.0),
            "PTFE_Powder": PolymerMaterial("聚四氟乙烯微粉 (PTFE)", "Filler", 2.20, 55, 5.0, 0.08, 0.0, 18.5),
            "Nano_Ceramic": PolymerMaterial("纳米二氧化硅/陶瓷基", "Filler", 2.50, 85, 15.0, 0.20, 0.0, 35.0),
            "Carbon_Black": PolymerMaterial("高分散炭黑 (N330)", "Filler", 1.80, 75, 18.0, 0.55, 0.0, 40.0)
        }
    def get(self, mat_id: str) -> PolymerMaterial:
        return self.db.get(mat_id)

# ==========================================
# 多物理场仿真引擎
# ==========================================
class SimulationEngine:
    def __init__(self, db: MaterialDatabase):
        self.db = db

    def calculate_properties(self, recipe: Dict[str, float]) -> Dict[str, float]:
        total_weight = sum(recipe.values())
        # 自动归一化，防止滑块数值相加不等于 100
        if total_weight == 0:
            return {"density": 0, "hardness": 0, "tensile": 0, "friction": 0, "tr10": 0, "surface_energy": 0}
        
        normalized_recipe = {k: (v / total_weight) * 100 for k, v in recipe.items()}
        props = {"density": 0.0, "hardness": 0.0, "tensile": 0.0, "friction": 0.0, "tr10": 0.0, "surface_energy": 0.0}
        
        for mat_id, pct in normalized_recipe.items():
            mat = self.db.get(mat_id)
            w_frac = pct / 100.0
            props["density"] += mat.density * w_frac
            props["hardness"] += mat.hardness * w_frac
        
        f_content = normalized_recipe.get("FVMQ_HighPhenyl", 0) + (normalized_recipe.get("PTFE_Powder", 0) * 1.5)
        base_friction = self.db.get("VMQ_Standard").friction if "VMQ_Standard" in normalized_recipe else 0.68
        
        if f_content > 15:
            props["friction"] = 0.22 + 0.15 * math.exp(-0.12 * (f_content - 15))
        else:
            props["friction"] = base_friction - (base_friction - 0.45) * (f_content / 15.0)

        base_tensile = sum([self.db.get(k).tensile * (v/100) for k, v in normalized_recipe.items() if self.db.get(k).type_class in ['BaseMatrix', 'Elastomer']])
        filler_content = normalized_recipe.get("Nano_Ceramic", 0) + normalized_recipe.get("Carbon_Black", 0)
        
        if filler_content <= 20:
            reinforcement = 1.0 + (filler_content * 0.06)
        else:
            reinforcement = 2.2 - ((filler_content - 20) * 0.03) 
        props["tensile"] = base_tensile * reinforcement

        fvmq_frac = normalized_recipe.get("FVMQ_HighPhenyl", 0) / 100.0
        vmq_frac = normalized_recipe.get("VMQ_Standard", 0) / 100.0
        if fvmq_frac + vmq_frac > 0:
            tr_tr10 = (fvmq_frac * self.db.get("FVMQ_HighPhenyl").tr10 + vmq_frac * self.db.get("VMQ_Standard").tr10) / (fvmq_frac + vmq_frac)
            props["tr10"] = tr_tr10 - (fvmq_frac * 12.0) 
        else:
            props["tr10"] = sum([self.db.get(k).tr10 * (v/100) for k, v in normalized_recipe.items() if self.db.get(k).type_class != 'Filler'])

        base_se = self.db.get("VMQ_Standard").surface_energy if "VMQ_Standard" in normalized_recipe else 32.0
        if f_content > 0:
            props["surface_energy"] = base_se - ((base_se - 19.0) * (1 - math.exp(-0.08 * f_content)))
        else:
            props["surface_energy"] = base_se

        return props

# ==========================================
# 智能解析系统
# ==========================================
class ChemicalEvaluator:
    @staticmethod
    def evaluate(props: Dict[str, float]) -> Dict[str, str]:
        eval_report = {}
        if props['friction'] < 0.28:
            eval_report["界面摩擦学 (Tribology)"] = "极低摩擦状态。有效抑制干刮振颤 (Judder) 与异响。"
        elif props['friction'] < 0.45:
            eval_report["界面摩擦学 (Tribology)"] = "中等阻力。具备基础润滑性，需配合机械骨架调校。"
        else:
            eval_report["界面摩擦学 (Tribology)"] = "高危！剪切应力过大，极易跳刷。"

        if props['tr10'] <= -45:
            eval_report["低温热力学 (Cryogenics)"] = f"深冷柔韧卓越 (相变点 {props['tr10']:.1f}°C)。非晶区大分子链段极寒下保持高自由度。"
        elif props['tr10'] <= -35:
            eval_report["低温热力学 (Cryogenics)"] = f"标准防冻 (相变点 {props['tr10']:.1f}°C)。可应对常规温带冬季。"

        if props['surface_energy'] < 22:
            eval_report["表面润湿性 (Surface Energy)"] = "超低表面能 (荷叶效应)。雨水呈现弹射剥离状态。"
        elif props['surface_energy'] < 26:
            eval_report["表面润湿性 (Surface Energy)"] = "优良疏水性。水膜易破裂凝珠。"

        return eval_report

# ==========================================
# Web 界面与渲染逻辑
# ==========================================
def main():
    st.title("🔬 Chemical Lab Pro v4.0")
    st.markdown("### Advanced Polymer Physics Simulation Engine (高级聚合物物理仿真)")
    
    db = MaterialDatabase()
    engine = SimulationEngine(db)
    evaluator = ChemicalEvaluator()

    # 左侧边栏：交互式配方调节器
    st.sidebar.header("🧪 实时配方调节 (Formulation)")
    st.sidebar.markdown("拖动滑块调节各材料的质量百分比：")
    
    recipe = {
        "FVMQ_HighPhenyl": st.sidebar.slider("FVMQ 高苯基特种氟硅 (%)", 0.0, 100.0, 15.0, step=1.0),
        "VMQ_Standard": st.sidebar.slider("VMQ 常规硅胶基体 (%)", 0.0, 100.0, 70.0, step=1.0),
        "PTFE_Powder": st.sidebar.slider("PTFE 纳米特氟龙粉 (%)", 0.0, 100.0, 5.0, step=1.0),
        "Nano_Ceramic": st.sidebar.slider("纳米陶瓷粉 (%)", 0.0, 100.0, 10.0, step=1.0),
        "EPDM_Standard": st.sidebar.slider("EPDM 三元乙丙橡胶 (%)", 0.0, 100.0, 0.0, step=1.0),
        "Carbon_Black": st.sidebar.slider("N330 高分散炭黑 (%)", 0.0, 100.0, 0.0, step=1.0)
    }
    
    total_input = sum(recipe.values())
    if total_input != 100.0:
        st.sidebar.warning(f"当前输入总和为 {total_input}%。系统将自动按比例归一化计算。")

    # 运行仿真
    props = engine.calculate_properties(recipe)
    chem_report = evaluator.evaluate(props)

    # 主体界面：指标看板
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="邵氏硬度 (Shore A)", value=f"{props['hardness']:.1f} HA")
    col2.metric(label="极限拉伸强度 (Tensile)", value=f"{props['tensile']:.1f} MPa")
    col3.metric(label="动摩擦系数 (Friction)", value=f"{props['friction']:.3f}")
    col4.metric(label="相变冰点 (TR-10)", value=f"{props['tr10']:.1f} °C")

    st.divider()

    # 图表与结论并排显示
    chart_col, text_col = st.columns([1.2, 1])

    with chart_col:
        st.subheader("📊 聚合物多维性能雷达图")
        labels = ['Tensile Strength\n(拉伸刚性)', 'Low Friction\n(极低摩擦学)', 
                  'Cryogenic Flex\n(极寒相变点)', 'Hardness Match\n(硬度匹配)', 'Hydrophobicity\n(疏水表面能)']
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]
        
        tensile_score = min(10, props['tensile'] / 1.8)        
        friction_score = max(0, 10 - (props['friction'] * 12)) 
        cold_score = min(10, abs(props['tr10']) / 5.5)         
        hard_score = max(0, 10 - abs(props['hardness'] - 60) / 2.5)
        hydro_score = max(0, 10 - (props['surface_energy'] - 18) / 1.5)
        values = [tensile_score, friction_score, cold_score, hard_score, hydro_score]
        values += values[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values, linewidth=2, color="#1f77b4", label="当前实时配方")
        ax.fill(angles, values, alpha=0.25, color="#1f77b4")
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        st.pyplot(fig)

    with text_col:
        st.subheader("🔬 化验室智能诊断报告")
        for k, v in chem_report.items():
            st.info(f"**{k}**\n\n{v}")
            
        st.markdown("---")
        st.markdown("### 物理参数速览表")
        st.markdown(f"- **表观密度**: {props['density']:.3f} g/cm³")
        st.markdown(f"- **表面张力**: {props['surface_energy']:.1f} mN/m")

if __name__ == "__main__":
    main()