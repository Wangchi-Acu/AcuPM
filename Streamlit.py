import streamlit as st
import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 加载模型
xgb_model = joblib.load('XGBoost.pkl')

# 示例数据的特征名称
feature_names = ["DUR", "LPRDR", "N1P", "N1LOL", "N3P", "DOM", "MAPTS", "BSDS", "TNM"]

# 创建SHAP解释器
explainer = shap.TreeExplainer(xgb_model)

# Streamlit 用户界面
st.title("“通督调神”针法治疗失眠症疗效预测")

# 创建用户输入界面
DUR = st.number_input("病程（月）:", min_value=0.0, max_value=100.0, value=1.0)
LPRDR = st.number_input("记录期间最低脉率（次/分钟）:", min_value=0.0, max_value=200.0, value=1.0)
N1P = st.number_input("N1期占总睡眠时长比例（%）:", min_value=0.0, max_value=100.0, value=1.0)
N1LOL = st.number_input("自关灯起的N1期潜伏期（分钟）:", min_value=0.0, max_value=500.0, value=1.0)
N3P = st.number_input("N3期占总睡眠时长比例（%）:", min_value=0.0, max_value=100.0, value=1.0)
DOM = st.number_input("微觉醒持续时间（分钟）:", min_value=0.0, max_value=100.0, value=1.0)
MAPTS = st.number_input("微觉醒占总睡眠时长比例（%）:", min_value=0.0, max_value=100.0, value=1.0)
BSDS = st.number_input("自评抑郁量表得分（分）:", min_value=0.0, max_value=100.0, value=1.0)
TNM = st.number_input("微觉醒总次数（次）:", min_value=0, max_value=1000, value=1)

# 将用户输入的值作为特征向量
feature_values = np.array([[DUR, LPRDR, N1P, N1LOL, N3P, DOM, MAPTS, BSDS, TNM]])

# 当用户点击“预测”按钮时，执行预测
if st.button("预测"):
    # 进行预测
    prediction_proba = xgb_model.predict_proba(feature_values)[0, 1]
    st.write(f"该患者经“通督调神”针法治疗后PSQI减分率≥50%的概率: {prediction_proba:.2%}")
    
    # 计算 SHAP 值
    shap_values = explainer.shap_values(feature_values)
    
    # 生成 SHAP force plot
    st.write("影响因子的 SHAP 力图：")
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame(feature_values, columns=feature_names), matplotlib=True, show=False)
    
    # 显示力图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
