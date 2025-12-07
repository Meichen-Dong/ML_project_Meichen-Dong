import streamlit as st
import joblib
import pandas as pd
import numpy as np
import math
import sys  # 用于在加载模型失败时退出程序

# --- 1. Streamlit 配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(
    page_title="💎 钻石价格预测应用 (Random Forest)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. 硬编码分类特征映射 ---
# 这些映射必须与训练模型时使用的编码一致！
CUT_MAPPING = {'Ideal': 3, 'Premium': 4, 'Good': 2, 'Very Good': 5, 'Fair': 1}
COLOR_MAPPING = {'E': 2, 'I': 6, 'J': 7, 'H': 5, 'F': 3, 'G': 4, 'D': 1}
CLARITY_MAPPING = {'SI2': 4, 'SI1': 3, 'VS1': 5, 'VS2': 6, 'VVS2': 7, 'VVS1': 8, 'I1': 1, 'IF': 2}


# --- 3. 模型加载函数 (使用 Streamlit 缓存) ---
@st.cache_resource
def load_model(path):
    """加载已保存的 Random Forest 模型。"""
    return joblib.load(path)

# --- 4. 安全地加载模型 ---
MODEL_PATH = 'random_forest_model.joblib'
try:
    model_rf = load_model(MODEL_PATH)
except FileNotFoundError:
    # ❌ 错误处理：模型文件不存在。
    # 这里我们使用 st.error() 提示用户，因为 st.set_page_config() 已经调用。
    st.error(f"严重错误：找不到模型文件 '{MODEL_PATH}'。请确保文件在相同目录下。")
    st.stop()
except Exception as e:
    # ❌ 错误处理：加载模型时发生其他错误。
    st.error(f"加载模型时发生错误: {e}")
    st.stop()


# --- 5. Streamlit 界面和用户输入 ---

st.title("💎 钻石价格预测应用")
st.markdown("### 使用优化后的 Random Forest 模型预测")

st.sidebar.header("输入钻石特征")

# 用户通过侧边栏输入特征
carat = st.sidebar.slider("克拉 (Carat)", min_value=0.2, max_value=5.01, value=0.7, step=0.01)
depth = st.sidebar.slider("深度百分比 (Depth %)", min_value=43.0, max_value=79.0, value=61.8, step=0.1)
table = st.sidebar.slider("桌面宽度百分比 (Table %)", min_value=43.0, max_value=95.0, value=57.0, step=0.1)
x = st.sidebar.slider("长度 (X) mm", min_value=0.0, max_value=10.74, value=5.7, step=0.01)
y = st.sidebar.slider("宽度 (Y) mm", min_value=0.0, max_value=58.9, value=5.7, step=0.01)
z = st.sidebar.slider("高度 (Z) mm", min_value=0.0, max_value=31.8, value=3.5, step=0.01)

cut_str = st.sidebar.selectbox("切工 (Cut)", options=list(CUT_MAPPING.keys()), index=0)
color_str = st.sidebar.selectbox("颜色 (Color)", options=list(COLOR_MAPPING.keys()), index=1)
clarity_str = st.sidebar.selectbox("净度 (Clarity)", options=list(CLARITY_MAPPING.keys()), index=3)


# --- 6. 特征工程和数据准备 ---

# 分类特征编码
cut = CUT_MAPPING[cut_str]
color = COLOR_MAPPING[color_str]
clarity = CLARITY_MAPPING[clarity_str]

# 计算新特征 (与Jupyter Notebook中的步骤一致)
volume = x * y * z
density = carat / volume if volume != 0 else 0.0
xy_ratio = x / y if y != 0 else 0.0

# 准备预测数据 DataFrame - 确保列顺序与训练时一致！
input_data = pd.DataFrame({
    'carat': [carat],
    'cut': [cut],
    'color': [color],
    'clarity': [clarity],
    'depth': [depth],
    'table': [table],
    'x': [x],
    'y': [y],
    'z': [z],
    'volume': [volume],
    'density': [density],
    'xy_ratio': [xy_ratio]
})

# --- 7. 预测与结果展示 ---

st.subheader("您输入的钻石特征和派生特征")
st.dataframe(input_data)
st.markdown("---")

if st.button("🚀 预测钻石价格"):
    
    # 预测对数价格 (Log Price)
    log_price_pred = model_rf.predict(input_data)[0]
    
    # 转换回实际价格 (Price)
    price_pred = np.exp(log_price_pred)

    st.subheader("✨ 预测结果")
    st.success(f"模型预测的钻石价格（美元）为：")
    st.balloons()
    st.write(f"## **${price_pred:,.2f}**")
    st.caption(f"---")
    st.info(f"注意：模型预测的是对数价格 ($\log(\text{{Price}})$: ${log_price_pred:.4f}$)，然后转换回实际价格。")