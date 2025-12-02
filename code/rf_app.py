import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor # 仅用于加载类型检查

# --- 配置 ---
MODEL_PATH = 'random_forest_model.joblib'

# 重要的：定义分类特征的所有可能值
# 这些值必须与您训练模型时数据集中出现的所有唯一值一致！
CUT_OPTIONS = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']
COLOR_OPTIONS = ['D', 'E', 'F', 'G', 'H', 'I', 'J'] 
CLARITY_OPTIONS = ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'] 
CATEGORICAL_FEATURES = ['cut', 'color', 'clarity']
NUMERICAL_FEATURES = ['carat', 'depth', 'table']

# --- 模型加载 ---
# 使用 st.cache_resource 确保模型只加载一次
@st.cache_resource
def load_rf_model():
    """加载已训练的 Random Forest 模型"""
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except FileNotFoundError:
        st.error(f"错误：未找到模型文件 '{MODEL_PATH}'。请确保文件已保存到当前目录。")
        return None

rf_model = load_rf_model()

# --- Streamlit 界面 ---
st.set_page_config(page_title="钻石价格预测", layout="wide")
st.title("🌲 Random Forest 钻石价格预测器")
st.markdown("请输入钻石的各项参数，模型将预测其对数价格。")

# --- 输入侧边栏 ---
with st.sidebar:
    st.header("钻石参数输入")

    # 数值特征
    carat = st.slider("克拉 (Carat)", min_value=0.2, max_value=5.01, value=1.0, step=0.01)
    depth = st.slider("深度百分比 (Depth %)", min_value=43.0, max_value=79.0, value=61.8, step=0.1)
    table = st.slider("桌面宽度百分比 (Table %)", min_value=43.0, max_value=95.0, value=57.0, step=1.0)

    # 分类特征
    cut = st.selectbox("切工 (Cut)", options=CUT_OPTIONS, index=CUT_OPTIONS.index('Ideal'))
    color = st.selectbox("颜色 (Color)", options=COLOR_OPTIONS, index=COLOR_OPTIONS.index('G'))
    clarity = st.selectbox("净度 (Clarity)", options=CLARITY_OPTIONS, index=CLARITY_OPTIONS.index('VS2'))

# --- 预测逻辑 ---

def preprocess_input(input_df):
    """
    对输入数据进行与训练集相同的预处理 (独热编码)
    **注意：这要求处理后的特征列名和顺序必须与训练模型时的X特征完全一致！**
    """
    
    # 1. 对分类特征进行独热编码
    df_dummies = pd.get_dummies(input_df, columns=CATEGORICAL_FEATURES, drop_first=False)
    
    # 2. 确保所有可能的哑变量列都存在 (即使当前输入中没有)
    # 这一步非常关键，以保证特征数量一致
    all_dummy_cols = [
        *[f'cut_{c}' for c in CUT_OPTIONS], 
        *[f'color_{c}' for c in COLOR_OPTIONS], 
        *[f'clarity_{c}' for c in CLARITY_OPTIONS]
    ]
    
    # 3. 填充缺失的列 (如果用户没有选某个类别，则该列值为0)
    for col in all_dummy_cols:
        if col not in df_dummies.columns:
            df_dummies[col] = 0
            
    # 4. 确保最终特征的顺序与训练模型时的顺序一致
    # 假设训练特征是 [数值特征] + [所有哑变量特征]
    final_cols = NUMERICAL_FEATURES + sorted(all_dummy_cols) 
    
    return df_dummies[final_cols]


if st.button("开始预测价格"):
    if rf_model is not None:
        try:
            # 1. 创建原始输入 DataFrame
            input_raw = pd.DataFrame({
                'carat': [carat], 'depth': [depth], 'table': [table],
                'cut': [cut], 'color': [color], 'clarity': [clarity]
            })

            # 2. 预处理数据
            input_processed = preprocess_input(input_raw)
            
            # 3. 进行对数价格预测
            log_price_prediction = rf_model.predict(input_processed)[0]
            
            # 4. 将对数价格转换回原始价格 (e^y)
            final_price = np.exp(log_price_prediction)

            # --- 结果展示 ---
            st.success("✅ 预测完成！")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="预测对数价格 (Log Price)", 
                    value=f"{log_price_prediction:.4f}"
                )
            with col2:
                 st.metric(
                    label="**预测最终价格 ($)**", 
                    value=f"${final_price:,.2f}"
                )

            st.balloons()
            
        except Exception as e:
            st.error(f"预测失败，请检查模型和预处理步骤是否匹配训练过程。错误详情: {e}")