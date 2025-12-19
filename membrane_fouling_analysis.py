# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import r2_score
import time

# ==============================================================================
# <<-- Core Model Module
# ==============================================================================
USE_LOG_FIT = True
USE_HUBER = True
HUBER_DELTA = 0.002
GA_POP = 60
GA_GEN = 100
GA_MUT = 0.10
GA_ELITE = 2
RANDOM_SEED = 42
BOUNDS = [(0,1)]*4 + [(0.05,1)]*2  # Ks,Kc,Kb,Ki,a,b
EPS = 1e-12
EXP_FLOOR = -50.0

def stage1_model(params, t, J0):
    """Four-mechanism unified model"""
    Ks, Kc, Kb, Ki, a, b = params
    c1 = 10.0 * Ks * J0 / 2.0
    c2 = 10.0 * Kb
    c3 = 10.0 * Ki * J0
    c4 = 20.0 * Kc * J0**2

    base1 = np.maximum(1.0 + c1 * t, EPS)
    base3 = np.maximum(1.0 + c3 * t, EPS)
    base4 = np.maximum(1.0 + c4 * t, EPS)

    term1 = base1 ** (-2.0 * a)
    expo = np.maximum(-b * c2 * t, EXP_FLOOR)
    term2 = np.exp(expo)
    term3 = base3 ** (-(1.0 - b))
    term4 = base4 ** (-(1.0 - a) / 2.0)

    J_pred = J0 * term1 * term2 * term3 * term4
    return np.maximum(J_pred, EPS)

def huber_loss(residual, delta):
    """Huber loss function"""
    abs_r = np.abs(residual)
    quad = 0.5 * (abs_r ** 2)
    lin = delta * (abs_r - 0.5 * delta)
    return np.where(abs_r <= delta, quad, lin)

def objective(params, t, J_obs, J0):
    """Objective function"""
    J_pred = stage1_model(params, t, J0)
    mask = np.isfinite(J_obs) & np.isfinite(J_pred)
    if mask.sum() < 5:
        return 1e9
    y = J_obs[mask]; yhat = J_pred[mask]
    if USE_LOG_FIT:
        y = np.maximum(y, EPS)
        yhat = np.maximum(yhat, EPS)
        r = np.log(y) - np.log(yhat)
    else:
        r = y - yhat
    return np.mean(huber_loss(r, HUBER_DELTA)) if USE_HUBER else np.mean(r**2)

def genetic_algorithm(objective_fn, bounds, t, J_obs, J0):
    """Genetic algorithm optimization"""
    rng = np.random.default_rng(RANDOM_SEED)
    dim = len(bounds)
    pop = rng.random((GA_POP, dim))
    for i in range(dim):
        lo, hi = bounds[i]
        pop[:, i] = lo + pop[:, i] * (hi - lo)

    def fitness(ind):
        try:
            val = float(objective_fn(ind, t, J_obs, J0))
            return val if np.isfinite(val) else 1e9
        except Exception:
            return 1e9

    for _ in range(GA_GEN):
        scores = np.array([fitness(ind) for ind in pop])
        elite_idx = np.argsort(scores)[:GA_ELITE]
        new_pop = pop[elite_idx].copy()
        while len(new_pop) < GA_POP:
            idx1 = rng.integers(0, len(pop), size=3)
            p1 = pop[idx1[np.argmin(scores[idx1])]].copy()
            idx2 = rng.integers(0, len(pop), size=3)
            p2 = pop[idx2[np.argmin(scores[idx2])]].copy()
            cp = rng.integers(1, dim)
            child = np.concatenate([p1[:cp], p2[cp:]])
            for i in range(dim):
                if rng.random() < GA_MUT:
                    lo, hi = bounds[i]
                    child[i] += rng.normal(0, 0.1 * (hi - lo))
                    child[i] = np.clip(child[i], lo, hi)
            new_pop = np.vstack([new_pop, child])
        pop = new_pop

    scores = np.array([fitness(ind) for ind in pop])
    best = pop[np.argmin(scores)]
    return best

def fit_model(t, J_obs, J0):
    """Model fitting main function"""
    if len(t) < 5:
        return np.array([0.1,0.1,0.1,0.1,0.5,0.5])
    return genetic_algorithm(objective, BOUNDS, t, J_obs, J0)

def calculate_mechanism_contribution(params, t, J0):
    """Calculate contribution ratio of four fouling mechanisms"""
    Ks, Kc, Kb, Ki, a, b = params
    c1 = 10.0 * Ks * J0 / 2.0
    c2 = 10.0 * Kb
    c3 = 10.0 * Ki * J0
    c4 = 20.0 * Kc * J0**2

    s1 = - (2.0 * a) * c1 / (1.0 + c1 * t + EPS)
    s2 = - b * c2 * np.ones_like(t)
    s3 = - (1.0 - b) * c3 / (1.0 + c3 * t + EPS)
    s4 = - (1.0 - a) * c4 / (2.0 * (1.0 + c4 * t + EPS))
    
    Di = []
    for si in [s1, s2, s3, s4]:
        val = -np.trapz(si, t)
        Di.append(max(val, 0.0))
    Dsum = sum(Di) + EPS
    eta = np.array([d / Dsum for d in Di])
    return eta  # [Standard fouling, Complete fouling, Intermediate fouling, Cake fouling]

# ==============================================================================
# <<-- Utility Functions Module
# ==============================================================================

def read_csv_robust(path):
    """Robust CSV file reading (supports multiple encodings)"""
    for enc in ("utf-8-sig", "utf-8", "gbk", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return df, enc
        except Exception:
            continue
    raise RuntimeError(f"无法读取文件: {path}")

def normalize_cols_to_standard(df):
    """Standardize column names (unify to "Time (s)" and "Flux")"""
    def norm_key(c):
        c = str(c).replace("\ufeff","").strip().replace("（","(").replace("）",")").replace(" ", "").lower()
        return c
    new_names = {}
    for c in df.columns:
        k = norm_key(c)
        if k in {"时间s","时间(s)","times","time(s)","time","t","时间"}:
            new_names[c] = "Time (s)"
        elif k in {"实际通量","通量","flux","j"}:
            new_names[c] = "Flux"
    return df.rename(columns=new_names)

def clean_series(t, J):
    """Data cleaning (remove invalid values and tail outliers)"""
    mask = np.isfinite(t) & np.isfinite(J)
    t = t[mask]; J = J[mask]
    if len(t) > 0:
        k = max(int(round(len(t) * 0.99)), 5)
        t = t[:k]; J = J[:k]
    return t, J

def calculate_metrics(J_obs, J_pred):
    """Calculate fitting metrics (R², NRMSE, MAPE)"""
    mask = np.isfinite(J_obs) & np.isfinite(J_pred)
    if mask.sum() == 0:
        return {"R2": np.nan, "NRMSE": np.nan, "MAPE": np.nan}
    y = J_obs[mask]; yhat = J_pred[mask]
    r2 = r2_score(y, yhat)
    rmse = np.sqrt(np.mean((y - yhat)**2))
    nrmse = rmse / (np.max(y) - np.min(y) + 1e-12) if (np.max(y) - np.min(y)) > 0 else np.nan
    mape_floor = max(1e-8, 0.05 * np.median(np.abs(y)))
    denom = np.maximum(np.abs(y), mape_floor)
    mape = np.mean(np.abs(y - yhat) / denom)
    return {
        "R2": round(r2, 3),
        "NRMSE": round(nrmse, 3) if np.isfinite(nrmse) else np.nan,
        "MAPE": round(mape, 3)
    }

def find_cleaning_time(t, J_pred, J0, acceptable_ratio=0.7):
    """
    Find the time point when flux drops to 70% of initial value (based on fitting curve)
    """
    acceptable_flux = J0 * acceptable_ratio
    # Find indices where fitting curve first drops below acceptable flux
    below_threshold_idx = np.where(J_pred <= acceptable_flux)[0]
    
    if len(below_threshold_idx) > 0:
        # Take the first point below threshold
        first_idx = below_threshold_idx[0]
        if first_idx == 0:
            # Initial point is already below threshold, return 0
            return 0.0, acceptable_flux, first_idx
        # Linear interpolation for more accurate time point
        t1, t2 = t[first_idx-1], t[first_idx]
        j1, j2 = J_pred[first_idx-1], J_pred[first_idx]
        # Interpolation formula: t = t1 + (t2-t1)*(acceptable_flux - j1)/(j2 - j1)
        cleaning_time = t1 + (t2 - t1) * (acceptable_flux - j1) / (j2 - j1)
        return cleaning_time, acceptable_flux, first_idx
    else:
        # Entire fitting curve is above threshold, return last point's time and flux
        return t[-1], J_pred[-1], len(t)-1

def recommend_cleaning_strategy(eta, stage="full"):
    """Recommend cleaning strategy based on dominant fouling mechanism"""
    # 污染机制名称改为中文
    mechanism_names = ["标准污染（孔道收缩）", "完全污染（孔道堵塞）", 
                      "中间污染（孔口桥接）", "滤饼污染（表面沉积）"]
    dominant_idx = np.argmax(eta)
    dominant_mechanism = mechanism_names[dominant_idx]
    dominant_ratio = round(eta[dominant_idx] * 100, 1)
    
    # Adjust cleaning recommendations based on different stages
    if stage == "partial":  # 100%-70%阶段
        if dominant_idx == 3:  # 滤饼污染主导
            return f"主导污染类型：{dominant_mechanism}（占比{dominant_ratio}%）\
                   \n推荐清洗策略：反洗（压力0.08-0.1 MPa，时长3-5分钟）\
                   \n优化建议：该阶段以表面滤饼层为主，反洗可有效恢复通量"
        elif dominant_idx == 0 or dominant_idx == 1:  # 标准/完全污染（内部污染）
            return f"主导污染类型：{dominant_mechanism}（占比{dominant_ratio}%）\
                   \n推荐清洗策略：柠檬酸溶液浸泡（浓度1-2%，时长10-15分钟）+ 反洗\
                   \n优化建议：早期内部污染需及时处理，避免污染物渗入膜孔内部"
        elif dominant_idx == 2:  # 中间污染
            return f"主导污染类型：{dominant_mechanism}（占比{dominant_ratio}%）\
                   \n推荐清洗策略：弱碱性清洗（NaOH溶液，pH=9-10，时长10分钟）+ 反洗\
                   \n优化建议：控制清洗强度，保护膜结构完整性"
        else:
            return f"主导污染类型：多种机制共存\
                   \n推荐清洗策略：温和复合清洗（先柠檬酸后弱碱）\
                   \n优化建议：针对多种污染类型进行协同处理"
    else:  # 全流程
        if dominant_idx == 3:  # 滤饼污染主导
            return f"主导污染类型：{dominant_mechanism}（占比{dominant_ratio}%）\
                   \n推荐清洗策略：反洗（压力0.1 MPa，时长5分钟）+ 次氯酸钠清洗（浓度500 ppm，时长10分钟）\
                   \n优化建议：适当降低运行压力，减少滤饼层压实"
        elif dominant_idx == 0 or dominant_idx == 1:  # 标准/完全污染（内部污染）
            return f"主导污染类型：{dominant_mechanism}（占比{dominant_ratio}%）\
                   \n推荐清洗策略：柠檬酸溶液浸泡（浓度2%，时长20分钟）+ 反洗（压力0.15 MPa，时长8分钟）\
                   \n优化建议：预处理去除小分子污染物，降低内部污染风险"
        elif dominant_idx == 2:  # 中间污染
            return f"主导污染类型：{dominant_mechanism}（占比{dominant_ratio}%）\
                   \n推荐清洗策略：碱性清洗（NaOH溶液，pH=10，时长15分钟）+ 反洗（压力0.12 MPa，时长6分钟）\
                   \n优化建议：控制进料流速，增强剪切力破除孔口桥接"
        else:
            return "污染类型均衡，推荐复合清洗：反洗 + 次氯酸钠 + 柠檬酸交替清洗"

def load_validation_data(data_type, data_id):
    """Load validation data (unified file naming format)"""
    # 云端路径：当前目录（与代码同层级）
    base_path = "."  
    
    # Updated file naming format: typeDataID.csv
    filename = f"{data_type}data{data_id}.csv"
    file_path = os.path.join(base_path, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在：{file_path}\n请检查{filename}是否已上传至GitHub仓库根目录。")
    
    df, _ = read_csv_robust(file_path)
    df = normalize_cols_to_standard(df)
    
    # 检查必要列是否存在
    if "Time (s)" not in df.columns or "Flux" not in df.columns:
        raise ValueError(f"文件{filename}缺少必要列！需要包含'Time (s)'和'Flux'列。")
    
    t_raw = df["Time (s)"].values.astype(float)
    J_raw = df["Flux"].values.astype(float)
    t_clean, J_clean = clean_series(t_raw, J_raw)
    
    if len(J_clean) == 0:
        raise ValueError(f"{filename}清洗后无有效通量数据！")
    
    J0 = J_clean[0]
    if J0 <= 0:
         raise ValueError(f"{filename}中的初始通量值为零或负数（{J0}），请检查数据。")
        
    return t_clean, J_clean, J0, filename

# ==============================================================================
# <<-- Analysis Logic and GUI Interface
# ==============================================================================

def analyze_single_file(data_type, data_id):
    """Analyze single file and return result dictionary"""
    try:
        t_clean_sec, J_clean, J0, filename = load_validation_data(data_type, data_id)
        
        # 1. 全流程分析
        params_full = fit_model(t_clean_sec, J_clean, J0)
        J_pred_full = stage1_model(params_full, t_clean_sec - t_clean_sec[0], J0)
        metrics_full = calculate_metrics(J_clean, J_pred_full)
        eta_full = calculate_mechanism_contribution(params_full, t_clean_sec, J0)
        
        # 2. 查找70%通量清洗点
        cleaning_time_sec, cleaning_flux, cleaning_idx = find_cleaning_time(
            t_clean_sec, J_pred_full, J0, 0.7
        )
        
        # 3. 100%-70%阶段分析
        # 截取清洗点前的数据
        t_partial = t_clean_sec[:cleaning_idx+1]
        J_clean_partial = J_clean[:cleaning_idx+1]
        
        # 重新拟合模型
        params_partial = fit_model(t_partial, J_clean_partial, J0)
        J_pred_partial = stage1_model(params_partial, t_partial - t_partial[0], J0)
        metrics_partial = calculate_metrics(J_clean_partial, J_pred_partial)
        eta_partial = calculate_mechanism_contribution(params_partial, t_partial, J0)
        
        # 4. 生成清洗建议
        cleaning_strategy_full = recommend_cleaning_strategy(eta_full, "full")
        cleaning_strategy_partial = recommend_cleaning_strategy(eta_partial, "partial")
        
        # 污染机制简称（中文）
        mechanism_names_short = ["标准污染", "完全污染", "中间污染", "滤饼污染"]
        dominant_idx_full = np.argmax(eta_full)
        dominant_idx_partial = np.argmax(eta_partial)
        
        return {
            "success": True,
            "filename": filename,
            "data_type": data_type,
            "data_id": data_id,
            "J0": J0,
            # 全流程分析结果
            "metrics_full": metrics_full,
            "eta_full": eta_full,
            "dominant_mechanism_full": mechanism_names_short[dominant_idx_full],
            "dominant_ratio_full": round(eta_full[dominant_idx_full] * 100, 1),
            "cleaning_strategy_full": cleaning_strategy_full,
            # 100%-70%阶段分析结果
            "metrics_partial": metrics_partial,
            "eta_partial": eta_partial,
            "dominant_mechanism_partial": mechanism_names_short[dominant_idx_partial],
            "dominant_ratio_partial": round(eta_partial[dominant_idx_partial] * 100, 1),
            "cleaning_strategy_partial": cleaning_strategy_partial,
            # 清洗点信息
            "cleaning_time": round(cleaning_time_sec, 2),
            "cleaning_flux": cleaning_flux,
            # 数据
            "t_clean_sec": t_clean_sec,
            "J_clean": J_clean,
            "J_pred_full": J_pred_full,
            "t_partial": t_partial,
            "J_pred_partial": J_pred_partial,
            "error": None
        }
    except Exception as e:
        filename = f"{data_type}data{data_id}.csv"
        return {
            "success": False,
            "filename": filename,
            "error": str(e)
        }

def main():
    # 适配云端字体（支持中文显示）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 增加黑体支持中文
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 页面配置（中文标题）
    st.set_page_config(page_title="水处理膜污染诊断与清洗预警系统", page_icon="💧", layout="wide")
    st.title("💧 水处理膜污染诊断与清洗预警系统")
    
    # 侧边栏分析模式选择（中文）
    analysis_mode = st.sidebar.selectbox(
        "请选择分析模式",
        ("单文件分析", "全部文件批量分析")
    )

    all_results = []
    
    if analysis_mode == "单文件分析":
        st.header("📊 单文件分析")
        col1, col2 = st.columns(2)
        with col1:
            # 污染物类型下拉框（中文）
            data_type = st.selectbox("污染物类型", ["BSA", "HA", "SA"])
        with col2:
            # 数据ID下拉框（中文）
            data_id = st.selectbox("数据ID", [1])
        
        # 分析按钮（中文）
        if st.button("开始分析"):
            with st.spinner(f"正在分析 {data_type}data{data_id}.csv ..."):
                result = analyze_single_file(data_type, data_id)
                all_results.append(result)
        
    else:
        st.header("📊 全部文件批量分析")
        st.warning("⚠️ 批量分析将处理全部3个文件（BSAdata1.csv、HAdata1.csv、SAdata1.csv）。")
        
        # 批量分析按钮（中文）
        if st.button("开始批量分析"):
            files_to_process = [("BSA", 1), ("HA", 1), ("SA", 1)]
            total_files = len(files_to_process)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (data_type, data_id) in enumerate(files_to_process):
                progress = (i + 1) / total_files
                status_text.text(f"正在分析（{i+1}/{total_files}）：{data_type}data{data_id}.csv")
                result = analyze_single_file(data_type, data_id)
                all_results.append(result)
                progress_bar.progress(progress)
                time.sleep(0.1)
            
            progress_bar.empty()
            status_text.text("✅ 批量分析完成！")

    if all_results:
        st.markdown("---")
        st.header("📈 分析结果汇总")
        
        summary_data = []
        for res in all_results:
            if res["success"]:
                cleaning_status = f"{res['cleaning_time']} 秒" if res['cleaning_time'] > 0 else "立即清洗"
                summary_data.append({
                    "文件名": res["filename"],
                    "类型": res["data_type"],
                    "初始通量 (LMS)": f'{res["J0"]:.2f}',
                    "推荐清洗时间": cleaning_status,
                    "全流程主导污染类型": f'{res["dominant_mechanism_full"]}（{res["dominant_ratio_full"]}%）',
                    "100%-70%阶段主导污染类型": f'{res["dominant_mechanism_partial"]}（{res["dominant_ratio_partial"]}%）',
                    "状态": "成功"
                })
            else:
                summary_data.append({
                    "文件名": res["filename"],
                    "类型": "无",
                    "初始通量 (LMS)": "无",
                    "推荐清洗时间": "无",
                    "全流程主导污染类型": "无",
                    "100%-70%阶段主导污染类型": "无",
                    "状态": f'失败：{res["error"]}'
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        if summary_data:
            csv = summary_df.to_csv(index=False, encoding='utf-8-sig')
            # 下载按钮（中文）
            st.download_button(
                label="💾 下载汇总结果（CSV）",
                data=csv,
                file_name="膜污染分析汇总结果.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.subheader("🔍 详细分析报告")
        
        for res in all_results:
            if res["success"]:
                with st.expander(f"📄 {res['filename']} 详细报告", expanded=False):
                    # 第一行：基础信息和拟合结果
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("基础信息")
                        st.write(f"初始通量：{res['J0']:.2f} LMS")
                        st.write(f"推荐清洗时间：{res['cleaning_time']:.1f} 秒")
                        st.write(f"清洗点通量：{res['cleaning_flux']:.2f} LMS（初始通量的70%）")
                    
                    with col2:
                        st.subheader("全流程拟合结果")
                        st.write(f"决定系数R²：{res['metrics_full']['R2']:.3f}")
                        st.write(f"归一化均方根误差NRMSE：{res['metrics_full']['NRMSE']:.3f}")
                        st.write(f"平均绝对百分比误差MAPE：{res['metrics_full']['MAPE']:.3f}")
                        st.write("**100%-70%阶段拟合结果**")
                        st.write(f"归一化均方根误差NRMSE：{res['metrics_partial']['NRMSE']:.3f}")
                        st.write(f"平均绝对百分比误差MAPE：{res['metrics_partial']['MAPE']:.3f}")
                    
                    # 第二行：污染机制分析对比
                    st.markdown("---")
                    st.subheader("污染机制分析对比")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**全流程污染机制占比**")
                        sizes_full = [max(round(e*100, 1), 0.0) for e in res["eta_full"]]
                        labels_full = ["标准污染", "完全污染", "中间污染", "滤饼污染"]
                        # 过滤掉占比为0的类别
                        filtered_sizes_full = []
                        filtered_labels_full = []
                        for s, l in zip(sizes_full, labels_full):
                            if s > 0:
                                filtered_sizes_full.append(s)
                                filtered_labels_full.append(l)
                        fig1, ax1 = plt.subplots(figsize=(5, 4))
                        ax1.pie(filtered_sizes_full, labels=filtered_labels_full, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
                        ax1.axis('equal')
                        st.pyplot(fig1)
                        st.info(f"**全流程清洗建议**\n\n{res['cleaning_strategy_full']}")
                    
                    with col2:
                        st.write("**100%-70%阶段污染机制占比**")
                        sizes_partial = [max(round(e*100, 1), 0.0) for e in res["eta_partial"]]
                        labels_partial = ["标准污染", "完全污染", "中间污染", "滤饼污染"]
                        # 过滤掉占比为0的类别
                        filtered_sizes_partial = []
                        filtered_labels_partial = []
                        for s, l in zip(sizes_partial, labels_partial):
                            if s > 0:
                                filtered_sizes_partial.append(s)
                                filtered_labels_partial.append(l)
                        fig2, ax2 = plt.subplots(figsize=(5, 4))
                        ax2.pie(filtered_sizes_partial, labels=filtered_labels_partial, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
                        ax2.axis('equal')
                        st.pyplot(fig2)
                        st.info(f"**100%-70%阶段清洗建议**\n\n{res['cleaning_strategy_partial']}")
                    
                    # 第三行：通量衰减拟合曲线分析
                    st.markdown("---")
                    st.subheader("通量衰减拟合曲线分析")
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    
                    # 绘制全流程数据
                    ax3.plot(res["t_clean_sec"], res["J_clean"], 'o', ms=3, label='实际观测值', color='gray', alpha=0.6)
                    ax3.plot(res["t_clean_sec"], res["J_pred_full"], '-', lw=2, label='全流程拟合曲线', color='orange', alpha=0.8)
                    
                    # 绘制100%-70%阶段数据（加粗）
                    ax3.plot(res["t_partial"], res["J_pred_partial"], '-', lw=3, label='100%-70%阶段拟合曲线', color='green')
                    
                    # 绘制70%通量阈值线
                    acceptable_flux = res["J0"] * 0.7
                    ax3.axhline(y=acceptable_flux, color='red', linestyle=':', label='70%初始通量（推荐清洗点）')
                    
                    # 绘制推荐清洗点和连接线
                    cleaning_time = res["cleaning_time"]
                    ax3.scatter(cleaning_time, acceptable_flux, color='red', s=80, zorder=5, label='推荐清洗时间点')
                    ax3.axvline(x=cleaning_time, color='red', linestyle='--', alpha=0.7)
                    ax3.text(cleaning_time, 0, f'推荐清洗时间：{cleaning_time:.1f}秒', 
                            horizontalalignment='center', verticalalignment='bottom', 
                            color='red', fontsize=10, fontweight='bold')
                    
                    # 高亮100%-70%阶段区域
                    ax3.axvspan(0, cleaning_time, alpha=0.1, color='green', label='100%-70%推荐运行区间')
                    
                    ax3.set_xlabel('时间（秒）')
                    ax3.set_ylabel('通量（LMS）')
                    ax3.legend(loc='best')
                    ax3.grid(alpha=0.3)
                    # 设置y轴范围
                    y_min = min(res["J_clean"].min(), acceptable_flux) * 0.8
                    y_max = res["J0"] * 1.1
                    ax3.set_ylim(y_min, y_max)
                    st.pyplot(fig3)
                    
            else:
                with st.expander(f"❌ {res['filename']} 分析失败", expanded=False):
                    st.error(res["error"])

if __name__ == "__main__":
    main()