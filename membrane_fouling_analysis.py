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
    raise RuntimeError(f"Cannot read file: {path}")

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
    mechanism_names = ["Standard fouling (pore constriction)", "Complete fouling (pore blocking)", 
                      "Intermediate fouling (pore orifice bridging)", "Cake fouling (surface deposition)"]
    dominant_idx = np.argmax(eta)
    dominant_mechanism = mechanism_names[dominant_idx]
    dominant_ratio = round(eta[dominant_idx] * 100, 1)
    
    # Adjust cleaning recommendations based on different stages
    if stage == "partial":  # 100%-70% stage
        if dominant_idx == 3:  # Cake fouling dominated
            return f"Dominant fouling: {dominant_mechanism} ({dominant_ratio}%)\
                   \nRecommended strategy: Backwashing (pressure 0.08-0.1 MPa, time 3-5 min)\
                   \nOptimization suggestion: This stage is mainly surface cake, which can be effectively restored by backwashing"
        elif dominant_idx == 0 or dominant_idx == 1:  # Standard/complete fouling (internal fouling)
            return f"Dominant fouling: {dominant_mechanism} ({dominant_ratio}%)\
                   \nRecommended strategy: Citric acid solution soaking (concentration 1-2%, time 10-15 min) + Backwashing\
                   \nOptimization suggestion: Early internal fouling needs timely treatment to avoid pollutants penetrating into membrane pores"
        elif dominant_idx == 2:  # Intermediate fouling
            return f"Dominant fouling: {dominant_mechanism} ({dominant_ratio}%)\
                   \nRecommended strategy: Mild alkaline cleaning (NaOH solution, pH=9-10, time 10 min) + Backwashing\
                   \nOptimization suggestion: Control cleaning intensity to protect membrane structure"
        else:
            return f"Dominant fouling: Multiple mechanisms coexist\
                   \nRecommended strategy: Mild composite cleaning (citric acid first, then weak alkali)\
                   \nOptimization suggestion: Synergistic treatment for multiple pollution types"
    else:  # Full process
        if dominant_idx == 3:  # Cake fouling dominated
            return f"Dominant fouling: {dominant_mechanism} ({dominant_ratio}%)\
                   \nRecommended strategy: Backwashing (pressure 0.1 MPa, time 5 min) + Sodium hypochlorite cleaning (concentration 500 ppm, time 10 min)\
                   \nOptimization suggestion: Appropriately reduce operating pressure to reduce cake layer compaction"
        elif dominant_idx == 0 or dominant_idx == 1:  # Standard/complete fouling (internal fouling)
            return f"Dominant fouling: {dominant_mechanism} ({dominant_ratio}%)\
                   \nRecommended strategy: Citric acid solution soaking (concentration 2%, time 20 min) + Backwashing (pressure 0.15 MPa, time 8 min)\
                   \nOptimization suggestion: Pretreatment to remove small molecule pollutants and reduce internal fouling risk"
        elif dominant_idx == 2:  # Intermediate fouling
            return f"Dominant fouling: {dominant_mechanism} ({dominant_ratio}%)\
                   \nRecommended strategy: Alkaline cleaning (NaOH solution, pH=10, time 15 min) + Backwashing (pressure 0.12 MPa, time 6 min)\
                   \nOptimization suggestion: Control feed flow rate to enhance shear force and break pore orifice bridging"
        else:
            return "Balanced pollution types, recommended composite cleaning: Backwashing + Sodium hypochlorite + Citric acid alternate cleaning"

def load_validation_data(data_type, data_id):
    """Load validation data (unified file naming format)"""
    # 云端路径：当前目录（与代码同层级）
    base_path = "."  
    
    # Updated file naming format: typeDataID.csv
    filename = f"{data_type}data{data_id}.csv"
    file_path = os.path.join(base_path, filename)
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file does not exist: {file_path}\nPlease check if {filename} is uploaded to the root directory of the GitHub repository.")
    
    df, _ = read_csv_robust(file_path)
    df = normalize_cols_to_standard(df)
    
    # 检查必要列是否存在
    if "Time (s)" not in df.columns or "Flux" not in df.columns:
        raise ValueError(f"File {filename} lacks required columns! Need 'Time (s)' and 'Flux' columns.")
    
    t_raw = df["Time (s)"].values.astype(float)
    J_raw = df["Flux"].values.astype(float)
    t_clean, J_clean = clean_series(t_raw, J_raw)
    
    if len(J_clean) == 0:
        raise ValueError(f"No valid flux data in {filename} after cleaning!")
    
    J0 = J_clean[0]
    if J0 <= 0:
         raise ValueError(f"The initial flux value in {filename} is zero or negative ({J0}), please check the data.")
        
    return t_clean, J_clean, J0, filename

# ==============================================================================
# <<-- Analysis Logic and GUI Interface
# ==============================================================================

def analyze_single_file(data_type, data_id):
    """Analyze single file and return result dictionary"""
    try:
        t_clean_sec, J_clean, J0, filename = load_validation_data(data_type, data_id)
        
        # 1. Overall analysis (full process)
        params_full = fit_model(t_clean_sec, J_clean, J0)
        J_pred_full = stage1_model(params_full, t_clean_sec - t_clean_sec[0], J0)
        metrics_full = calculate_metrics(J_clean, J_pred_full)
        eta_full = calculate_mechanism_contribution(params_full, t_clean_sec, J0)
        
        # 2. Find 70% cleaning point
        cleaning_time_sec, cleaning_flux, cleaning_idx = find_cleaning_time(
            t_clean_sec, J_pred_full, J0, 0.7
        )
        
        # 3. Partial analysis (100%-70% process)
        # Intercept data before cleaning point
        t_partial = t_clean_sec[:cleaning_idx+1]
        J_clean_partial = J_clean[:cleaning_idx+1]
        
        # Refit model
        params_partial = fit_model(t_partial, J_clean_partial, J0)
        J_pred_partial = stage1_model(params_partial, t_partial - t_partial[0], J0)
        metrics_partial = calculate_metrics(J_clean_partial, J_pred_partial)
        eta_partial = calculate_mechanism_contribution(params_partial, t_partial, J0)
        
        # 4. Generate cleaning recommendations
        cleaning_strategy_full = recommend_cleaning_strategy(eta_full, "full")
        cleaning_strategy_partial = recommend_cleaning_strategy(eta_partial, "partial")
        
        mechanism_names_short = ["Standard", "Complete", "Intermediate", "Cake"]
        dominant_idx_full = np.argmax(eta_full)
        dominant_idx_partial = np.argmax(eta_partial)
        
        return {
            "success": True,
            "filename": filename,
            "data_type": data_type,
            "data_id": data_id,
            "J0": J0,
            # Overall analysis results
            "metrics_full": metrics_full,
            "eta_full": eta_full,
            "dominant_mechanism_full": mechanism_names_short[dominant_idx_full],
            "dominant_ratio_full": round(eta_full[dominant_idx_full] * 100, 1),
            "cleaning_strategy_full": cleaning_strategy_full,
            # Partial analysis results
            "metrics_partial": metrics_partial,
            "eta_partial": eta_partial,
            "dominant_mechanism_partial": mechanism_names_short[dominant_idx_partial],
            "dominant_ratio_partial": round(eta_partial[dominant_idx_partial] * 100, 1),
            "cleaning_strategy_partial": cleaning_strategy_partial,
            # Cleaning point information
            "cleaning_time": round(cleaning_time_sec, 2),
            "cleaning_flux": cleaning_flux,
            # Data
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
    # 适配云端字体（仅保留DejaVu Sans，云端Linux环境支持）
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    st.set_page_config(page_title="Membrane Fouling Diagnosis Tool", page_icon="💧", layout="wide")
    st.title("💧 Ultrafiltration Membrane Fouling Diagnosis and Cleaning Early Warning Tool")
    
    analysis_mode = st.sidebar.selectbox(
        "Please select analysis mode",
        ("Single File Analysis", "Batch Analysis of All Files")
    )

    all_results = []
    
    if analysis_mode == "Single File Analysis":
        st.header("📊 Single File Analysis")
        col1, col2 = st.columns(2)
        with col1:
            data_type = st.selectbox("Data type", ["BSA", "HA", "SA"])
        with col2:
            # Unified data ID as 1
            data_id = st.selectbox("Data ID", [1])
        
        if st.button("Start Analysis"):
            with st.spinner(f"Analyzing {data_type}data{data_id}.csv ..."):
                result = analyze_single_file(data_type, data_id)
                all_results.append(result)
        
    else:
        st.header("📊 Batch Analysis of All Files")
        st.warning("⚠️ Batch analysis will process all 3 files (BSAdata1.csv, HAdata1.csv, SAdata1.csv).")
        
        if st.button("Start Batch Analysis"):
            files_to_process = [("BSA", 1), ("HA", 1), ("SA", 1)]
            total_files = len(files_to_process)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (data_type, data_id) in enumerate(files_to_process):
                progress = (i + 1) / total_files
                status_text.text(f"Analyzing ({i+1}/{total_files}): {data_type}data{data_id}.csv")
                result = analyze_single_file(data_type, data_id)
                all_results.append(result)
                progress_bar.progress(progress)
                time.sleep(0.1)
            
            progress_bar.empty()
            status_text.text("✅ Batch analysis completed!")

    if all_results:
        st.markdown("---")
        st.header("📈 Analysis Results Summary")
        
        summary_data = []
        for res in all_results:
            if res["success"]:
                cleaning_status = f"{res['cleaning_time']} seconds" if res['cleaning_time'] > 0 else "Immediate"
                summary_data.append({
                    "File Name": res["filename"],
                    "Type": res["data_type"],
                    "Initial Flux (LMH)": f'{res["J0"]:.2f}',
                    "Recommended Cleaning Time": cleaning_status,
                    "Full Process Dominant Fouling": f'{res["dominant_mechanism_full"]} ({res["dominant_ratio_full"]}%)',
                    "100%-70% Stage Dominant Fouling": f'{res["dominant_mechanism_partial"]} ({res["dominant_ratio_partial"]}%)',
                    "Status": "Success"
                })
            else:
                summary_data.append({
                    "File Name": res["filename"],
                    "Type": "N/A",
                    "Initial Flux (LMH)": "N/A",
                    "Recommended Cleaning Time": "N/A",
                    "Full Process Dominant Fouling": "N/A",
                    "100%-70% Stage Dominant Fouling": "N/A",
                    "Status": f'Failed: {res["error"]}'
                })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)

        if summary_data:
            csv = summary_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 Download Summary Results (CSV)",
                data=csv,
                file_name="MembraneFoulingAnalysisSummary.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.subheader("🔍 Detailed Analysis Report")
        
        for res in all_results:
            if res["success"]:
                with st.expander(f"📄 Detailed Report for {res['filename']}", expanded=False):
                    # First row: Basic information and fitting results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Basic Information")
                        st.write(f"Initial Flux: {res['J0']:.2f} LMH")
                        st.write(f"Recommended Cleaning Time: {res['cleaning_time']:.1f} seconds")
                        st.write(f"Flux at Cleaning Point: {res['cleaning_flux']:.2f} LMH (70% of initial)")
                    
                    with col2:
                        st.subheader("Full Process Fitting Results")
                        st.write(f"R²: {res['metrics_full']['R2']:.3f}")
                        st.write(f"NRMSE: {res['metrics_full']['NRMSE']:.3f}")
                        st.write(f"MAPE: {res['metrics_full']['MAPE']:.3f}")
                        st.write("**100%-70% Stage Fitting Results**")
                        st.write(f"NRMSE: {res['metrics_partial']['NRMSE']:.3f}")
                        st.write(f"MAPE: {res['metrics_partial']['MAPE']:.3f}")
                    
                    # Second row: Fouling analysis for both stages
                    st.markdown("---")
                    st.subheader("Fouling Mechanism Analysis Comparison")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Full Process Fouling Mechanism Ratio**")
                        sizes_full = [max(round(e*100, 1), 0.0) for e in res["eta_full"]]
                        labels_full = ["Standard", "Complete", "Intermediate", "Cake"]
                        # Filter out categories with 0 ratio
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
                        st.info(f"**Full Process Cleaning Recommendation**\n\n{res['cleaning_strategy_full']}")
                    
                    with col2:
                        st.write("**100%-70% Stage Fouling Mechanism Ratio**")
                        sizes_partial = [max(round(e*100, 1), 0.0) for e in res["eta_partial"]]
                        labels_partial = ["Standard", "Complete", "Intermediate", "Cake"]
                        # Filter out categories with 0 ratio
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
                        st.info(f"**100%-70% Stage Cleaning Recommendation**\n\n{res['cleaning_strategy_partial']}")
                    
                    # Third row: Fitting curve plot
                    st.markdown("---")
                    st.subheader("Flux Decay Fitting Curve Analysis")
                    fig3, ax3 = plt.subplots(figsize=(10, 6))
                    
                    # Plot full process data
                    ax3.plot(res["t_clean_sec"], res["J_clean"], 'o', ms=3, label='Actual Observations', color='gray', alpha=0.6)
                    ax3.plot(res["t_clean_sec"], res["J_pred_full"], '-', lw=2, label='Full Process Fitting Curve', color='orange', alpha=0.8)
                    
                    # Plot 100%-70% stage data (bold line)
                    ax3.plot(res["t_partial"], res["J_pred_partial"], '-', lw=3, label='100%-70% Stage Fitting Curve', color='green')
                    
                    # Plot 70% flux threshold line
                    acceptable_flux = res["J0"] * 0.7
                    ax3.axhline(y=acceptable_flux, color='red', linestyle=':', label='70% Initial Flux (Recommended Cleaning Point)')
                    
                    # Plot recommended cleaning point and connecting lines
                    cleaning_time = res["cleaning_time"]
                    ax3.scatter(cleaning_time, acceptable_flux, color='red', s=80, zorder=5, label='Recommended Cleaning Point')
                    ax3.axvline(x=cleaning_time, color='red', linestyle='--', alpha=0.7)
                    ax3.text(cleaning_time, 0, f'Recommended Cleaning Time: {cleaning_time:.1f}s', 
                            horizontalalignment='center', verticalalignment='bottom', 
                            color='red', fontsize=10, fontweight='bold')
                    
                    # Highlight 100%-70% stage area
                    ax3.axvspan(0, cleaning_time, alpha=0.1, color='green', label='100%-70% Recommended Operation Interval')
                    
                    ax3.set_xlabel('Time (seconds)')
                    ax3.set_ylabel('Flux (LMH)')
                    ax3.legend(loc='best')
                    ax3.grid(alpha=0.3)
                    # Set y-axis range
                    y_min = min(res["J_clean"].min(), acceptable_flux) * 0.8
                    y_max = res["J0"] * 1.1
                    ax3.set_ylim(y_min, y_max)
                    st.pyplot(fig3)
                    
            else:
                with st.expander(f"❌ Analysis Failed for {res['filename']}", expanded=False):
                    st.error(res["error"])

if __name__ == "__main__":
    main()