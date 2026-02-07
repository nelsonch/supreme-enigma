import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

from mmm_class import MMM
from util import (generate_exogenous_data, compute_true_roi, 
                  create_model_specs)

# Custom data generation for Weibull C-shape
def generate_sales_weibull_c(spend, data, params, seed=None):
    """Generate sales using Weibull C-shape saturation + adstock"""
    if seed is not None:
        np.random.seed(seed)
    
    C, L, T = spend.shape
    log_sales = np.zeros((L, T))
    
    beta_media_loc = params['beta_media_loc']
    alpha_true = params['alpha_true']
    K_true = params['K_true']
    S_true = params['S_true']
    
    def adstock_transform(x, alpha):
        T = len(x)
        result = np.zeros(T)
        for t in range(T):
            if t == 0:
                result[t] = x[t]
            else:
                result[t] = x[t] + alpha * result[t-1]
        return result
    
    def weibull_c_saturation(z, K, S):
        """Weibull C-shape (0 < S < 1)"""
        return 1 - np.exp(-(z/K)**S)
    
    for l in range(L):
        for t in range(T):
            log_y = params['beta_0'] + params['beta_location'][l]
            
            for c in range(C):
                spend_adstock = adstock_transform(spend[c, l, :t+1], alpha_true[c])[-1]
                spend_sat = weibull_c_saturation(spend_adstock, K_true[c], S_true[c])
                log_y += beta_media_loc[c, l] * spend_sat
            
            log_y += params['beta_price'] * np.log(data['price'][l, t])
            log_y += params['beta_discount'] * data['discount'][l, t]
            log_y += params['beta_unemployment'] * data['unemployment'][t]
            log_y += params['beta_confidence'] * data['confidence'][t]
            log_y += params['beta_trend'] * t
            log_y += params['beta_sin'] * np.sin(2*np.pi*t/52)
            log_y += params['beta_cos'] * np.cos(2*np.pi*t/52)
            log_y += np.sum(params['gamma_holiday'] * data['holidays'][:, t])
            log_y += np.random.normal(0, params['sigma'])
            
            log_sales[l, t] = log_y
    
    return np.exp(log_sales)

def generate_sales_deterministic_weibull_c(spend, data, params):
    """Generate sales without error (for ROI calculation)"""
    C, L, T = spend.shape
    log_sales = np.zeros((L, T))
    
    beta_media_loc = params['beta_media_loc']
    alpha = params['alpha_true']
    K = params['K_true']
    S = params['S_true']
    
    def adstock_transform(x, alpha):
        T = len(x)
        result = np.zeros(T)
        for t in range(T):
            if t == 0:
                result[t] = x[t]
            else:
                result[t] = x[t] + alpha * result[t-1]
        return result
    
    def weibull_c_saturation(z, K, S):
        return 1 - np.exp(-(z/K)**S)
    
    for l in range(L):
        for t in range(T):
            log_y = params['beta_0'] + params['beta_location'][l]
            
            for c in range(C):
                spend_adstock = adstock_transform(spend[c, l, :t+1], alpha[c])[-1]
                spend_sat = weibull_c_saturation(spend_adstock, K[c], S[c])
                log_y += beta_media_loc[c, l] * spend_sat
            
            log_y += params['beta_price'] * np.log(data['price'][l, t])
            log_y += params['beta_discount'] * data['discount'][l, t]
            log_y += params['beta_unemployment'] * data['unemployment'][t]
            log_y += params['beta_confidence'] * data['confidence'][t]
            log_y += params['beta_trend'] * t
            log_y += params['beta_sin'] * np.sin(2*np.pi*t/52)
            log_y += params['beta_cos'] * np.cos(2*np.pi*t/52)
            log_y += np.sum(params['gamma_holiday'] * data['holidays'][:, t])
            
            log_sales[l, t] = log_y
    
    return np.exp(log_sales)

def compute_true_roi_weibull_c(spend, data, params):
    """Compute true ROI using counterfactuals"""
    C, L, T = spend.shape
    roi_true = np.zeros(C)
    
    for c in range(C):
        sales_full = generate_sales_deterministic_weibull_c(spend, data, params)
        spend_cf = spend.copy()
        spend_cf[c, :, :] = 0
        sales_cf = generate_sales_deterministic_weibull_c(spend_cf, data, params)
        inc_revenue = np.sum((sales_full - sales_cf) * data['price'])
        total_spend = np.sum(spend[c, :, :])
        roi_true[c] = inc_revenue / total_spend if total_spend > 0 else 0
    
    return roi_true

# ============================================================================
# CONFIGURATION
# ============================================================================

TRUE_SATURATION = "weibull_cshape"

print("="*100)
print(f"MMM SATURATION FUNCTION SELECTION SIMULATION - TRUE DGP: Weibull C-shape")
print("="*100)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

scenarios = {
    'S1': {'C': 4, 'L': 5},
    'S2': {'C': 6, 'L': 8},
    'S3': {'C': 10, 'L': 12},
    'S4': {'C': 12, 'L': 15}
}

T_train = 156
T_test = 52
T = T_train + T_test

opt_tol = 1e-6
opt_maxiter = 1000
n_init = 5
eta_sq = 1.0

SEED = 42

# ============================================================================
# MAIN SIMULATION LOOP
# ============================================================================

all_performance = []
all_roi_comparison = []
all_selection = []

for scenario_name, scenario_config in scenarios.items():
    
    print("\n" + "="*100)
    print(f"RUNNING SCENARIO: {scenario_name}")
    print("="*100)
    
    scenario_start_time = time.time()
    
    C = scenario_config['C']
    L = scenario_config['L']
    
    print(f"Configuration: C={C} channels, L={L} locations")
    print(f"Training: {T_train} weeks, Test: {T_test} weeks")
    print(f"Ground Truth: Weibull C-shape saturation + adstock\n")
    
    np.random.seed(SEED)
    
    beta_media = np.linspace(1.15, 0.55, C)
    alpha_true = np.random.uniform(0.3, 0.6, C)
    
    # Weibull C-shape: 0 < S < 1
    K_true = np.linspace(8000, 4000, C)
    S_true = np.random.uniform(0.65, 0.85, C)
    
    beta_media_loc = np.random.normal(beta_media[:, None], 0.1, (C, L))
    beta_media_loc = np.maximum(beta_media_loc, 0.01)
    
    beta_0 = 7.5
    beta_price = -1.2
    beta_discount = 0.015
    beta_unemployment = -0.05
    beta_confidence = 0.01
    beta_trend = 0.001
    beta_sin = 0.08
    beta_cos = 0.05
    gamma_holiday = np.array([0.10, 0.15, 0.20])
    beta_location = np.random.normal(0, 0.1, L)
    beta_location[0] = 0
    sigma = 0.05
    
    true_params = {
        'beta_media_loc': beta_media_loc,
        'alpha_true': alpha_true,
        'K_true': K_true,
        'S_true': S_true,
        'beta_0': beta_0,
        'beta_location': beta_location,
        'beta_price': beta_price,
        'beta_discount': beta_discount,
        'beta_unemployment': beta_unemployment,
        'beta_confidence': beta_confidence,
        'beta_trend': beta_trend,
        'beta_sin': beta_sin,
        'beta_cos': beta_cos,
        'gamma_holiday': gamma_holiday,
        'sigma': sigma
    }
    
    print(f"True adstock (α): {alpha_true}")
    print(f"True K (scale): {K_true}")
    print(f"True S (shape, 0<S<1): {S_true}")
    
    print("\nGenerating data...")
    spend, exog_data = generate_exogenous_data(C, L, T, seed=SEED)
    sales = generate_sales_weibull_c(spend, exog_data, true_params, seed=SEED)
    roi_true = compute_true_roi_weibull_c(spend, exog_data, true_params)
    
    print(f"True ROI: {roi_true}")
    print(f"Mean True ROI: {np.mean(roi_true):.4f}")
    
    spend_train = spend[:, :, :T_train]
    spend_test = spend[:, :, T_train:]
    sales_train = sales[:, :T_train]
    sales_test = sales[:, T_train:]
    
    exog_train = {k: v[..., :T_train] if v.ndim > 1 else v[:T_train] 
                  for k, v in exog_data.items()}
    exog_test = {k: v[..., T_train:] if v.ndim > 1 else v[T_train:] 
                 for k, v in exog_data.items()}
    
    print("\nCreating model specifications...")
    models = create_model_specs(opt_tol, opt_maxiter, n_init, eta_sq)
    
    print("Model specifications (all estimate adstock):")
    for i, name in enumerate(models.keys(), 1):
        print(f"  {i}. {name}")
    
    print("\nFitting models...")
    results = []
    
    for name, model in models.items():
        model_start = time.time()
        print(f"\n  Fitting {name}...", end=" ")
        
        try:
            model.fit(spend_train, sales_train, exog_train)
            sales_pred_test = model.predict(spend_test, exog_test)
            
            y_true = sales_test.flatten()
            y_pred = sales_pred_test.flatten()
            
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            
            roi_est = model.compute_roi(spend_test, exog_test)
            roi_mae = np.mean(np.abs(roi_true - roi_est))
            roi_mape = np.mean(np.abs((roi_true - roi_est) / roi_true)) * 100
            
            model_time = time.time() - model_start
            
            results.append({
                'Scenario': scenario_name,
                'Model': name,
                'R2': r2,
                'Sales_MAPE': mape,
                'Sales_RMSE': rmse,
                'ROI_MAE': roi_mae,
                'ROI_MAPE': roi_mape,
                'ROI_Est': roi_est,
                'Fit_Time': model_time,
                'Status': 'Success'
            })
            
            print(f"[{model_time:.1f}s] R²={r2:.4f}, ROI-MAE={roi_mae:.4f}")
            
        except Exception as e:
            model_time = time.time() - model_start
            print(f"[{model_time:.1f}s] FAILED: {str(e)[:50]}")
            
            results.append({
                'Scenario': scenario_name,
                'Model': name,
                'R2': np.nan,
                'Sales_MAPE': np.nan,
                'Sales_RMSE': np.nan,
                'ROI_MAE': np.nan,
                'ROI_MAPE': np.nan,
                'ROI_Est': np.full(C, np.nan),
                'Fit_Time': model_time,
                'Status': f'Failed: {str(e)[:50]}'
            })
    
    df_performance = pd.DataFrame([{k: v for k, v in r.items() if k != 'ROI_Est'} 
                                   for r in results])
    
    valid_models = df_performance[df_performance['R2'].notna()]
    if len(valid_models) > 0:
        df_performance['Selected_Traditional'] = df_performance['R2'] == valid_models['R2'].max()
        df_performance['Selected_Proposed'] = df_performance['ROI_MAE'] == valid_models['ROI_MAE'].min()
    else:
        df_performance['Selected_Traditional'] = False
        df_performance['Selected_Proposed'] = False
    
    all_performance.append(df_performance)
    
    roi_data = []
    for c in range(C):
        row = {
            'Scenario': scenario_name,
            'Channel': f'Ch_{c+1}',
            'True_ROI': roi_true[c]
        }
        for r in results:
            if isinstance(r['ROI_Est'], np.ndarray) and len(r['ROI_Est']) > c:
                row[r['Model']] = r['ROI_Est'][c]
            else:
                row[r['Model']] = np.nan
        roi_data.append(row)
    
    df_roi = pd.DataFrame(roi_data)
    all_roi_comparison.append(df_roi)
    
    if len(valid_models) > 0:
        traditional_idx = valid_models['R2'].idxmax()
        proposed_idx = valid_models['ROI_MAE'].idxmin()
        
        traditional_model = df_performance.loc[traditional_idx, 'Model']
        proposed_model = df_performance.loc[proposed_idx, 'Model']
        
        traditional_roi_mae = df_performance.loc[traditional_idx, 'ROI_MAE']
        proposed_roi_mae = df_performance.loc[proposed_idx, 'ROI_MAE']
        
        traditional_r2 = df_performance.loc[traditional_idx, 'R2']
        proposed_r2 = df_performance.loc[proposed_idx, 'R2']
        
        improvement_factor = traditional_roi_mae / proposed_roi_mae
        
        df_selection = pd.DataFrame({
            'Scenario': [scenario_name, scenario_name],
            'Criterion': ['Traditional (Max R²)', 'Proposed (Min ROI-MAE)'],
            'Selected_Model': [traditional_model, proposed_model],
            'R2': [traditional_r2, proposed_r2],
            'ROI_MAE': [traditional_roi_mae, proposed_roi_mae],
            'Improvement_Factor': [1.0, improvement_factor]
        })
        
        all_selection.append(df_selection)
        
        print("\n" + "-"*100)
        print(f"SCENARIO {scenario_name} RESULTS:")
        print("-"*100)
        print(f"Ground Truth: Weibull C-shape")
        print(f"Traditional criterion (Max R²) selected: {traditional_model}")
        print(f"Proposed criterion (Min ROI-MAE) selected: {proposed_model}")
        print(f"Improvement Factor: {improvement_factor:.2f}x")
        
        if proposed_model == 'Weibull C-shape':
            print("✓ PROPOSED CRITERION CORRECTLY IDENTIFIED THE TRUE MODEL")
        else:
            print("✗ Proposed criterion did not identify the true model")
        
        if traditional_model == 'Weibull C-shape':
            print("✓ Traditional criterion correctly identified the true model")
        else:
            print("✗ TRADITIONAL CRITERION FAILED TO IDENTIFY THE TRUE MODEL")
    else:
        print(f"\nWARNING: No valid models for scenario {scenario_name}")
    
    scenario_time = time.time() - scenario_start_time
    print(f"\nScenario {scenario_name} completed in {scenario_time/60:.1f} minutes")

# ============================================================================
# CONSOLIDATE AND SAVE RESULTS
# ============================================================================

print("\n" + "="*100)
print("CONSOLIDATING RESULTS")
print("="*100)

df_all_performance = pd.concat(all_performance, ignore_index=True)
df_all_roi = pd.concat(all_roi_comparison, ignore_index=True)
df_all_selection = pd.concat(all_selection, ignore_index=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

filename_perf = f'results_all_performance_{TRUE_SATURATION}_{timestamp}.csv'
filename_roi = f'results_all_roi_{TRUE_SATURATION}_{timestamp}.csv'
filename_sel = f'results_all_selection_{TRUE_SATURATION}_{timestamp}.csv'

df_all_performance.to_csv(filename_perf, index=False)
df_all_roi.to_csv(filename_roi, index=False)
df_all_selection.to_csv(filename_sel, index=False)

print(f"\nResults saved:")
print(f"  1. {filename_perf}")
print(f"  2. {filename_roi}")
print(f"  3. {filename_sel}")

print("\n" + "="*100)
print("OVERALL SUMMARY")
print("="*100)

print("\nModel Selection Accuracy:")
print(df_all_selection.to_string(index=False))

avg_improvement = df_all_selection[
    df_all_selection['Criterion'] == 'Proposed (Min ROI-MAE)'
]['Improvement_Factor'].mean()

print(f"\nAverage Improvement Factor across all scenarios: {avg_improvement:.2f}x")

traditional_correct = df_all_selection[
    (df_all_selection['Criterion'] == 'Traditional (Max R²)') & 
    (df_all_selection['Selected_Model'] == 'Weibull C-shape')
].shape[0]

proposed_correct = df_all_selection[
    (df_all_selection['Criterion'] == 'Proposed (Min ROI-MAE)') & 
    (df_all_selection['Selected_Model'] == 'Weibull C-shape')
].shape[0]

total_scenarios = len(scenarios)

print(f"\nCorrect Model Selection Rate:")
print(f"  Traditional (Max R²): {traditional_correct}/{total_scenarios} ({100*traditional_correct/total_scenarios:.0f}%)")
print(f"  Proposed (Min ROI-MAE): {proposed_correct}/{total_scenarios} ({100*proposed_correct/total_scenarios:.0f}%)")

print("\n" + "="*100)
print("AVERAGE PERFORMANCE BY MODEL (Across All Scenarios)")
print("="*100)

model_summary = df_all_performance.groupby('Model').agg({
    'R2': ['mean', 'std'],
    'ROI_MAE': ['mean', 'std'],
    'ROI_MAPE': ['mean', 'std'],
    'Fit_Time': 'mean'
}).round(4)

print(model_summary)

print("\n" + "="*100)
print(f"SIMULATION COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)

