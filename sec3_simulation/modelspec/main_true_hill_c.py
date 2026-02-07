import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

from mmm_class import MMM
from util import (generate_exogenous_data, generate_sales, compute_true_roi, 
                  create_model_specs)

# ============================================================================
# CONFIGURATION
# ============================================================================

print("="*100)
print("MMM SATURATION FUNCTION SELECTION SIMULATION")
print("="*100)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Define all scenarios
scenarios = {
    'S1': {'C': 4, 'L': 5},
    'S2': {'C': 6, 'L': 8},
    'S3': {'C': 10, 'L': 12},
    'S4': {'C': 12, 'L': 15}
}

# Fixed parameters
T_train = 156
T_test = 52
T = T_train + T_test

# Optimization settings
opt_tol = 1e-6
opt_maxiter = 1000
n_init = 5
eta_sq = 1.0

# Random seed
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
    print(f"Ground Truth: Hill C-shape saturation + adstock\n")
    
    # ------------------------------------------------------------------------
    # Generate True Parameters
    # ------------------------------------------------------------------------
    
    np.random.seed(SEED)
    
    # Media effects (base coefficients)
    beta_media = np.linspace(1.15, 0.55, C)
    
    # Adstock parameters (vary by channel)
    alpha_true = np.random.uniform(0.3, 0.6, C)
    
    # Hill C-shape saturation parameters (0 < S < 1)
    K_true = np.linspace(8000, 4000, C)
    S_true = np.random.uniform(0.65, 0.85, C)
    
    # Location-specific media effects (C x L)
    beta_media_loc = np.random.normal(beta_media[:, None], 0.1, (C, L))
    beta_media_loc = np.maximum(beta_media_loc, 0.01)
    
    # Control variables
    beta_0 = 7.5
    beta_price = -1.2
    beta_discount = 0.015
    beta_unemployment = -0.05
    beta_confidence = 0.01
    beta_trend = 0.001
    
    # Seasonality
    beta_sin = 0.08
    beta_cos = 0.05
    
    # Holidays
    gamma_holiday = np.array([0.10, 0.15, 0.20])
    
    # Location effects
    beta_location = np.random.normal(0, 0.1, L)
    beta_location[0] = 0
    
    # Error variance
    sigma = 0.05
    
    # Package true parameters
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
    print(f"True K (half-saturation): {K_true}")
    print(f"True S (shape): {S_true}")
    
    # ------------------------------------------------------------------------
    # Generate Data
    # ------------------------------------------------------------------------
    
    print("\nGenerating data...")
    spend, exog_data = generate_exogenous_data(C, L, T, seed=SEED)
    sales = generate_sales(spend, exog_data, true_params, seed=SEED)
    roi_true = compute_true_roi(spend, exog_data, true_params)
    
    print(f"True ROI: {roi_true}")
    print(f"Mean True ROI: {np.mean(roi_true):.4f}")
    
    # Train/test split
    spend_train = spend[:, :, :T_train]
    spend_test = spend[:, :, T_train:]
    sales_train = sales[:, :T_train]
    sales_test = sales[:, T_train:]
    
    exog_train = {k: v[..., :T_train] if v.ndim > 1 else v[:T_train] 
                  for k, v in exog_data.items()}
    exog_test = {k: v[..., T_train:] if v.ndim > 1 else v[T_train:] 
                 for k, v in exog_data.items()}
    
    # ------------------------------------------------------------------------
    # Create and Fit Models
    # ------------------------------------------------------------------------
    
    print("\nCreating model specifications...")
    models = create_model_specs(opt_tol, opt_maxiter, n_init, eta_sq)
    
    print("Model specifications (all estimate adstock):")
    for i, name in enumerate(models.keys(), 1):
        print(f"  {i}. {name}")
    
    # Fit and evaluate all models
    print("\nFitting models...")
    results = []
    
    for name, model in models.items():
        model_start = time.time()
        print(f"\n  Fitting {name}...", end=" ")
        
        try:
            # Fit on training data
            model.fit(spend_train, sales_train, exog_train)
            
            # Predict on test set
            sales_pred_test = model.predict(spend_test, exog_test)
            
            # Sales metrics
            y_true = sales_test.flatten()
            y_pred = sales_pred_test.flatten()
            
            r2 = r2_score(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            
            # ROI metrics
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
            
            # Add placeholder results for failed model
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
    
    # ------------------------------------------------------------------------
    # Process Results
    # ------------------------------------------------------------------------
    
    # DataFrame 1: Model Performance Summary
    df_performance = pd.DataFrame([{k: v for k, v in r.items() if k != 'ROI_Est'} 
                                   for r in results])
    
    # Handle NaN values in selection
    valid_models = df_performance[df_performance['R2'].notna()]
    if len(valid_models) > 0:
        df_performance['Selected_Traditional'] = df_performance['R2'] == valid_models['R2'].max()
        df_performance['Selected_Proposed'] = df_performance['ROI_MAE'] == valid_models['ROI_MAE'].min()
    else:
        df_performance['Selected_Traditional'] = False
        df_performance['Selected_Proposed'] = False
    
    all_performance.append(df_performance)
    
    # DataFrame 2: Channel-Level ROI Comparison
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
    
    # DataFrame 3: Selection Summary
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
        
        # Print summary
        print("\n" + "-"*100)
        print(f"SCENARIO {scenario_name} RESULTS:")
        print("-"*100)
        print(f"Ground Truth: Hill C-shape")
        print(f"Traditional criterion (Max R²) selected: {traditional_model}")
        print(f"Proposed criterion (Min ROI-MAE) selected: {proposed_model}")
        print(f"Improvement Factor: {improvement_factor:.2f}x")
        
        if proposed_model == 'Hill C-shape':
            print("✓ PROPOSED CRITERION CORRECTLY IDENTIFIED THE TRUE MODEL")
        else:
            print("✗ Proposed criterion did not identify the true model")
        
        if traditional_model == 'Hill C-shape':
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

# Combine all results
df_all_performance = pd.concat(all_performance, ignore_index=True)
df_all_roi = pd.concat(all_roi_comparison, ignore_index=True)
df_all_selection = pd.concat(all_selection, ignore_index=True)

# Save to CSV
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

filename_perf = f'results_all_performance_{timestamp}.csv'
filename_roi = f'results_all_roi_{timestamp}.csv'
filename_sel = f'results_all_selection_{timestamp}.csv'

df_all_performance.to_csv(filename_perf, index=False)
df_all_roi.to_csv(filename_roi, index=False)
df_all_selection.to_csv(filename_sel, index=False)

print(f"\nResults saved:")
print(f"  1. {filename_perf}")
print(f"  2. {filename_roi}")
print(f"  3. {filename_sel}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*100)
print("OVERALL SUMMARY")
print("="*100)

# Selection accuracy by criterion
print("\nModel Selection Accuracy:")
print(df_all_selection.to_string(index=False))

# Average improvement factor
avg_improvement = df_all_selection[
    df_all_selection['Criterion'] == 'Proposed (Min ROI-MAE)'
]['Improvement_Factor'].mean()

print(f"\nAverage Improvement Factor across all scenarios: {avg_improvement:.2f}x")

# Success rate
traditional_correct = df_all_selection[
    (df_all_selection['Criterion'] == 'Traditional (Max R²)') & 
    (df_all_selection['Selected_Model'] == 'Hill C-shape')
].shape[0]

proposed_correct = df_all_selection[
    (df_all_selection['Criterion'] == 'Proposed (Min ROI-MAE)') & 
    (df_all_selection['Selected_Model'] == 'Hill C-shape')
].shape[0]

total_scenarios = len(scenarios)

print(f"\nCorrect Model Selection Rate:")
print(f"  Traditional (Max R²): {traditional_correct}/{total_scenarios} ({100*traditional_correct/total_scenarios:.0f}%)")
print(f"  Proposed (Min ROI-MAE): {proposed_correct}/{total_scenarios} ({100*proposed_correct/total_scenarios:.0f}%)")

# Performance by model across scenarios
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

# ============================================================================
# FINISH
# ============================================================================

print("\n" + "="*100)
print(f"SIMULATION COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
