import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import warnings
import time
from datetime import datetime
warnings.filterwarnings('ignore')

from mmm_class import MMM
from util import (generate_exogenous_data, generate_sales, compute_true_roi, 
                  create_model_specs_with_misspec, simulate_geo_experiment)

# ============================================================================
# CONFIGURATION
# ============================================================================

print("="*100)
print("MMM CALIBRATION WITH GEO-EXPERIMENTS SIMULATION")
print("="*100)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Define scenarios (S1 and S2 only)
scenarios = {
    'S1': {'C': 4, 'L': 5},
    'S2': {'C': 6, 'L': 8}
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

# Geo-experiment settings
n_test_channels = 2  # Number of channels to test with geo-experiments
n_test_markets = 2   # Number of test markets for holdout
holdout_weeks = 4    # Duration of holdout experiment

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
    # Simulate Geo-Experiments
    # ------------------------------------------------------------------------
    
    print("\n" + "-"*100)
    print("SIMULATING GEO-EXPERIMENTS")
    print("-"*100)
    
    # Select channels to test (first n_test_channels)
    test_channels = list(range(n_test_channels))
    untested_channels = list(range(n_test_channels, C))
    
    print(f"Testing channels: {test_channels}")
    print(f"Untested channels (for validation): {untested_channels}")
    
    # Run geo-experiments
    geo_roi_estimates = simulate_geo_experiment(
        spend_test, sales_test, exog_test, true_params,
        test_channels=test_channels,
        n_test_markets=n_test_markets,
        holdout_weeks=holdout_weeks,
        seed=SEED
    )
    
    print(f"\nGeo-Experiment Results (with measurement noise):")
    for c in test_channels:
        print(f"  Channel {c+1}: Estimated ROI = {geo_roi_estimates[c]:.4f} (True: {roi_true[c]:.4f})")
    
    # ------------------------------------------------------------------------
    # Create and Fit Models
    # ------------------------------------------------------------------------
    
    print("\n" + "-"*100)
    print("CREATING MODEL SPECIFICATIONS")
    print("-"*100)
    
    models = create_model_specs_with_misspec(opt_tol, opt_maxiter, n_init, eta_sq)
    
    print("Model specifications:")
    print("  WITH ADSTOCK:")
    for i, name in enumerate(['Hill C-shape', 'Hill S-shape', 'Weibull C-shape', 'Weibull S-shape'], 1):
        print(f"    {i}. {name}")
    print("  WITHOUT ADSTOCK (Misspecified):")
    for i, name in enumerate(['Hill C-shape (No Adstock)', 'Hill S-shape (No Adstock)', 
                              'Weibull C-shape (No Adstock)', 'Weibull S-shape (No Adstock)'], 5):
        print(f"    {i}. {name}")
    
    # Fit and evaluate all models
    print("\n" + "-"*100)
    print("FITTING MODELS")
    print("-"*100)
    
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
            
            # Calibration error (on tested channels only)
            calibration_error = np.mean(np.abs(geo_roi_estimates[test_channels] - roi_est[test_channels]))
            
            # Validation error (on untested channels only)
            if len(untested_channels) > 0:
                validation_error = np.mean(np.abs(roi_true[untested_channels] - roi_est[untested_channels]))
            else:
                validation_error = np.nan
            
            model_time = time.time() - model_start
            
            results.append({
                'Scenario': scenario_name,
                'Model': name,
                'R2': r2,
                'Sales_MAPE': mape,
                'Sales_RMSE': rmse,
                'ROI_MAE': roi_mae,
                'ROI_MAPE': roi_mape,
                'Calibration_Error': calibration_error,
                'Validation_Error': validation_error,
                'ROI_Est': roi_est,
                'Fit_Time': model_time,
                'Status': 'Success'
            })
            
            print(f"[{model_time:.1f}s] R²={r2:.4f}, ROI-MAE={roi_mae:.4f}, Calib-Err={calibration_error:.4f}")
            
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
                'Calibration_Error': np.nan,
                'Validation_Error': np.nan,
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
        df_performance['Selected_Calibration'] = df_performance['Calibration_Error'] == valid_models['Calibration_Error'].min()
        df_performance['Selected_Oracle'] = df_performance['ROI_MAE'] == valid_models['ROI_MAE'].min()
    else:
        df_performance['Selected_Traditional'] = False
        df_performance['Selected_Calibration'] = False
        df_performance['Selected_Oracle'] = False
    
    all_performance.append(df_performance)
    
    # DataFrame 2: Channel-Level ROI Comparison
    roi_data = []
    for c in range(C):
        row = {
            'Scenario': scenario_name,
            'Channel': f'Ch_{c+1}',
            'True_ROI': roi_true[c],
            'Geo_ROI': geo_roi_estimates[c] if c in test_channels else np.nan,
            'Is_Tested': c in test_channels
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
        calibration_idx = valid_models['Calibration_Error'].idxmin()
        oracle_idx = valid_models['ROI_MAE'].idxmin()
        
        traditional_model = df_performance.loc[traditional_idx, 'Model']
        calibration_model = df_performance.loc[calibration_idx, 'Model']
        oracle_model = df_performance.loc[oracle_idx, 'Model']
        
        traditional_r2 = df_performance.loc[traditional_idx, 'R2']
        calibration_r2 = df_performance.loc[calibration_idx, 'R2']
        oracle_r2 = df_performance.loc[oracle_idx, 'R2']
        
        traditional_calib = df_performance.loc[traditional_idx, 'Calibration_Error']
        calibration_calib = df_performance.loc[calibration_idx, 'Calibration_Error']
        oracle_calib = df_performance.loc[oracle_idx, 'Calibration_Error']
        
        traditional_valid = df_performance.loc[traditional_idx, 'Validation_Error']
        calibration_valid = df_performance.loc[calibration_idx, 'Validation_Error']
        oracle_valid = df_performance.loc[oracle_idx, 'Validation_Error']
        
        df_selection = pd.DataFrame({
            'Scenario': [scenario_name, scenario_name, scenario_name],
            'Criterion': ['Traditional (Max R²)', 'Calibration (Min Calib-Error)', 'Oracle (Min ROI-MAE)'],
            'Selected_Model': [traditional_model, calibration_model, oracle_model],
            'R2': [traditional_r2, calibration_r2, oracle_r2],
            'Calibration_Error': [traditional_calib, calibration_calib, oracle_calib],
            'Validation_Error': [traditional_valid, calibration_valid, oracle_valid]
        })
        
        all_selection.append(df_selection)
        
        # Print summary
        print("\n" + "="*100)
        print(f"SCENARIO {scenario_name} SELECTION RESULTS")
        print("="*100)
        print(f"Ground Truth: Hill C-shape + Adstock")
        print(f"\nTraditional (Max R²) selected: {traditional_model}")
        print(f"  R² = {traditional_r2:.4f}, Calib-Error = {traditional_calib:.4f}, Valid-Error = {traditional_valid:.4f}")
        print(f"\nCalibration (Min Calib-Error) selected: {calibration_model}")
        print(f"  R² = {calibration_r2:.4f}, Calib-Error = {calibration_calib:.4f}, Valid-Error = {calibration_valid:.4f}")
        print(f"\nOracle (Min ROI-MAE) selected: {oracle_model}")
        print(f"  R² = {oracle_r2:.4f}, Calib-Error = {oracle_calib:.4f}, Valid-Error = {oracle_valid:.4f}")
        
        # Compare validation errors
        if calibration_valid < traditional_valid:
            improvement = traditional_valid / calibration_valid
            print(f"\n✓ CALIBRATION-BASED SELECTION WINS: {improvement:.2f}x better validation error")
        elif calibration_valid > traditional_valid:
            deterioration = calibration_valid / traditional_valid
            print(f"\n✗ Calibration-based worse: {deterioration:.2f}x higher validation error")
        else:
            print(f"\n= TIE: Same validation error")
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

filename_perf = f'results_calibration_performance_{timestamp}.csv'
filename_roi = f'results_calibration_roi_{timestamp}.csv'
filename_sel = f'results_calibration_selection_{timestamp}.csv'

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

# Selection comparison
print("\nModel Selection Comparison:")
print(df_all_selection.to_string(index=False))

# Validation error comparison
print("\n" + "="*100)
print("VALIDATION ERROR COMPARISON (Untested Channels)")
print("="*100)

for criterion in ['Traditional (Max R²)', 'Calibration (Min Calib-Error)', 'Oracle (Min ROI-MAE)']:
    criterion_results = df_all_selection[df_all_selection['Criterion'] == criterion]
    avg_valid = criterion_results['Validation_Error'].mean()
    print(f"{criterion:40s}: Avg Validation Error = {avg_valid:.4f}")

# Success metrics
calibration_wins = 0
traditional_wins = 0
ties = 0

for scenario in scenarios.keys():
    scenario_sel = df_all_selection[df_all_selection['Scenario'] == scenario]
    trad_valid = scenario_sel[scenario_sel['Criterion'] == 'Traditional (Max R²)']['Validation_Error'].values[0]
    calib_valid = scenario_sel[scenario_sel['Criterion'] == 'Calibration (Min Calib-Error)']['Validation_Error'].values[0]
    
    if calib_valid < trad_valid:
        calibration_wins += 1
    elif calib_valid > trad_valid:
        traditional_wins += 1
    else:
        ties += 1

print(f"\nWin/Loss Record:")
print(f"  Calibration-based wins: {calibration_wins}/{len(scenarios)}")
print(f"  Traditional wins: {traditional_wins}/{len(scenarios)}")
print(f"  Ties: {ties}/{len(scenarios)}")

# ============================================================================
# FINISH
# ============================================================================

print("\n" + "="*100)
print(f"SIMULATION COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)
