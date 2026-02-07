import numpy as np
from scipy.special import erf

def generate_exogenous_data(C, L, T, seed=None):
    """Generate exogenous variables"""
    if seed is not None:
        np.random.seed(seed)
    
    data = {}
    
    # Media spend (LogNormal with correlation)
    mean_spend = np.linspace(5000, 3000, C)
    cov_matrix = 0.3 * np.outer(mean_spend, mean_spend) + 0.7 * np.diag(mean_spend**2)
    
    spend = np.zeros((C, L, T))
    for l in range(L):
        for t in range(T):
            spend[:, l, t] = np.random.multivariate_normal(mean_spend, cov_matrix)
            spend[:, l, t] = np.maximum(spend[:, l, t], 100)  # Floor
    
    # Price (with variation)
    data['price'] = 50 + 5 * np.random.randn(L, T) + 2 * np.sin(2*np.pi*np.arange(T)/52)
    data['price'] = np.maximum(data['price'], 30)
    
    # Discount
    data['discount'] = np.random.beta(2, 5, (L, T)) * 0.3
    
    # Macro variables
    data['unemployment'] = 5 + 0.5 * np.random.randn(T)
    data['confidence'] = 100 + 5 * np.random.randn(T)
    
    # Holidays (week indicators)
    holidays = np.zeros((3, T))
    for year in range(T // 52 + 1):
        base = year * 52
        if base + 25 < T:
            holidays[0, base + 25] = 1  # Independence Day
        if base + 46 < T:
            holidays[1, base + 46] = 1  # Thanksgiving
        if base + 51 < T:
            holidays[2, base + 51] = 1  # Christmas
    data['holidays'] = holidays
    
    # Time trend
    data['time'] = np.arange(T)
    
    return spend, data

def adstock_transform(x, alpha):
    """Geometric adstock transformation"""
    T = len(x)
    result = np.zeros(T)
    for t in range(T):
        if t == 0:
            result[t] = x[t]
        else:
            result[t] = x[t] + alpha * result[t-1]
    return result

def hill_c_saturation(z, K, S):
    """Hill C-shape saturation (0 < S < 1)"""
    return z**S / (K**S + z**S)

def generate_sales(spend, data, params, seed=None):
    """Generate sales using true DGP (Hill C-shape + adstock)"""
    if seed is not None:
        np.random.seed(seed)
    
    C, L, T = spend.shape
    log_sales = np.zeros((L, T))
    
    beta_media_loc = params['beta_media_loc']
    alpha_true = params['alpha_true']
    K_true = params['K_true']
    S_true = params['S_true']
    beta_0 = params['beta_0']
    beta_location = params['beta_location']
    beta_price = params['beta_price']
    beta_discount = params['beta_discount']
    beta_unemployment = params['beta_unemployment']
    beta_confidence = params['beta_confidence']
    beta_trend = params['beta_trend']
    beta_sin = params['beta_sin']
    beta_cos = params['beta_cos']
    gamma_holiday = params['gamma_holiday']
    sigma = params['sigma']
    
    for l in range(L):
        for t in range(T):
            # Base
            log_y = beta_0 + beta_location[l]
            
            # Media effects (Hill C-shape + adstock)
            for c in range(C):
                # Adstock
                spend_adstock = adstock_transform(spend[c, l, :t+1], alpha_true[c])[-1]
                # Hill C-shape saturation
                spend_sat = hill_c_saturation(spend_adstock, K_true[c], S_true[c])
                # Contribution
                log_y += beta_media_loc[c, l] * spend_sat
            
            # Controls
            log_y += beta_price * np.log(data['price'][l, t])
            log_y += beta_discount * data['discount'][l, t]
            log_y += beta_unemployment * data['unemployment'][t]
            log_y += beta_confidence * data['confidence'][t]
            
            # Temporal
            log_y += beta_trend * t
            log_y += beta_sin * np.sin(2*np.pi*t/52)
            log_y += beta_cos * np.cos(2*np.pi*t/52)
            
            # Holidays
            log_y += np.sum(gamma_holiday * data['holidays'][:, t])
            
            # Error
            log_y += np.random.normal(0, sigma)
            
            log_sales[l, t] = log_y
    
    sales = np.exp(log_sales)
    return sales

def generate_sales_deterministic(spend, data, params):
    """Generate sales without error (for ROI calculation)"""
    C, L, T = spend.shape
    log_sales = np.zeros((L, T))
    
    beta_media_loc = params['beta_media_loc']
    alpha = params['alpha_true']
    K = params['K_true']
    S = params['S_true']
    beta_0 = params['beta_0']
    beta_location = params['beta_location']
    beta_price = params['beta_price']
    beta_discount = params['beta_discount']
    beta_unemployment = params['beta_unemployment']
    beta_confidence = params['beta_confidence']
    beta_trend = params['beta_trend']
    beta_sin = params['beta_sin']
    beta_cos = params['beta_cos']
    gamma_holiday = params['gamma_holiday']
    
    for l in range(L):
        for t in range(T):
            log_y = beta_0 + beta_location[l]
            
            for c in range(C):
                spend_adstock = adstock_transform(spend[c, l, :t+1], alpha[c])[-1]
                spend_sat = hill_c_saturation(spend_adstock, K[c], S[c])
                log_y += beta_media_loc[c, l] * spend_sat
            
            log_y += beta_price * np.log(data['price'][l, t])
            log_y += beta_discount * data['discount'][l, t]
            log_y += beta_unemployment * data['unemployment'][t]
            log_y += beta_confidence * data['confidence'][t]
            log_y += beta_trend * t
            log_y += beta_sin * np.sin(2*np.pi*t/52)
            log_y += beta_cos * np.cos(2*np.pi*t/52)
            log_y += np.sum(gamma_holiday * data['holidays'][:, t])
            
            log_sales[l, t] = log_y
    
    return np.exp(log_sales)

def compute_true_roi(spend, data, params):
    """Compute true ROI using counterfactuals"""
    C, L, T = spend.shape
    roi_true = np.zeros(C)
    
    for c in range(C):
        # Full scenario
        sales_full = generate_sales_deterministic(spend, data, params)
        
        # Counterfactual: remove channel c
        spend_cf = spend.copy()
        spend_cf[c, :, :] = 0
        sales_cf = generate_sales_deterministic(spend_cf, data, params)
        
        # Incremental revenue
        inc_revenue = np.sum((sales_full - sales_cf) * data['price'])
        total_spend = np.sum(spend[c, :, :])
        
        roi_true[c] = inc_revenue / total_spend if total_spend > 0 else 0
    
    return roi_true

def create_model_specs(opt_tol=1e-6, opt_maxiter=1000, n_init=5, eta_sq=1.0):
    """Create 6 saturation function model specifications"""
    from mmm_class import MMM
    
    models = {
        'Hill C-shape': MMM('Hill C-shape', saturation_type='hill_c',
                           opt_tol=opt_tol, opt_maxiter=opt_maxiter, n_init=n_init, eta_sq=eta_sq),
        'Hill S-shape': MMM('Hill S-shape', saturation_type='hill_s',
                           opt_tol=opt_tol, opt_maxiter=opt_maxiter, n_init=n_init, eta_sq=eta_sq),
        'Weibull C-shape': MMM('Weibull C-shape', saturation_type='weibull_c',
                              opt_tol=opt_tol, opt_maxiter=opt_maxiter, n_init=n_init, eta_sq=eta_sq),
        'Weibull S-shape': MMM('Weibull S-shape', saturation_type='weibull_s',
                              opt_tol=opt_tol, opt_maxiter=opt_maxiter, n_init=n_init, eta_sq=eta_sq),
        'Sigmoid': MMM('Sigmoid', saturation_type='sigmoid',
                      opt_tol=opt_tol, opt_maxiter=opt_maxiter, n_init=n_init, eta_sq=eta_sq),
        'Erf': MMM('Erf', saturation_type='erf',
                  opt_tol=opt_tol, opt_maxiter=opt_maxiter, n_init=n_init, eta_sq=eta_sq)
    }
    
    return models
