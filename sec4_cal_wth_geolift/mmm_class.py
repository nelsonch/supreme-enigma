import numpy as np
from scipy.optimize import minimize

class MMM:
    def __init__(self, spec_name, saturation_type, use_adstock=True,
                 opt_tol=1e-6, opt_maxiter=1000, n_init=5, eta_sq=1.0):
        self.spec_name = spec_name
        self.saturation_type = saturation_type  # 'hill_c', 'hill_s', 'weibull_c', 'weibull_s'
        self.use_adstock = use_adstock  # NEW: whether to use adstock transformation
        self.opt_tol = opt_tol
        self.opt_maxiter = opt_maxiter
        self.n_init = n_init
        self.eta_sq = eta_sq
        self.params = None
    
    def adstock_transform(self, x, alpha):
        """Geometric adstock transformation"""
        T = len(x)
        result = np.zeros(T)
        for t in range(T):
            if t == 0:
                result[t] = x[t]
            else:
                result[t] = x[t] + alpha * result[t-1]
        return result
    
    def hill_c_saturation(self, z, K, S):
        """Hill C-shape saturation (0 < S < 1)"""
        return z**S / (K**S + z**S)
    
    def hill_s_saturation(self, z, K, S):
        """Hill S-shape saturation (S > 1)"""
        return z**S / (K**S + z**S)
    
    def weibull_c_saturation(self, z, K, S):
        """Weibull C-shape (0 < S < 1)"""
        return 1 - np.exp(-(z/K)**S)
    
    def weibull_s_saturation(self, z, K, S):
        """Weibull S-shape (S > 1)"""
        return 1 - np.exp(-(z/K)**S)
    
    def apply_saturation(self, z, K, S):
        """Apply appropriate saturation function"""
        if self.saturation_type == 'hill_c':
            return self.hill_c_saturation(z, K, S)
        elif self.saturation_type == 'hill_s':
            return self.hill_s_saturation(z, K, S)
        elif self.saturation_type == 'weibull_c':
            return self.weibull_c_saturation(z, K, S)
        elif self.saturation_type == 'weibull_s':
            return self.weibull_s_saturation(z, K, S)
        else:
            raise ValueError(f"Unknown saturation type: {self.saturation_type}")
    
    def transform_media(self, spend, alpha, K, S):
        """Apply adstock (if enabled) + saturation"""
        C, L, T = spend.shape
        transformed = np.zeros((C, L, T))
        
        for c in range(C):
            for l in range(L):
                if self.use_adstock:
                    # Adstock first, then saturation
                    x_ad = self.adstock_transform(spend[c, l, :], alpha[c])
                    transformed[c, l, :] = self.apply_saturation(x_ad, K[c], S[c])
                else:
                    # Skip adstock, apply saturation directly to raw spend
                    transformed[c, l, :] = self.apply_saturation(spend[c, l, :], K[c], S[c])
        
        return transformed
    
    def neg_log_likelihood(self, params, spend, sales, exog):
        """Negative log-likelihood"""
        C, L, T = spend.shape
        
        # Unpack parameters
        idx = 0
        beta_0_est = params[idx]; idx += 1
        beta_loc_est = np.concatenate([[0], params[idx:idx+L-1]]); idx += L-1
        
        beta_media_est = params[idx:idx+C*L].reshape(C, L); idx += C*L
        
        if self.use_adstock:
            alpha_est = params[idx:idx+C]; idx += C
        else:
            alpha_est = np.zeros(C)  # Dummy alpha (not used)
        
        K_est = params[idx:idx+C]; idx += C
        S_est = params[idx:idx+C]; idx += C
        
        beta_price_est = params[idx]; idx += 1
        beta_discount_est = params[idx]; idx += 1
        beta_unemp_est = params[idx]; idx += 1
        beta_conf_est = params[idx]; idx += 1
        beta_trend_est = params[idx]; idx += 1
        beta_sin_est = params[idx]; idx += 1
        beta_cos_est = params[idx]; idx += 1
        gamma_est = params[idx:idx+3]; idx += 3
        
        sigma_est = params[idx]
        
        # Transform media
        media_transformed = self.transform_media(spend, alpha_est, K_est, S_est)
        
        # Predict log sales
        log_sales_pred = np.zeros((L, T))
        for l in range(L):
            for t in range(T):
                log_y = beta_0_est + beta_loc_est[l]
                
                for c in range(C):
                    log_y += beta_media_est[c, l] * media_transformed[c, l, t]
                
                log_y += beta_price_est * np.log(exog['price'][l, t])
                log_y += beta_discount_est * exog['discount'][l, t]
                log_y += beta_unemp_est * exog['unemployment'][t]
                log_y += beta_conf_est * exog['confidence'][t]
                log_y += beta_trend_est * exog['time'][t]
                log_y += beta_sin_est * np.sin(2*np.pi*exog['time'][t]/52)
                log_y += beta_cos_est * np.cos(2*np.pi*exog['time'][t]/52)
                log_y += np.sum(gamma_est * exog['holidays'][:, t])
                
                log_sales_pred[l, t] = log_y
        
        # Likelihood
        residuals = np.log(sales + 1e-10) - log_sales_pred
        nll = 0.5 * np.sum(residuals**2) / sigma_est**2 + L*T*np.log(sigma_est)
        
        # Add hierarchical prior penalties
        nll += 0.5 * np.sum((beta_media_est - beta_media_est.mean(axis=1, keepdims=True))**2) / self.eta_sq
        
        return nll
    
    def fit(self, spend, sales, exog):
        """Fit model with multiple initializations"""
        C, L, T = spend.shape
        
        # Initialize parameters
        n_params = 1 + (L-1) + C*L  # beta_0, locations, media
        if self.use_adstock:
            n_params += C  # alpha
        n_params += C  # K
        n_params += C  # S
        n_params += 10  # controls + sigma
        
        # Bounds
        bounds = []
        bounds.append((None, None))  # beta_0
        bounds.extend([(None, None)] * (L-1))  # location effects
        bounds.extend([(0, None)] * (C*L))  # media effects (non-negative)
        
        if self.use_adstock:
            bounds.extend([(0.01, 0.99)] * C)  # alpha
        
        bounds.extend([(100, 20000)] * C)  # K
        
        if self.saturation_type in ['hill_c', 'weibull_c']:
            bounds.extend([(0.1, 0.99)] * C)  # S (C-shape)
        else:  # hill_s, weibull_s
            bounds.extend([(1.01, 5.0)] * C)  # S (S-shape)
        
        bounds.extend([
            (None, None),  # price
            (0, None),     # discount
            (None, None),  # unemployment
            (None, None),  # confidence
            (None, None),  # trend
            (None, None),  # sin
            (None, None),  # cos
            (0, None), (0, None), (0, None),  # holidays
            (0.01, None)   # sigma
        ])
        
        best_nll = np.inf
        best_params = None
        
        for init_idx in range(self.n_init):
            # Random initialization
            p0 = []
            p0.append(np.random.randn())  # beta_0
            p0.extend(np.random.randn(L-1) * 0.1)  # locations
            p0.extend(np.random.uniform(0.1, 2, C*L))  # media
            
            if self.use_adstock:
                p0.extend(np.random.uniform(0.2, 0.7, C))  # alpha
            
            p0.extend(np.random.uniform(3000, 10000, C))  # K
            
            if self.saturation_type in ['hill_c', 'weibull_c']:
                p0.extend(np.random.uniform(0.4, 0.9, C))  # S (C-shape)
            else:
                p0.extend(np.random.uniform(1.5, 3.5, C))  # S (S-shape)
            
            p0.extend([
                -1.0, 0.01, -0.03, 0.008, 0.0005,
                0.05, 0.03,
                0.08, 0.12, 0.15,
                0.05
            ])
            
            p0 = np.array(p0)
            
            try:
                result = minimize(
                    self.neg_log_likelihood,
                    p0,
                    args=(spend, sales, exog),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': self.opt_maxiter, 'ftol': self.opt_tol}
                )
                
                if result.fun < best_nll:
                    best_nll = result.fun
                    best_params = result.x
            except Exception as e:
                continue
        
        if best_params is None:
            raise ValueError("All optimization attempts failed")
        
        self.params = best_params
        return self
    
    def predict(self, spend, exog):
        """Predict sales"""
        if self.params is None:
            raise ValueError("Model not fitted")
        
        C, L, T = spend.shape
        
        # Unpack parameters
        idx = 0
        beta_0_est = self.params[idx]; idx += 1
        beta_loc_est = np.concatenate([[0], self.params[idx:idx+L-1]]); idx += L-1
        
        beta_media_est = self.params[idx:idx+C*L].reshape(C, L); idx += C*L
        
        if self.use_adstock:
            alpha_est = self.params[idx:idx+C]; idx += C
        else:
            alpha_est = np.zeros(C)  # Dummy alpha
        
        K_est = self.params[idx:idx+C]; idx += C
        S_est = self.params[idx:idx+C]; idx += C
        
        beta_price_est = self.params[idx]; idx += 1
        beta_discount_est = self.params[idx]; idx += 1
        beta_unemp_est = self.params[idx]; idx += 1
        beta_conf_est = self.params[idx]; idx += 1
        beta_trend_est = self.params[idx]; idx += 1
        beta_sin_est = self.params[idx]; idx += 1
        beta_cos_est = self.params[idx]; idx += 1
        gamma_est = self.params[idx:idx+3]
        
        # Transform media
        media_transformed = self.transform_media(spend, alpha_est, K_est, S_est)
        
        # Predict
        log_sales_pred = np.zeros((L, T))
        for l in range(L):
            for t in range(T):
                log_y = beta_0_est + beta_loc_est[l]
                
                for c in range(C):
                    log_y += beta_media_est[c, l] * media_transformed[c, l, t]
                
                log_y += beta_price_est * np.log(exog['price'][l, t])
                log_y += beta_discount_est * exog['discount'][l, t]
                log_y += beta_unemp_est * exog['unemployment'][t]
                log_y += beta_conf_est * exog['confidence'][t]
                log_y += beta_trend_est * exog['time'][t]
                log_y += beta_sin_est * np.sin(2*np.pi*exog['time'][t]/52)
                log_y += beta_cos_est * np.cos(2*np.pi*exog['time'][t]/52)
                log_y += np.sum(gamma_est * exog['holidays'][:, t])
                
                log_sales_pred[l, t] = log_y
        
        return np.exp(log_sales_pred)
    
    def compute_roi(self, spend, exog):
        """Compute ROI via counterfactuals"""
        C, L, T = spend.shape
        roi = np.zeros(C)
        
        sales_full = self.predict(spend, exog)
        
        for c in range(C):
            spend_cf = spend.copy()
            spend_cf[c, :, :] = 0
            sales_cf = self.predict(spend_cf, exog)
            
            inc_revenue = np.sum((sales_full - sales_cf) * exog['price'])
            total_spend = np.sum(spend[c, :, :])
            
            roi[c] = inc_revenue / total_spend if total_spend > 0 else 0
        
        return roi
