import numpy as np
import pandas as pd
from scipy.optimize import minimize

def optimize_portfolio(
        returns: pd.DataFrame,
        objective: str = "sharpe",   # "sharpe" | "min_vol" | "max_return"
        risk_free_rate: float = 0.02
    ) -> dict:
    """
    Compute optimal weights for a portfolio based on the specified objective.
    
    Returns dict with:
        - weights: dict {ticker: weight}
        - expected_return: float (annualized)
        - volatility: float (annualized)
        - sharpe_ratio: float
    """
    mu = returns.mean() * 252 # trading days in a year
    Sigma = returns.cov() * 252

    # neg_sharpe(weights), portfolio_vol(weights), neg_return(weights)
    neg_sharpe = lambda w: -(w @ mu - risk_free_rate) / np.sqrt(w @ Sigma @ w)
    portfolio_vol = lambda w: np.sqrt(w @ Sigma @ w)
    neg_return = lambda w: -(w @ mu)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(len(mu))]\
    
    if objective == "sharpe":
        result = minimize(neg_sharpe, x0=np.ones(len(mu)) / len(mu), constraints=constraints, bounds=bounds)
    elif objective == "min_vol":
        result = minimize(portfolio_vol, x0=np.ones(len(mu)) / len(mu), constraints=constraints, bounds=bounds)
    elif objective == "max_return":
        result = minimize(neg_return, x0=np.ones(len(mu)) / len(mu), constraints=constraints, bounds=bounds)
    else:
        raise ValueError("Invalid objective. Choose from 'sharpe', 'min_vol', 'max_return'.")
    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    weights = np.where(result.x < 1e-4, 0, result.x)
    weights = weights / weights.sum()  # normalize to ensure sum to 1

    return {
        'weights': dict(zip(returns.columns, weights)),
        'expected_return': weights @ mu,
        'volatility': np.sqrt(weights @ Sigma @ weights),
        'sharpe_ratio': (weights @ mu - risk_free_rate) / np.sqrt(weights @ Sigma @ weights)
    }