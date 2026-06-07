#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy        as np
import yfinance     as yf
from scipy.optimize import minimize
from typing         import Tuple, Dict


# %% SYNTHETIC DATA
def generate_synthetic_prices(
    n_assets: int = 5, n_days: int = 1500, seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic price series using geometric Brownian motion.
    """
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.05, 0.15, n_assets)
    sigma = rng.uniform(0.1, 0.3, n_assets)
    dt = 1 / 252
    prices = np.zeros((n_days, n_assets))
    prices[0] = 100

    for t in range(1, n_days):
        shock = rng.normal(0, 1, n_assets)
        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shock
        )
    return prices


# %% DATA


def download_prices(
    symbols: list[str],
    start: str = "2018-01-01",
    testing: bool = False
) -> np.ndarray:
    """
    Download prices using yfinance.
    """
    if testing:
        print("Testing mode: using synthetic data")
        return generate_synthetic_prices(len(symbols))
    try:
        data = yf.download(
            tickers=symbols,
            start=start,
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if data.empty:
            raise RuntimeError("Empty dataset")

        # Extract adjusted prices
        prices = data["Close"]
        prices = prices.dropna()
        if prices.shape[0] < 2:
            raise RuntimeError("Too few observations")
        return prices.values
    except Exception:
        print("Can't download data!")


def compute_log_returns(prices: np.ndarray) -> np.ndarray:
    """
    Compute log returns from price matrix.
    """
    assert prices.shape[0] > 1, "Need at least two observations"
    returns = np.log(prices[1:] / prices[:-1])
    return returns


# %% STATISTICS


def mean_returns(returns: np.ndarray) -> np.ndarray:
    """
    Annualized mean returns.
    """
    return np.mean(returns, axis=0) * 252


def covariance_matrix(returns: np.ndarray) -> np.ndarray:
    """
    Annualized covariance matrix.
    """
    return np.cov(returns, rowvar=False) * 252


# %% PORTFOLIO METRICS


def portfolio_return(weights: np.ndarray, mean_ret: np.ndarray) -> float:
    return float(np.dot(weights, mean_ret))


def portfolio_variance(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(np.dot(weights, cov_matrix @ weights))


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    return float(np.sqrt(portfolio_variance(weights, cov_matrix)))


def portfolio_sharpe(
    weights: np.ndarray,
    mean_ret: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free: float = 0,
) -> float:
    r = portfolio_return(weights, mean_ret)
    v = portfolio_volatility(weights, cov_matrix)
    assert v > 0
    return float((r - risk_free) / v)


# %% WEIGHT UTILITIES
def random_weights(n_assets: int) -> np.ndarray:
    w = np.random.random(n_assets)
    w /= w.sum()
    return w

def generate_random_portfolios(n_sim: int, n_assets: int) -> np.ndarray:
    weights = np.random.random((n_sim, n_assets))
    weights /= weights.sum(axis=1)[:, None]
    return weights

# %% MONTE CARLO
def monte_carlo_portfolios(
    weights: np.ndarray, mean_ret: np.ndarray, cov_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    returns = weights @ mean_ret
    var = np.einsum("ij,jk,ik->i", weights, cov_matrix, weights)
    vol = np.sqrt(var)
    sharpe = returns / vol
    return returns, vol, sharpe

# %% OPTIMIZATION
def maximize_sharpe(
    mean_ret: np.ndarray, cov_matrix: np.ndarray
) -> np.ndarray:
    n = len(mean_ret)

    def objective(w):
        return -portfolio_sharpe(w, mean_ret, cov_matrix)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    initial = np.ones(n) / n
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    assert result.success
    return result.x


def minimum_variance_portfolio(
    mean_ret: np.ndarray, cov_matrix: np.ndarray
) -> np.ndarray:
    n = len(mean_ret)

    def objective(w):
        return portfolio_variance(w, cov_matrix)

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = tuple((0, 1) for _ in range(n))
    initial = np.ones(n) / n
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )
    assert result.success
    return result.x


# %% MARKOWITZ ANALYTICAL FRONTIER
def markowitz_constants(
    mean_ret: np.ndarray, cov_matrix: np.ndarray
) -> Tuple[float, float, float]:
    inv_cov = np.linalg.inv(cov_matrix)
    ones = np.ones(len(mean_ret))
    A = ones @ inv_cov @ ones
    B = ones @ inv_cov @ mean_ret
    C = mean_ret @ inv_cov @ mean_ret
    return float(A), float(B), float(C)


def efficient_frontier_analytical(
    mean_ret: np.ndarray, cov_matrix: np.ndarray, points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized analytical efficient frontier.
    """
    A, B, C = markowitz_constants(mean_ret, cov_matrix)
    r = np.linspace(mean_ret.min(), mean_ret.max(), points)
    variance = (A * r**2 - 2 * B * r + C) / (A * C - B**2)
    vol = np.sqrt(variance)
    return vol, r


# %% EXAMPLE PIPELINE


def run_example() -> Dict[str, np.ndarray]:
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    prices = download_prices(symbols)
    returns = compute_log_returns(prices)
    mu = mean_returns(returns)
    cov = covariance_matrix(returns)
    weights = generate_random_portfolios(5000, len(symbols))
    r, v, s = monte_carlo_portfolios(weights, mu, cov)
    w_sharpe = maximize_sharpe(mu, cov)
    w_minvol = minimum_variance_portfolio(mu, cov)
    frontier_vol, frontier_ret = efficient_frontier_analytical(mu, cov)
    return {
        "montecarlo_returns": r,
        "montecarlo_vol": v,
        "montecarlo_sharpe": s,
        "max_sharpe_weights": w_sharpe,
        "min_vol_weights": w_minvol,
        "frontier_vol": frontier_vol,
        "frontier_ret": frontier_ret,
    }


# %% MAIN

if __name__ == "__main__":
    results = run_example()
    print(results)
    print("Example run completed")
