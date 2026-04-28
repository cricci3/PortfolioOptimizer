import yfinance as yf
import pandas as pd
import numpy as np

def fetch_returns(
        tickers: list[str],
        period: str = "2y",
        min_data: int = 30
    ) -> pd.DataFrame:

    data = yf.download(tickers, period=period, progress=False)
    close_prices = pd.DataFrame(data['Close'])

    invalid = close_prices.columns[close_prices.isna().all()].tolist()
    if invalid:
        raise ValueError(f"Tickers not found or no data: {invalid}")

    returns = np.log(close_prices / close_prices.shift(1)).dropna()

    if len(returns) < min_data:
        raise ValueError(f"Not enough data: got {len(returns)} days, need {min_data}.")

    return returns
