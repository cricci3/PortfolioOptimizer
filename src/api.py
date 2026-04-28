from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
import mlflow

from src.data import fetch_returns
from src.optimizer import optimize_portfolio

@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("portfolio-optimizer")
    yield

app = FastAPI(title="Portfolio Optimizer API", lifespan=lifespan)

class OptimizeRequest(BaseModel):
    tickers: list[str]
    objective: str = "sharpe"
    risk_free_rate: float = 0.02
    period: str = "2y"

    @field_validator('tickers')
    @classmethod
    def check_tickers(cls, v):
        if len(v) < 2:
            raise ValueError("At least 2 tickers are required")
        if not all(ticker.isupper() for ticker in v):
            raise ValueError("Tickers must be in uppercase")
        return v

    @field_validator('objective')
    @classmethod
    def check_objective(cls, v):
        if v not in ["sharpe", "min_vol", "max_return"]:
            raise ValueError("Invalid objective")
        return v

    @field_validator('risk_free_rate')
    @classmethod
    def check_risk_free_rate(cls, v):
        if not (0 <= v <= 1):
            raise ValueError("Risk free rate must be between 0 and 1")
        return v

    @field_validator('period')
    @classmethod
    def check_period(cls, v):
        if not isinstance(v, str) or not v.endswith(('d', 'mo', 'y')):
            raise ValueError("Period must be a string ending with 'd', 'mo', or 'y'")
        return v


class OptimizeResponse(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float

    @field_validator('weights')
    @classmethod
    def check_weights(cls, v):
        if not v:
            raise ValueError("Weights cannot be empty")
        if not all(isinstance(w, (int, float)) and 0 <= w <= 1 for w in v.values()):
            raise ValueError("Weights must be numbers between 0 and 1")
        if abs(sum(v.values()) - 1) > 1e-4:
            raise ValueError("Weights must sum to 1")
        return v


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/optimize", response_model=OptimizeResponse)
def optimize(request: OptimizeRequest):
    try:
        returns = fetch_returns(request.tickers, request.period)
        portfolio = optimize_portfolio(returns, request.objective, request.risk_free_rate)

        with mlflow.start_run():
            mlflow.log_params({
                "tickers": str(request.tickers),
                "objective": request.objective,
                "risk_free_rate": request.risk_free_rate,
                **{t: round(w, 4) for t, w in portfolio['weights'].items()}
            })
            mlflow.log_metrics({
                "expected_return": portfolio['expected_return'],
                "volatility": portfolio['volatility'],
                "sharpe_ratio": portfolio['sharpe_ratio'],
            })

        return OptimizeResponse(
            weights=portfolio['weights'],
            expected_return=portfolio['expected_return'],
            volatility=portfolio['volatility'],
            sharpe_ratio=portfolio['sharpe_ratio']
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    