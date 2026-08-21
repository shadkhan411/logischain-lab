FROM python:3.12-slim

WORKDIR /app

# System deps for scientific python / xgboost / lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Sanity check on build: import every top-level module
RUN python -c "import src.data.synthetic_generator, src.models.gnn, src.models.tcn, \
    src.models.transformer, src.models.xgboost_model, src.models.survival, \
    src.models.ensemble, src.financial.trade_finance_default, \
    src.financial.ccc_predictor, src.financial.credit_risk_scorer, \
    src.simulation.engine, src.simulation.game_modes; print('LogisChain AI: all modules import cleanly')"

CMD ["python", "-m", "demo.run_pipeline", "--quick"]
