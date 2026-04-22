import polars as pl
import pandas as pd
import numpy as np
import logging
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

class PredictionService:
    """Motor Nixtla Fleet com Agregação Semanal."""

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """Previsão de volume para o ciclo de 4 semanas seguinte."""
        if df.is_empty() or "purchase_date" not in df.columns: return 0, "Sem Dados"
        try:
            # Agregação Semanal
            v_sem = df.with_columns(
                pl.col("purchase_date").dt.truncate("1w").alias("ds")
            ).group_by("ds").len(name="y").sort("ds").to_pandas()
            
            v_sem["unique_id"] = "fleet_weekly"
            
            # Se tivermos menos de 5 semanas, fallback linear
            if len(v_sem) < 5:
                return PredictionService._linear_trend_fallback(pl.from_pandas(v_sem))

            fcst = MLForecast(
                models=[LGBMRegressor(random_state=42, verbosity=-1)], 
                freq='W-MON', # Weekly starting on Monday
                lags=[1, 4]   # Uma semana e um mês atrás
            )
            fcst.fit(v_sem)
            pred = fcst.predict(4) # Prever as próximas 4 semanas
            
            proj_vol = int(pred["LGBMRegressor"].sum())
            last_val = v_sem["y"].iloc[-1]
            next_val = pred["LGBMRegressor"].iloc[0]
            diff = (next_val - last_val) / (last_val if last_val > 0 else 1)
            
            return proj_vol, ("Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável")
        except Exception as e:
            logger.error(f"Erro Nixtla Semanal: {e}")
            return 0, "Erro"

    @staticmethod
    def get_daily_forecast(df: pl.DataFrame, horizon=4):
        """Gera forecast semanal para a linha pontilhada."""
        if df.is_empty() or "purchase_date" not in df.columns: return pl.DataFrame()
        try:
            v_sem = df.with_columns(
                pl.col("purchase_date").dt.truncate("1w").alias("ds")
            ).group_by("ds").len(name="y").sort("ds").to_pandas()
            v_sem["unique_id"] = "fleet_weekly"
            
            if len(v_sem) < 5: return pl.DataFrame()

            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='W-MON', lags=[1, 4])
            fcst.fit(v_sem)
            preds = fcst.predict(horizon)
            
            return pl.from_pandas(preds).select([
                pl.col("ds").alias("semana"), 
                pl.col("LGBMRegressor").alias("vol").cast(pl.Float64)
            ])
        except: return pl.DataFrame()

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """Antecipação Nominal por Cliente (Visão de Ciclo de 4 semanas)."""
        if df.is_empty(): return pl.DataFrame()
        try:
            data_prep = df.with_columns(
                pl.col("purchase_date").dt.truncate("1w").alias("ds")
            ).select([
                pl.col("marca").alias("unique_id"),
                pl.col("ds"),
                pl.lit(1).alias("y"),
                pl.col("order_value").alias("faturamento")
            ]).to_pandas()
            
            ts_data = data_prep.groupby(['unique_id', 'ds']).agg({'y': 'sum', 'faturamento': 'mean'}).reset_index()
            valid_ids = ts_data.groupby('unique_id').size()
            ts_filtered = ts_data[ts_data['unique_id'].isin(valid_ids[valid_ids > 2].index)]

            if ts_filtered.empty: return pl.DataFrame()

            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='W-MON', lags=[1])
            fcst.fit(ts_filtered)
            preds = fcst.predict(4) # Prever próximo mês (4 semanas)

            res = pl.from_pandas(preds).group_by("unique_id").agg(pl.col("LGBMRegressor").sum().round(0).alias("Qtd_Prevista"))
            ticket_medio = df.group_by("marca").agg(pl.col("order_value").mean().alias("avg_price"))
            final = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            
            return final.with_columns([
                pl.col("unique_id").alias("Cliente"),
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
                pl.lit(0.85).alias("Probabilidade")
            ]).select(["Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"]).sort("Valor_Est", descending=True).head(10)
        except: return pl.DataFrame()

    @staticmethod
    def _linear_trend_fallback(v_dia):
        y = v_dia["vol"].to_numpy()
        X = np.arange(len(y)).reshape(-1, 1)
        slope = LinearRegression().fit(X, y).coef_[0]
        return int(y.mean() * 4), ("Alta" if slope > 0.05 else "Baixa" if slope < -0.05 else "Estável")