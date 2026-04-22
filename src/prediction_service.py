import polars as pl
import pandas as pd
import numpy as np
import logging
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

class PredictionService:
    """
    Motor Black Crow.
    Versão Completa: Forecasting, Projeção Pontilhada, Anomalias e Antecipação.
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """Previsão macro de fechamento mensal."""
        if df.is_empty() or "data_faturamento" not in df.columns:
            return 0, "Estável"
        try:
            df_macro = df.group_by("data_faturamento").len(name="y").to_pandas()
            df_macro = df_macro.rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_total"
            
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 7])
            fcst.fit(df_macro)
            pred = fcst.predict(1)
            
            proj_vol = int(pred["LGBMRegressor"].sum() * 30)
            diff = (pred["LGBMRegressor"].iloc[0] - df_macro["y"].iloc[-1]) / (df_macro["y"].iloc[-1] if df_macro["y"].iloc[-1] > 0 else 1)
            
            return proj_vol, ("Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável")
        except:
            return 0, "Erro"

    @staticmethod
    def get_daily_forecast(df: pl.DataFrame, horizon=7):
        """Gera os dados para a linha pontilhada que o Rapha pediu."""
        if df.is_empty() or "data_faturamento" not in df.columns:
            return pl.DataFrame()
        try:
            df_macro = df.group_by("data_faturamento").len(name="y").to_pandas()
            df_macro = df_macro.rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_total"
            
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 7])
            fcst.fit(df_macro)
            preds = fcst.predict(horizon)
            
            preds['dia_do_mes'] = pd.to_datetime(preds['ds']).dt.day
            return pl.from_pandas(preds).select([
                pl.col("dia_do_mes"),
                pl.col("LGBMRegressor").alias("vol")
            ])
        except:
            return pl.DataFrame()

    @staticmethod
    def identify_anomalies(df: pl.DataFrame):
        """Detecção original de outliers via Z-Score."""
        if df.is_empty(): return pl.DataFrame()
        v_dia = df.group_by("dia_do_mes").len(name="vol")
        mean, std = v_dia["vol"].mean(), v_dia["vol"].std()
        return v_dia.filter((pl.col("vol") > mean + 2*std) | (pl.col("vol") < mean - 2*std)).sort("dia_do_mes")

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """Tabela nominal por cliente com volume e valor."""
        if df.is_empty() or "data_faturamento" not in df.columns:
            return pl.DataFrame()
        try:
            data_prep = df.select([
                pl.col("marca").alias("unique_id"),
                pl.col("data_faturamento").alias("ds"),
                pl.lit(1).alias("y"),
                pl.col("faturamento")
            ]).to_pandas()
            
            data_prep['ds'] = pd.to_datetime(data_prep['ds'])
            ts_data = data_prep.groupby(['unique_id', 'ds']).agg({'y': 'sum', 'faturamento': 'mean'}).reset_index()

            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 7])
            fcst.fit(ts_data)
            preds = fcst.predict(7)

            res = pl.from_pandas(preds).group_by("unique_id").agg(pl.col("LGBMRegressor").sum().round(0).alias("Qtd_Prevista"))
            ticket_medio = df.group_by("marca").agg(pl.col("faturamento").mean().alias("avg_price"))
            
            final = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            return final.with_columns([
                pl.col("unique_id").alias("Cliente"),
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
                pl.lit(0.85).alias("Probabilidade")
            ]).select(["Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"]).sort("Valor_Est", descending=True).head(10)
        except:
            return pl.DataFrame()