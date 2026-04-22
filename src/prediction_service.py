import polars as pl
import pandas as pd
import numpy as np
import logging
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

logger = logging.getLogger("PredictionService")

class PredictionService:
    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """Calcula a tendência macro. Resolve o erro de contagem de grupos."""
        # CORREÇÃO: n_unique() é a forma correta de contar dias no Polars
        if df.is_empty() or df["dia_do_mes"].n_unique() < 5:
            return 0, "Estável"

        try:
            # Agregação para o Nixtla
            df_macro = df.group_by("data_faturamento").len(name="y").to_pandas()
            df_macro = df_macro.rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "total_market"
            
            fcst = MLForecast(
                models=[LGBMRegressor(random_state=42, verbosity=-1)],
                freq='D', lags=[1, 7]
            )
            fcst.fit(df_macro)
            pred = fcst.predict(1)
            
            proj_vol = int(pred["LGBMRegressor"].sum() * 30)
            last_val = df_macro["y"].iloc[-1]
            diff = (pred["LGBMRegressor"].iloc[0] - last_val) / (last_val if last_val > 0 else 1)
            
            trend = "Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável"
            return proj_vol, trend
        except Exception as e:
            logger.error(f"Erro Macro: {e}")
            return 0, "Erro"

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """Gera a tabela nominal com Cliente, Volume e Valor."""
        if df.is_empty(): return pl.DataFrame()

        try:
            # Preparação Nixtla Micro
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

            # Consolidação para Polars
            res = pl.from_pandas(preds).group_by("unique_id").agg(pl.col("LGBMRegressor").sum().round(0).alias("Qtd_Prevista"))
            ticket_medio = df.group_by("marca").agg(pl.col("faturamento").mean().alias("avg_price"))
            
            final = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            return final.with_columns([
                pl.col("unique_id").alias("Cliente"),
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
                pl.lit(0.85).alias("Probabilidade")
            ]).select(["Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"]).sort("Valor_Est", descending=True).head(10)
        except Exception as e:
            logger.error(f"Erro Micro: {e}")
            return pl.DataFrame()