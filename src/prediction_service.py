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
    """
    Motor Black Crow - Versão Fleet Real.
    Trata os dados do Parquet GMSA e aplica Nixtla com fallback estatístico.
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """Calcula a trajetória macro para o KPI de Forecast."""
        if df.is_empty() or "data_faturamento" not in df.columns:
            return 0, "Sem Dados"
        
        try:
            # Agrupamento diário para série temporal
            v_dia = df.group_by("data_faturamento").len(name="y").sort("data_faturamento")
            
            # Fallback se houver poucos dias para o Nixtla (precisa de lags)
            if v_dia["data_faturamento"].n_unique() < 5:
                return PredictionService._linear_fallback(v_dia)

            # Processamento Nixtla
            df_macro = v_dia.to_pandas().rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_total"
            
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 2])
            fcst.fit(df_macro)
            pred = fcst.predict(1)
            
            proj_vol = int(pred["LGBMRegressor"].sum() * 30)
            last_val = df_macro["y"].iloc[-1]
            diff = (pred["LGBMRegressor"].iloc[0] - last_val) / (last_val if last_val > 0 else 1)
            
            return proj_vol, ("Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável")
        except:
            return 0, "Erro"

    @staticmethod
    def get_daily_forecast(df: pl.DataFrame, horizon=7):
        """Gera a linha pontilhada futura no gráfico SPC."""
        if df.is_empty() or "data_faturamento" not in df.columns: return pl.DataFrame()
        try:
            df_macro = df.group_by("data_faturamento").len(name="y").to_pandas().rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_total"
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 2])
            fcst.fit(df_macro)
            preds = fcst.predict(horizon)
            preds['dia_do_mes'] = pd.to_datetime(preds['ds']).dt.day
            return pl.from_pandas(preds).select([pl.col("dia_do_mes"), pl.col("LGBMRegressor").alias("vol")])
        except: return pl.DataFrame()

    @staticmethod
    def identify_anomalies(df: pl.DataFrame):
        """Diagnóstico estatístico via Z-Score."""
        if df.is_empty(): return pl.DataFrame()
        v_dia = df.group_by("dia_do_mes").len(name="vol")
        m, s = v_dia["vol"].mean(), v_dia["vol"].std()
        if s == 0 or s is None: return pl.DataFrame()
        return v_dia.filter((pl.col("vol") > m + 2*s) | (pl.col("vol") < m - 2*s)).sort("dia_do_mes")

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """Tabela nominal por cliente (Bloco Oportunidades)."""
        if df.is_empty() or "data_faturamento" not in df.columns: return pl.DataFrame()
        try:
            data_prep = df.select([
                pl.col("marca").alias("unique_id"),
                pl.col("data_faturamento").alias("ds"),
                pl.lit(1).alias("y"),
                pl.col("faturamento")
            ]).to_pandas()
            data_prep['ds'] = pd.to_datetime(data_prep['ds'])
            ts_data = data_prep.groupby(['unique_id', 'ds']).agg({'y': 'sum', 'faturamento': 'mean'}).reset_index()
            
            # Filtro de marcas com histórico mínimo
            valid_ids = ts_data.groupby('unique_id').size()
            ts_filtered = ts_data[ts_data['unique_id'].isin(valid_ids[valid_ids > 1].index)]

            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1])
            fcst.fit(ts_filtered)
            preds = fcst.predict(7)

            res = pl.from_pandas(preds).group_by("unique_id").agg(pl.col("LGBMRegressor").sum().round(0).alias("Qtd_Prevista"))
            ticket_medio = df.group_by("marca").agg(pl.col("faturamento").mean().alias("avg_price"))
            final = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            return final.with_columns([
                pl.col("unique_id").alias("Cliente"),
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
                pl.lit(0.85).alias("Probabilidade")
            ]).select(["Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"]).sort("Valor_Est", descending=True).head(10)
        except: return pl.DataFrame()

    @staticmethod
    def _linear_fallback(v_dia):
        """Fallback para quando o Nixtla não tem dados suficientes."""
        y = v_dia["y"].to_numpy()
        return int(y.mean() * 30), "Estável (Média)"