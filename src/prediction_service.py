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
    Motor de Inteligência Black Crow.
    Versão Full: Nixtla, Fallbacks Lineares, Detecção de Anomalias e Logic Engine.
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """Previsão macro de fechamento mensal com fallback."""
        if df.is_empty() or "data_faturamento" not in df.columns:
            return 0, "Sem Dados"
        try:
            v_dia = df.group_by("data_faturamento").len(name="y").sort("data_faturamento")
            if v_dia["data_faturamento"].n_unique() < 10:
                return PredictionService._linear_trend_fallback(v_dia)

            df_macro = v_dia.to_pandas().rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_total"
            
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 2])
            fcst.fit(df_macro)
            pred = fcst.predict(1)
            
            val_pred = pred["LGBMRegressor"].iloc[0]
            proj_vol = int(val_pred * 30)
            last_val = df_macro["y"].iloc[-1]
            diff = (val_pred - last_val) / (last_val if last_val > 0 else 1)
            
            return proj_vol, ("Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável")
        except:
            return PredictionService._linear_trend_fallback(v_dia)

    @staticmethod
    def get_daily_forecast(df: pl.DataFrame, horizon=7):
        """Gera a linha pontilhada futura para o gráfico SPC."""
        if df.is_empty() or "data_faturamento" not in df.columns: 
            return pl.DataFrame()
        try:
            v_dia = df.group_by("data_faturamento").len(name="y").sort("data_faturamento")
            if v_dia["data_faturamento"].n_unique() < 10:
                return PredictionService._linear_forecast_fallback(v_dia, horizon)

            df_macro = v_dia.to_pandas().rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_total"
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 2])
            fcst.fit(df_macro)
            preds = fcst.predict(horizon)
            
            preds['dia_do_mes'] = pd.to_datetime(preds['ds']).dt.day
            # Garante schema limpo para o concat na interface
            return pl.from_pandas(preds).select([
                pl.col("dia_do_mes").cast(pl.Int64), 
                pl.col("LGBMRegressor").alias("vol").cast(pl.Float64)
            ])
        except:
            return PredictionService._linear_forecast_fallback(v_dia, horizon)

    @staticmethod
    def identify_anomalies(df: pl.DataFrame):
        """Diagnóstico de Outliers via Z-Score."""
        if df.is_empty(): return pl.DataFrame()
        v_dia = df.group_by(pl.col("data_faturamento").dt.day().alias("dia_do_mes")).len(name="vol")
        m, s = v_dia["vol"].mean(), v_dia["vol"].std()
        if s == 0 or s is None: return pl.DataFrame()
        return v_dia.filter((pl.col("vol") > m + 2*s) | (pl.col("vol") < m - 2*s)).sort("dia_do_mes")

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """Antecipação Nominal por Cliente."""
        if df.is_empty() or "data_faturamento" not in df.columns: return pl.DataFrame()
        try:
            data_prep = df.select([pl.col("marca").alias("unique_id"), pl.col("data_faturamento").alias("ds"), pl.lit(1).alias("y"), pl.col("faturamento")]).to_pandas()
            data_prep['ds'] = pd.to_datetime(data_prep['ds'])
            ts_data = data_prep.groupby(['unique_id', 'ds']).agg({'y': 'sum', 'faturamento': 'mean'}).reset_index()
            valid_ids = ts_data.groupby('unique_id').size()
            ts_filtered = ts_data[ts_data['unique_id'].isin(valid_ids[valid_ids > 2].index)]

            if ts_filtered.empty: return PredictionService._heuristic_fallback(df)

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
        except: return PredictionService._heuristic_fallback(df)

    @staticmethod
    def _linear_trend_fallback(v_dia):
        y = v_dia["y"].to_numpy()
        if len(y) < 2: return int(y.sum() if len(y)>0 else 0), "Estável"
        X = np.arange(len(y)).reshape(-1, 1)
        slope = LinearRegression().fit(X, y).coef_[0]
        return int(y.mean() * 30), ("Alta" if slope > 0.05 else "Baixa" if slope < -0.05 else "Estável")

    @staticmethod
    def _linear_forecast_fallback(v_dia, horizon):
        try:
            y = v_dia["y"].to_numpy()
            if len(y) < 2: return pl.DataFrame()
            X = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            preds = model.predict(np.arange(len(y), len(y) + horizon).reshape(-1, 1))
            last_date = v_dia["data_faturamento"].max()
            future_days = [(last_date + pd.Timedelta(days=i+1)).day for i in range(horizon)]
            return pl.DataFrame({
                "dia_do_mes": pl.Series(future_days, dtype=pl.Int64), 
                "vol": pl.Series(preds, dtype=pl.Float64)
            })
        except: return pl.DataFrame()

    @staticmethod
    def _heuristic_fallback(df):
        return df.group_by("marca").agg([
            pl.len().alias("Qtd_Prevista"),
            pl.col("faturamento").mean().alias("avg_price")
        ]).with_columns([
            pl.col("marca").alias("Cliente"),
            (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
            pl.lit(0.6).alias("Probabilidade")
        ]).sort("Valor_Est", descending=True).head(10)