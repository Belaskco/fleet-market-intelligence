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
    Trata os dados do Parquet GMSA e aplica Nixtla com fallback estatístico robusto.
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """Calcula a trajetória macro para o KPI de Forecast com fallback automático."""
        if df.is_empty() or "data_faturamento" not in df.columns:
            return 0, "Sem Dados"
        
        try:
            # Agrupamento diário para série temporal
            v_dia = df.group_by("data_faturamento").len(name="y").sort("data_faturamento")
            
            # Nixtla (lags=[1, 2]) exige massa crítica. 
            # Se tivermos menos de 7 dias, o fallback linear é muito mais preciso e estável.
            if v_dia["data_faturamento"].n_unique() < 7:
                return PredictionService._linear_fallback(v_dia)

            # Processamento Nixtla
            df_macro = v_dia.to_pandas().rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_total"
            
            fcst = MLForecast(
                models=[LGBMRegressor(random_state=42, verbosity=-1)], 
                freq='D', 
                lags=[1, 2]
            )
            fcst.fit(df_macro)
            pred = fcst.predict(1)
            
            # Extração segura do valor previsto
            model_col = "LGBMRegressor"
            if model_col not in pred.columns:
                return PredictionService._linear_fallback(v_dia)
                
            val_pred = pred[model_col].iloc[0]
            proj_vol = int(val_pred * 30)
            
            last_val = df_macro["y"].iloc[-1]
            diff = (val_pred - last_val) / (last_val if last_val > 0 else 1)
            
            trend_status = "Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável"
            return proj_vol, trend_status

        except Exception as e:
            logger.warning(f"Falha no motor Nixtla: {e}. Acionando fallback linear.")
            return PredictionService._linear_fallback(v_dia)

    @staticmethod
    def get_daily_forecast(df: pl.DataFrame, horizon=7):
        """Gera a linha pontilhada futura no gráfico SPC com tratamento de erros."""
        if df.is_empty() or "data_faturamento" not in df.columns: 
            return pl.DataFrame()
            
        try:
            v_dia = df.group_by("data_faturamento").len(name="y").sort("data_faturamento")
            if v_dia["data_faturamento"].n_unique() < 7:
                return pl.DataFrame()

            df_macro = v_dia.to_pandas().rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_total"
            
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 2])
            fcst.fit(df_macro)
            preds = fcst.predict(horizon)
            
            if "ds" not in preds.columns:
                return pl.DataFrame()
                
            preds['dia_do_mes'] = pd.to_datetime(preds['ds']).dt.day
            model_col = "LGBMRegressor"
            
            return pl.from_pandas(preds).select([
                pl.col("dia_do_mes"), 
                pl.col(model_col).alias("vol")
            ])
        except Exception as e:
            logger.error(f"Erro no forecast diário: {e}")
            return pl.DataFrame()

    @staticmethod
    def identify_anomalies(df: pl.DataFrame):
        """Diagnóstico estatístico via Z-Score."""
        if df.is_empty(): return pl.DataFrame()
        v_dia = df.group_by("dia_do_mes").len(name="vol")
        if v_dia.is_empty(): return pl.DataFrame()
        
        m, s = v_dia["vol"].mean(), v_dia["vol"].std()
        if s == 0 or s is None: return pl.DataFrame()
        
        return v_dia.filter((pl.col("vol") > m + 2*s) | (pl.col("vol") < m - 2*s)).sort("dia_do_mes")

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """Tabela nominal por cliente (Bloco Oportunidades) com Nixtla."""
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
            
            # Filtro de marcas com histórico mínimo para evitar erros de fit()
            valid_ids = ts_data.groupby('unique_id').size()
            ts_filtered = ts_data[ts_data['unique_id'].isin(valid_ids[valid_ids > 2].index)]

            if ts_filtered.empty:
                return pl.DataFrame()

            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1])
            fcst.fit(ts_filtered)
            preds = fcst.predict(7)

            model_col = "LGBMRegressor"
            res = pl.from_pandas(preds).group_by("unique_id").agg(
                pl.col(model_col).sum().round(0).alias("Qtd_Prevista")
            )
            
            ticket_medio = df.group_by("marca").agg(pl.col("faturamento").mean().alias("avg_price"))
            final = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            
            return final.with_columns([
                pl.col("unique_id").alias("Cliente"),
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
                pl.lit(0.85).alias("Probabilidade")
            ]).select(["Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"]).sort("Valor_Est", descending=True).head(10)
        except Exception as e:
            logger.error(f"Erro em predições nominais: {e}")
            return pl.DataFrame()

    @staticmethod
    def _linear_fallback(v_dia: pl.DataFrame):
        """
        Cálculo de tendência via Regressão Linear para cenários de baixa densidade de dados.
        Garante que o dashboard exiba informações mesmo em condições adversas.
        """
        try:
            y = v_dia["y"].to_numpy()
            if len(y) < 2:
                return int(y.sum() if len(y) > 0 else 0), "Estável (Média)"
                
            X = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            
            proj_vol = int(y.mean() * 30)
            trend_status = "Alta" if slope > 0.1 else "Baixa" if slope < -0.1 else "Estável"
            
            return proj_vol, trend_status
        except:
            return 0, "Indisponível"