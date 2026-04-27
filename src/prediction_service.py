import polars as pl
import pandas as pd
import numpy as np
import logging
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

# Configuração de Logs para depuração sênior
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

class PredictionService:
    """
    Motor de Inteligência Black Crow v5.6.2.
    Responsável por Forecasts (Nixtla), Tendências e Insights Estratégicos.
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """Calcula a tendência volumétrica para os próximos 4 ciclos semanais."""
        if df.is_empty(): return 0, "Sem Dados"
        try:
            v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("ds")).group_by("ds").len(name="y").sort("ds").to_pandas()
            v_sem["ds"] = pd.to_datetime(v_sem["ds"])
            v_sem["unique_id"] = "fleet_trend"
            
            if len(v_sem) < 5: return PredictionService._linear_trend_fallback(v_sem)

            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='W-MON', lags=[1, 2])
            fcst.fit(v_sem)
            pred = fcst.predict(4)
            
            res_col = pred.columns[-1]
            proj_vol = int(pred[res_col].sum())
            diff = (pred[res_col].iloc[0] - v_sem["y"].iloc[-1]) / (v_sem["y"].iloc[-1] if v_sem["y"].iloc[-1] > 0 else 1)
            
            return proj_vol, ("Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável")
        except Exception as e:
            logger.error(f"Erro no Trend Motor: {e}")
            return 0, "Erro"

    @staticmethod
    def get_daily_forecast(df: pl.DataFrame, horizon=4):
        """Gera a série temporal de previsão para o gráfico SPC."""
        if df.is_empty(): return pl.DataFrame()
        try:
            v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("ds")).group_by("ds").len(name="y").sort("ds").to_pandas()
            v_sem["ds"] = pd.to_datetime(v_sem["ds"])
            v_sem["unique_id"] = "fleet_spc"
            
            if len(v_sem) < 5: return pl.DataFrame()

            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='W-MON', lags=[1, 2])
            fcst.fit(v_sem)
            preds = fcst.predict(horizon)
            
            return pl.from_pandas(preds).select([
                pl.col("ds").alias("semana"), 
                pl.col(preds.columns[-1]).alias("vol").cast(pl.Float64)
            ])
        except Exception as e:
            logger.error(f"Erro no SPC Forecast: {e}")
            return pl.DataFrame()

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """
        Gera a Antecipação Nominal por cliente.
        IMPORTANTE: Retorna apenas dados brutos; a interface processa Recência e Ticket.
        """
        if df.is_empty(): return pl.DataFrame()
        try:
            # Preparação de Dados para o Nixtla (Multi-Séries)
            data_prep = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("ds")).select([
                pl.col("marca").alias("unique_id"), pl.col("ds"), pl.lit(1).alias("y"), pl.col("faturamento")
            ]).to_pandas()
            data_prep["ds"] = pd.to_datetime(data_prep["ds"])
            
            # Agregação por cliente e semana
            ts_data = data_prep.groupby(['unique_id', 'ds']).agg({'y': 'sum', 'faturamento': 'mean'}).reset_index()
            
            # Filtro de clientes com massa crítica (mínimo 3 semanas de histórico)
            counts = ts_data.groupby('unique_id').size()
            valid_ids = counts[counts >= 3].index
            
            if len(valid_ids) == 0: 
                return PredictionService._heuristic_client_fallback(df)

            ts_filtered = ts_data[ts_data['unique_id'].isin(valid_ids)]
            
            # Motor MLForecast
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='W-MON', lags=[1])
            fcst.fit(ts_filtered)
            preds = fcst.predict(4)

            # Consolidação de resultados
            model_col = preds.columns[-1]
            res = pl.from_pandas(preds).group_by("unique_id").agg(pl.col(model_col).sum().round(0).alias("Qtd_Prevista"))
            
            # Mapeamento de Valor Financeiro (Venda Estimada)
            ticket_medio = df.group_by("marca").agg(pl.col("faturamento").mean().alias("avg_price"))
            final = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            
            return final.with_columns([
                pl.col("unique_id").alias("Cliente"), 
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est")
            ]).select(["Cliente", "Qtd_Prevista", "Valor_Est"]).sort("Valor_Est", descending=True).head(15)
            
        except Exception as e:
            logger.error(f"Erro no Client Prediction: {e}")
            return PredictionService._heuristic_client_fallback(df)

    @staticmethod
    def get_strategic_insights(df: pl.DataFrame):
        """Analisa a saúde estatística da carteira (HHI, CV, Confiança)."""
        if df.is_empty(): return {}
        try:
            v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol")
            m, s = v_sem["vol"].mean(), v_sem["vol"].std()
            cv = (s / m) if m > 0 else 0
            
            # Confiança baseada na Estabilidade (CV)
            confianca = max(70.0, min(99.4, 100 * (1 - (cv * 0.2))))
            
            # Índice de Concentração HHI (Herfindahl-Hirschman Index)
            dist_marca = df.group_by("marca").len(name="vendas")
            hhi = (dist_marca["vendas"] / dist_marca["vendas"].sum()).pow(2).sum()
            
            return {
                "confianca": confianca, 
                "hhi": hhi, 
                "estabilidade": "Estável" if cv <= 0.3 else "Volátil", 
                "perfil": "Diversificado" if hhi < 0.25 else "Concentrado", 
                "cv": cv
            }
        except: return {"confianca": 85.0}

    @staticmethod
    def _linear_trend_fallback(v_sem_pd):
        """Fallback estatístico quando o Nixtla não tem massa crítica."""
        y = v_sem_pd["y"].values
        X = np.arange(len(y)).reshape(-1, 1)
        slope = LinearRegression().fit(X, y).coef_[0]
        return int(y.mean() * 4), ("Alta" if slope > 0.05 else "Baixa" if slope < -0.05 else "Estável")

    @staticmethod
    def _heuristic_client_fallback(df):
        """Fallback para predição nominal baseado em média histórica simples."""
        return df.group_by("marca").agg([
            pl.len().alias("Qtd_Prevista"), 
            pl.col("faturamento").mean().alias("avg_price")
        ]).with_columns([
            pl.col("marca").alias("Cliente"), 
            (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est")
        ]).select(["Cliente", "Qtd_Prevista", "Valor_Est"]).sort("Valor_Est", descending=True).head(15)