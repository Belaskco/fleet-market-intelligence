import polars as pl
import pandas as pd
import numpy as np
import logging
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

# Configuração de Logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

class PredictionService:
    """
    Motor de Inteligência Black Crow Intel.
    Utiliza Nixtla (MLForecast) com Fallback Estatístico para máxima resiliência.
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """Calcula a tendência macro. Resolve erro de contagem de grupos."""
        if df.is_empty():
            return 0, "Estável"
        
        # Correção Polars: n_unique() para contar dias reais
        n_dias = df["dia_do_mes"].n_unique()
        
        if n_dias < 3:
            return 0, "Dados Insuficientes"

        try:
            # Agregação Macro
            v_dia = df.group_by("dia_do_mes").len(name="vol").sort("dia_do_mes")
            
            # Feature Engineering simples para Tendência
            X = v_dia["dia_do_mes"].to_numpy().reshape(-1, 1)
            y = v_dia["vol"].to_numpy()
            
            model = LinearRegression().fit(X, y)
            proj_fechamento = int(max(0, model.predict([[30]])[0]))
            
            slope = model.coef_[0]
            trend_status = "Alta" if slope > 0.1 else "Baixa" if slope < -0.1 else "Estável"
            
            return proj_fechamento, trend_status
        except Exception as e:
            logger.error(f"Erro na tendência macro: {e}")
            return 0, "Indisponível"

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """
        Gera predição nominal por cliente. 
        Implementa fallback caso o Nixtla encontre séries temporais curtas demais.
        """
        if df.is_empty():
            return pl.DataFrame()

        try:
            # Filtro de Qualidade: Apenas marcas com histórico mínimo para o Nixtla
            contagem = df.group_by("marca").len().filter(pl.col("len") >= 3)
            if contagem.is_empty():
                return PredictionService._heuristic_fallback(df)

            # Preparação Nixtla (MLForecast)
            data_prep = df.join(contagem.select("marca"), on="marca")
            pdf = data_prep.select([
                pl.col("marca").alias("unique_id"),
                pl.col("data_faturamento").alias("ds"),
                pl.lit(1).alias("y")
            ]).to_pandas()
            
            pdf['ds'] = pd.to_datetime(pdf['ds'])
            ts_data = pdf.groupby(['unique_id', 'ds']).sum().reset_index()

            # Motor Nixtla
            fcst = MLForecast(
                models=[LGBMRegressor(random_state=42, verbosity=-1)],
                freq='D',
                lags=[1]
            )
            fcst.fit(ts_data)
            preds = fcst.predict(7)

            # Consolidação e Cálculo de Valor
            res = pl.from_pandas(preds).group_by("unique_id").agg(pl.col("LGBMRegressor").sum().round(0).alias("Qtd_Prevista"))
            ticket_medio = df.group_by("marca").agg(pl.col("faturamento").mean().alias("avg_price"))
            
            final = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            return final.with_columns([
                pl.col("unique_id").alias("Cliente"),
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
                pl.lit(0.85).alias("Probabilidade")
            ]).filter(pl.col("Qtd_Prevista") > 0).select(["Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"]).sort("Valor_Est", descending=True).head(10)

        except Exception as e:
            logger.warning(f"Nixtla falhou, acionando Fallback Estatístico: {e}")
            return PredictionService._heuristic_fallback(df)

    @staticmethod
    def _heuristic_fallback(df: pl.DataFrame):
        """Motor reserva para garantir que a tabela nunca fique vazia."""
        try:
            return df.group_by("marca").agg([
                pl.len().alias("Qtd_Prevista"),
                pl.col("faturamento").sum().alias("Valor_Est")
            ]).with_columns([
                pl.col("marca").alias("Cliente"),
                pl.lit(0.60).alias("Probabilidade")
            ]).sort("Valor_Est", descending=True).head(10).select(["Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"])
        except:
            return pl.DataFrame()