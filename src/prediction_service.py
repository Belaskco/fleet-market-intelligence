import polars as pl
import numpy as np
import logging
from sklearn.linear_model import LinearRegression

# Configuração de logger para monitoramento de saúde do modelo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

class PredictionService:
    """
    Motor de inteligência preditiva do framework Black Crow.
    Implementa lógicas de Forecasting (Macro) e Sales Propensity (Micro).
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """
        Executa uma análise de tendência baseada em regressão linear.
        """
        try:
            v_dia = df.group_by("dia_do_mes").len(name="vol").sort("dia_do_mes")
            
            if len(v_dia) < 5:
                return None, "Dados Insuficientes"
            
            X = v_dia["dia_do_mes"].to_numpy().reshape(-1, 1)
            y = v_dia["vol"].to_numpy()
            
            model = LinearRegression().fit(X, y)
            proj_fechamento = model.predict([[30]])[0]
            slope = model.coef_[0]
            
            trend_status = "Alta" if slope > 0.1 else "Baixa" if slope < -0.1 else "Estável"
            return int(max(0, proj_fechamento)), trend_status
            
        except Exception as e:
            logger.error(f"Erro no forecasting: {e}")
            return None, "Erro"

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """
        MÉTODO INTEGRADO COM A INTERFACE:
        Retorna a tabela de predições nominais formatada para o Bloco 3.
        """
        if df.is_empty():
            return pl.DataFrame()

        try:
            # Agregação por cliente (Marca)
            # Nota: O Nixtla seria implementado aqui. No momento, usamos 
            # uma heurística de propensão baseada em frequência e volume.
            prediction = df.group_by("marca").agg([
                pl.len().alias("Qtd_Prevista"),
                pl.col("faturamento").sum().alias("Valor_Est")
            ])

            # Cálculo de Probabilidade (Confiança) baseada na constância
            # Aqui simulamos o score que o Nixtla entregaria
            prediction = prediction.with_columns(
                pl.col("Qtd_Prevista").map_elements(lambda x: min(x * 0.2, 0.95), return_dtype=pl.Float64).alias("Probabilidade"),
                pl.col("marca").alias("Cliente")
            )

            # Seleciona apenas as colunas que a Interface v2.2.5 espera
            return prediction.select([
                "Cliente", 
                "Qtd_Prevista", 
                "Valor_Est", 
                "Probabilidade"
            ]).sort("Valor_Est", descending=True).head(10)

        except Exception as e:
            logger.error(f"Erro ao gerar predições nominais: {e}")
            return pl.DataFrame()

    @staticmethod
    def identify_anomalies(df: pl.DataFrame):
        """Detecta ruídos estatísticos via Z-Score."""
        v_dia = df.group_by("dia_do_mes").len(name="vol")
        if v_dia.is_empty(): return pl.DataFrame()
        
        mean, std = v_dia["vol"].mean(), v_dia["vol"].std()
        return v_dia.filter((pl.col("vol") > mean + 2*std) | (pl.col("vol") < mean - 2*std))