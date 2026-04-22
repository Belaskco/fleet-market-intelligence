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
        Simula o comportamento de modelos de tendência de curto prazo.
        """
        try:
            # Agregação volumétrica diária para input do modelo
            v_dia = df.group_by("dia_do_mes").len(name="vol").sort("dia_do_mes")
            
            if len(v_dia) < 5:
                logger.warning("Volume de dados insuficiente para projeção linear confiável.")
                return None, "Dados Insuficientes"
            
            # Preparação de Features (X) e Target (y)
            X = v_dia["dia_do_mes"].to_numpy().reshape(-1, 1)
            y = v_dia["vol"].to_numpy()
            
            # Fitting do modelo de tendência
            model = LinearRegression().fit(X, y)
            
            # Projeção para o fechamento do ciclo mensal (Dia 30)
            proj_fechamento = model.predict([[30]])[0]
            
            # Cálculo de inclinação para definição de status de tendência
            slope = model.coef_[0]
            trend_status = "Alta" if slope > 0.1 else "Baixa" if slope < -0.1 else "Estável"
            
            return int(max(0, proj_fechamento)), trend_status
            
        except Exception as e:
            logger.error(f"Erro no forecasting de mercado: {e}")
            return None, "Erro"

    @staticmethod
    def identify_anomalies(df: pl.DataFrame):
        """
        Aplica técnicas de detecção de outliers baseadas em desvio padrão (Z-Score).
        Identifica ruídos que podem distorcer a visão de negócios.
        """
        v_dia = df.group_by("dia_do_mes").len(name="vol")
        
        if v_dia.is_empty():
            return pl.DataFrame()
            
        mean = v_dia["vol"].mean()
        std = v_dia["vol"].std()
        
        # Filtro de anomalia: 2 sigmas de distância da média (95% de confiança)
        anomalies = v_dia.filter(
            (pl.col("vol") > mean + 2*std) | (pl.col("vol") < mean - 2*std)
        ).sort("dia_do_mes")
        
        return anomalies

    @staticmethod
    def get_sales_opportunities(df: pl.DataFrame):
        """
        Algoritmo de Sales Radar: Identifica 'compras quase certas'.
        Baseia-se em frequência de compra, faturamento total e média de frota.
        """
        if df.is_empty():
            return pl.DataFrame()

        # Agregação por cliente (Marca) para análise de comportamento histórico
        opportunities = df.group_by("marca").agg([
            pl.col("faturamento").sum().alias("total_revenue"),
            pl.col("frota").mean().alias("avg_fleet_size"),
            pl.len().alias("purchase_frequency")
        ])

        # Scoring de Propensão:
        # 1. 'Alta Probabilidade': Clientes recorrentes (> 2 compras no período).
        # 2. 'Monitoramento': Clientes esporádicos com faturamento relevante.
        opportunities = opportunities.with_columns(
            pl.when(pl.col("purchase_frequency") >= 2)
            .then(pl.lit("Alta Probabilidade 🚀"))
            .otherwise(pl.lit("Monitoramento Ativo 🧐"))
            .alias("propensity_score")
        )

        # Ordenação por valor estratégico (Revenue) para priorização comercial
        return opportunities.sort("total_revenue", descending=True).head(10)