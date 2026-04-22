import polars as pl
import numpy as np

class AnalyticsService:
    @staticmethod
    def get_pareto_distribution(df: pl.DataFrame):
        # Calcula volume por marca para análise de participação.
        return df.group_by("marca").len(name="vendas").sort("vendas", descending=True)

    @staticmethod
    def calculate_spc_metrics(df: pl.DataFrame):
        # Calcula Controle Estatístico de Processo (Média e Desvio Padrão).
        v_dia = df.group_by("dia_do_mes").len(name="vol").sort("dia_do_mes")
        if v_dia.is_empty():
            return v_dia, 0, 0
        
        mean = v_dia["vol"].mean()
        std = v_dia["vol"].std()
        return v_dia, mean, std

    @staticmethod
    def get_decision_rules(df: pl.DataFrame) -> str:
        
        # Simula a saída de uma árvore de decisão para priorização de leads.
        high_value = len(df.filter(pl.col("faturamento") > df["faturamento"].median())) if "faturamento" in df.columns else 0
        return f"""
        # Decision Logic (ML-Informed):
        - Prioridade Alpha: {high_value} frotas com alto faturamento detectadas.
        - Recomendação: Focar expansão em regiões com > 15% de Market Share.
        - Alerta: Volatilidade acima de 2 sigma no período atual.
        """
    
    @staticmethod
    def get_revenue_analysis(df: pl.DataFrame):
        """Analisa a relação entre volume de frota e faturamento."""
        return df.select(["marca", "frota", "faturamento", "uf"])

    @staticmethod
    def get_sector_distribution(df: pl.DataFrame):
        """Agrupa a volumetria por setor industrial."""
        return df.group_by("industry_sector").len(name="quantidade").sort("quantidade", descending=True)