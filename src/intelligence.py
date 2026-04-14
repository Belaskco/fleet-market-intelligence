import polars as pl
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from typing import Tuple, Dict, Any

class MarketIntelligence:
    """
    Framework de Inteligência Black Crow.
    Concentra toda a lógica matemática e estatística do portal.
    """
    
    @staticmethod
    def get_pareto_data(df: pl.DataFrame) -> pd.DataFrame:
        """Calcula ranking de marcas e % acumulada."""
        total = df.height
        pareto = (
            df.group_by("marca")
            .agg(pl.len().alias("vendas"))
            .sort("vendas", descending=True)
            .with_columns([
                (pl.col("vendas").cum_sum() / total * 100).alias("acc_perc")
            ])
        )
        return pareto.to_pandas()

    @staticmethod
    def calculate_spc(df: pl.DataFrame) -> Tuple[pd.DataFrame, float, float]:
        """Calcula limites 3-Sigma (SPC)."""
        v_dia = df.group_by("dia_do_mes").agg(pl.len().alias("vol")).sort("dia_do_mes")
        mean = v_dia["vol"].mean()
        std = v_dia["vol"].std()
        return v_dia.to_pandas(), mean, std

    @staticmethod
    def prepare_boxplot_data(df: pl.DataFrame, selected_marcas: list, days_range: tuple) -> pd.DataFrame:
        """Prepara dados para o Boxplot garantindo zeros em dias sem vendas."""
        all_days = list(range(days_range[0], days_range[1] + 1))
        v_m_d = df.group_by(["marca", "dia_do_mes"]).len().rename({"len": "vendas"})
        
        pdf = v_m_d.to_pandas()
        idx = pd.MultiIndex.from_product([selected_marcas, all_days], names=['marca', 'dia_do_mes'])
        full_df = pd.DataFrame(index=idx).reset_index()
        
        return full_df.merge(pdf, on=['marca', 'dia_do_mes'], how='left').fillna(0)

    @staticmethod
    def get_decision_logic(df: pl.DataFrame) -> str:
        """Modelagem via Decision Tree para entender drivers de compra."""
        if df.height < 20:
            return "Volume de dados insuficiente para inferência."
        
        pdf = df.to_pandas()
        X = pd.get_dummies(pdf[['dia_do_mes', 'uf']])
        y = pdf['marca']
        
        clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5)
        clf.fit(X, y)
        return export_text(clf, feature_names=list(X.columns))