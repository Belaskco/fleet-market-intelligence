import polars as pl
import logging
import os
from src.config import FCT_SALES_PATH
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataEngine")

def load_processed_data() -> pl.DataFrame:
    """
    Carrega e normaliza os dados. 
    Centraliza o mapeamento de colunas para simplificar a visualização.
    """
    if not os.path.exists(FCT_SALES_PATH):
        logger.error(f"Ficheiro não encontrado: {FCT_SALES_PATH}")
        return pl.DataFrame()

    try:
        df = pl.read_parquet(FCT_SALES_PATH)
        
        # Mapeamento Universal: O dashboard só conhece estes 4 nomes
        schema_map = {
            "company_name": "marca",
            "marca": "marca",
            "hq_country": "uf",
            "uf": "uf",
            "order_value": "faturamento",
            "faturamento": "faturamento",
            "purchase_date": "data_faturamento",
            "data_faturamento": "data_faturamento"
        }
        
        # Renomeia apenas as colunas que existem no ficheiro
        current_cols = df.columns
        rename_dict = {k: v for k, v in schema_map.items() if k in current_cols and k != v}
        df = df.rename(rename_dict)
        
        # Normalização de Tipos
        if "data_faturamento" in df.columns:
            df = df.with_columns([
                pl.col("data_faturamento").cast(pl.Date),
                pl.col("data_faturamento").dt.day().cast(pl.Int64).alias("dia_do_mes")
            ])
            
        return df
    except Exception as e:
        logger.error(f"Falha na carga/normalização: {e}")
        return pl.DataFrame()

def apply_business_filters(df: pl.DataFrame, marcas: List[str], ufs: List[str], dias: tuple) -> pl.DataFrame:
    """Aplica filtros usando a performance da Rust Engine do Polars."""
    if df.is_empty():
        return df
    
    return df.filter(
        (pl.col("marca").is_in(marcas)) &
        (pl.col("uf").is_in(ufs)) &
        (pl.col("dia_do_mes").is_between(dias[0], dias[1]))
    )