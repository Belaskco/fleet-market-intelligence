import polars as pl
import logging
import os
from src.config import FCT_SALES_PATH, COLUMN_MAP
from typing import List


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataEngine")

def load_processed_data() -> pl.DataFrame:
    
    # Carrega e normaliza os dados para o padrão do sistema.
    if not os.path.exists(FCT_SALES_PATH):
        logger.error(f"Arquivo não encontrado: {FCT_SALES_PATH}")
        return pl.DataFrame()

    try:
        df = pl.read_parquet(FCT_SALES_PATH)
        
        # Normalização de Schema via Config
        existing_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
        df = df.rename(existing_map)
        
        # Engenharia de Atributos Temporal
        if "data_transacao" in df.columns:
            df = df.with_columns(
                pl.col("data_transacao").dt.day().cast(pl.Int64).alias("dia_do_mes")
            )
        return df
    except Exception as e:
        logger.error(f"Falha na carga: {e}")
        return pl.DataFrame()

def apply_business_filters(df: pl.DataFrame, marcas: List[str], ufs: List[str], dias: tuple) -> pl.DataFrame:
    
    # Aplica filtros operacionais usando Rust Engine do Polars.
    if df.is_empty():
        return df
    
    return df.filter(
        (pl.col("marca").is_in(marcas)) &
        (pl.col("uf").is_in(ufs)) &
        (pl.col("dia_do_mes").is_between(dias[0], dias[1]))
    )