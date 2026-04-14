import polars as pl
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("DataEngine")

def load_data() -> pl.DataFrame:
    """Carrega o arquivo Parquet usando Polars."""
    path = os.getenv("MARKET_DATA_PATH", "data/market_data.parquet")
    if not os.path.exists(path):
        logger.error("Data Lake local não encontrado.")
        return pl.DataFrame()
    return pl.read_parquet(path)

def apply_filters(df: pl.DataFrame, marcas: List[str], ufs: List[str], dias: tuple) -> pl.DataFrame:
    """
    Aplica filtros estratégicos usando o motor de Rust do Polars.
    """
    if df.is_empty():
        return df

    return df.filter(
        (pl.col("marca").is_in(marcas)) &
        (pl.col("uf").is_in(ufs)) &
        (pl.col("dia_do_mes").is_between(dias[0], dias[1]))
    )