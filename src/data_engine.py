import polars as pl
import logging
import os
from src.config import FCT_SALES_PATH, COLUMN_MAP
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataEngine")

def load_processed_data() -> pl.DataFrame:
    """Carrega o Parquet real e reconstrói a coluna de data para o Nixtla."""
    if not os.path.exists(FCT_SALES_PATH):
        logger.error(f"Arquivo não encontrado: {FCT_SALES_PATH}")
        return pl.DataFrame()

    try:
        df = pl.read_parquet(FCT_SALES_PATH)
        
        # 1. Normalização de Schema (Mapeia company_name -> marca, etc.)
        existing_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
        df = df.rename(existing_map)
        
        # 2. Reconstrução da Coluna de Data (Chave para Nixtla e D-1)
        # Assumindo que o Parquet tem colunas de ano, mes, dia ou uma coluna 'DATA'
        # Vamos garantir que exista uma coluna 'data_faturamento' do tipo Date
        if "ano" in df.columns and "mes" in df.columns and "dia" in df.columns:
            df = df.with_columns(
                pl.date(pl.col("ano"), pl.col("mes"), pl.col("dia")).alias("data_faturamento")
            )
        elif "DATA" in df.columns:
            df = df.with_columns(pl.col("DATA").cast(pl.Date).alias("data_faturamento"))
        
        # 3. Engenharia de Atributos Temporal para Filtros
        if "data_faturamento" in df.columns:
            df = df.with_columns(
                pl.col("data_faturamento").dt.day().cast(pl.Int64).alias("dia_do_mes")
            )
        
        return df
    except Exception as e:
        logger.error(f"Falha na carga: {e}")
        return pl.DataFrame()

def apply_business_filters(df: pl.DataFrame, marcas: List[str], ufs: List[str], dias: tuple) -> pl.DataFrame:
    """Aplica filtros operacionais usando a engine Rust do Polars."""
    if df.is_empty():
        return df
    
    return df.filter(
        (pl.col("marca").is_in(marcas)) &
        (pl.col("uf").is_in(ufs)) & # Aqui 'uf' mapeia para hq_country/território
        (pl.col("dia_do_mes").is_between(dias[0], dias[1]))
    )