import polars as pl
import logging
import os
from src.config import FCT_SALES_PATH, COLUMN_MAP
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataEngine")

def load_processed_data() -> pl.DataFrame:
    """
    Carrega o Parquet e reconstrói a coluna 'data_faturamento'.
    Aqui é onde a mágica da data acontece.
    """
    if not os.path.exists(FCT_SALES_PATH):
        logger.error(f"Arquivo não encontrado: {FCT_SALES_PATH}")
        return pl.DataFrame()

    try:
        df = pl.read_parquet(FCT_SALES_PATH)
        
        # 1. Normalização de Schema (Mapeia nomes do Parquet para o sistema)
        existing_map = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
        df = df.rename(existing_map)
        
        # 2. COMO CRIAR A COLUNA 'data_faturamento':
        # Se você tem colunas chamadas 'ano', 'mes', 'dia':
        if all(col in df.columns for col in ["ano", "mes", "dia"]):
            df = df.with_columns(
                pl.date(pl.col("ano"), pl.col("mes"), pl.col("dia")).alias("data_faturamento")
            )
        # Se você tem uma coluna 'DATA' que é string ou timestamp:
        elif "DATA" in df.columns:
            df = df.with_columns(
                pl.col("DATA").cast(pl.Date).alias("data_faturamento")
            )
        # Fallback: Se vier como 'data_transacao'
        elif "data_transacao" in df.columns:
            df = df.with_columns(
                pl.col("data_transacao").cast(pl.Date).alias("data_faturamento")
            )

        # 3. Engenharia de Atributos para os filtros da Sidebar
        if "data_faturamento" in df.columns:
            df = df.with_columns([
                pl.col("data_faturamento").dt.day().cast(pl.Int64).alias("dia_do_mes")
            ])
            
        return df
    except Exception as e:
        logger.error(f"Falha na carga ou processamento de data: {e}")
        return pl.DataFrame()

def apply_business_filters(df: pl.DataFrame, marcas: List[str], ufs: List[str], dias: tuple) -> pl.DataFrame:
    """Aplica filtros operacionais usando a Rust Engine do Polars."""
    if df.is_empty():
        return df
    
    return df.filter(
        (pl.col("marca").is_in(marcas)) &
        (pl.col("uf").is_in(ufs)) &
        (pl.col("dia_do_mes").is_between(dias[0], dias[1]))
    )