import polars as pl
import logging
import os
from src.config import FCT_SALES_PATH, COLUMN_MAP
from typing import List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataEngine")

def load_processed_data() -> pl.DataFrame:
    """
    Carrega o Parquet real (fct_sales_history.parquet) e normaliza 
    as colunas detetadas na inspeção para o padrão do sistema.
    """
    if not os.path.exists(FCT_SALES_PATH):
        logger.error(f"Ficheiro não encontrado: {FCT_SALES_PATH}")
        return pl.DataFrame()

    try:
        df = pl.read_parquet(FCT_SALES_PATH)
        
        # 1. Normalização de Schema baseada na inspeção (image_564ce2.png)
        # Mapeamos os nomes reais do Parquet para os nomes que o resto do app usa
        schema_map = {
            "company_name": "marca",
            "hq_country": "uf",
            "order_value": "faturamento",
            "purchase_date": "data_faturamento"
        }
        
        # Aplicamos o mapeamento (respeitando o que já existe no COLUMN_MAP se necessário)
        df = df.rename({k: v for k, v in schema_map.items() if k in df.columns})
        
        # 2. Tratamento Crítico de Datas
        # O Nixtla precisa que 'data_faturamento' seja Date (sem horas)
        if "data_faturamento" in df.columns:
            df = df.with_columns([
                pl.col("data_faturamento").cast(pl.Date),
                pl.col("data_faturamento").dt.day().cast(pl.Int64).alias("dia_do_mes")
            ])
            
        return df
    except Exception as e:
        logger.error(f"Falha na carga dos dados reais: {e}")
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