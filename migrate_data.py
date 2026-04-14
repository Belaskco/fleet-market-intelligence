import polars as pl
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataMigration")
load_dotenv()

def migrate_raw_data():
    """
    Converte o JSON bruto para Parquet, preservando a granularidade
    para permitir filtros dinâmicos de Marca e UF.
    """
    source = "data/MOCK_DATA.json"
    target = "data/market_data.parquet"
    
    try:
        logger.info("Lendo JSON original...")
        df = pl.read_json(source)

        # 1. Tratamento de Tipos (O Rigor do Mestre)
        # Convertemos date e garantimos que model_year seja inteiro
        df = df.with_columns([
            pl.col("date").str.to_date("%Y-%m-%d").alias("data"),
            pl.col("model_year").cast(pl.Int32),
            pl.col("date").str.to_date("%Y-%m-%d").dt.day().alias("dia_do_mes")
        ])

        # 2. Seleção de colunas necessárias
        df = df.select(["data", "dia_do_mes", "marca", "modelo", "model_year", "uf"])

        # 3. Persistência em Parquet ZSTD
        df.write_parquet(target, compression="zstd")
        logger.info(f"Sucesso! {df.height} registros migrados para {target}")

    except Exception as e:
        logger.error(f"Falha na migração: {str(e)}")

if __name__ == "__main__":
    migrate_raw_data()