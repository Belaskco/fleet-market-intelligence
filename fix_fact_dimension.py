import polars as pl
import numpy as np
import os
from datetime import datetime, timedelta

# Como o script está na raiz de black_crow_intel, o BASE_DIR é o local do script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos absolutos diretos
input_path = os.path.join(BASE_DIR, 'data', 'black_crow_intel.parquet')
output_path = os.path.join(BASE_DIR, 'data', 'gold_sales_fact.parquet')

def create_sales_fact():
    if not os.path.exists(input_path):
        print(f"❌ Erro: Arquivo não encontrado em: {input_path}")
        print(f"💡 Verifique se o arquivo está em: {os.path.join(BASE_DIR, 'data')}")
        return

    print(f"🔄 Processando {os.path.basename(input_path)}...")
    
    # Lendo o Parquet (Preservando as 10 colunas originais)
    df = pl.read_parquet(input_path)
    total_rows = df.height

    # --- Geração de Dados de Transação (O Caos que você pediu) ---
    start_date = datetime(2025, 1, 1)
    random_days = np.random.randint(0, 540, size=total_rows)
    purchase_dates = [start_date + timedelta(days=int(d)) for d in random_days]
    
    order_ids = [f"B2B-ORD-{i:05d}" for i in range(1, total_rows + 1)]
    
    # Valor do pedido (baseado no annual_revenue existente)
    order_values = (df["annual_revenue"] / 150) * np.random.uniform(0.1, 1.5, size=total_rows)

    # --- ADICIONANDO COLUNAS SEM REMOVER NADA ---
    # .with_columns é aditivo no Polars
    df_fact = df.with_columns([
        pl.Series("order_id", order_ids),
        pl.Series("purchase_date", purchase_dates),
        pl.Series("order_value", order_values),
        pl.Series("order_status", np.random.choice(['Completed', 'Pending', 'Cancelled'], 
                                                 size=total_rows, 
                                                 p=[0.85, 0.10, 0.05]))
    ])

    # Ordenação cronológica para análise de série temporal
    df_fact = df_fact.sort("purchase_date")

    # Salvando a versão Gold
    df_fact.write_parquet(output_path)
    
    print("="*60)
    print(f"✅ SUCESSO: Tabela de Fato (Gold) gerada!")
    print(f"📂 Arquivo: {output_path}")
    print(f"🧬 Colunas Totais: {df_fact.width} (Originais + Novas)")
    print("-" * 60)
    # Mostra o head com todas as colunas para você conferir
    print(df_fact.head(5))
    print("="*60)

if __name__ == "__main__":
    create_sales_fact()