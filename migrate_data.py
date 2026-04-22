import pandas as pd
import os

# Definição de caminhos baseada na sua estrutura de pastas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(BASE_DIR, 'data', 'black_crow_unified.json')
parquet_path = os.path.join(BASE_DIR, 'data', 'black_crow_intel.parquet')

def migrate_to_parquet():
    if not os.path.exists(json_path):
        print(f"❌ Erro: JSON não encontrado em {json_path}")
        return

    print(f"🔄 Migrando JSON para Parquet...")
    
    # Lendo o JSON
    df = pd.read_json(json_path)

    # Sanity Check: Garantindo que faturamento e frota sejam numéricos
    # Útil para evitar erros em cálculos no Power BI
    df['annual_revenue'] = pd.to_numeric(df['annual_revenue'], errors='coerce')
    df['fleet_size'] = pd.to_numeric(df['fleet_size'], errors='coerce')

    # Salvando em Parquet com compressão snappy (padrão ouro do Databricks)
    df.to_parquet(parquet_path, compression='snappy', index=False)
    
    print(f"✅ Sucesso! Camada Silver gerada em: {parquet_path}")
    print(f"📊 Linhas processadas: {len(df)}")

if __name__ == "__main__":
    migrate_to_parquet()