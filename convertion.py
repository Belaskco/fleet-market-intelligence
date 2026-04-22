import pandas as pd
import numpy as np
import os

# Pega o diretório onde o convertion.py está (raiz da black_crow_intel)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Constrói os caminhos corretos entrando na pasta 'data'
input_file = os.path.join(BASE_DIR, 'data', 'MOCK_DATA.csv')
output_file = os.path.join(BASE_DIR, 'data', 'black_crow_unified.json')

def enrich_data():
    if not os.path.exists(input_file):
        print(f"❌ Erro: Arquivo não encontrado em: {input_file}")
        return

    print(f"🔄 Lendo: {input_file}")
    df = pd.read_csv(input_file)

    # Lógica de Fleet Size (Hard Mode)
    def calc_fleet(company):
        c = str(company)
        if any(x in c for x in ['Localiza', 'Movida', 'Américas']): return int(np.random.randint(5000, 50000))
        if any(x in c for x in ['JSL', 'Andreani', 'Servientrega', 'Ransa']): return int(np.random.randint(500, 3000))
        return int(np.random.randint(10, 500))

    df['fleet_size'] = df['company_name'].apply(calc_fleet)
    df['origin_source'] = np.random.choice(['WhatsApp', 'Legacy Access', 'Excel_Manual'], size=len(df))

    # Exportando
    df.to_json(output_file, orient='records', indent=4, force_ascii=False)
    print(f"✅ Sucesso! JSON gerado em: {output_file}")

if __name__ == "__main__":
    enrich_data()