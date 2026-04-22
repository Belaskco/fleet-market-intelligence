import os

# Caminhos de Diretório
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FCT_SALES_PATH = os.path.join(DATA_DIR, "fct_sales_history.parquet")

# Mapeamento de Colunas
COLUMN_MAP = {
    "company_name": "marca",
    "hq_country": "uf",
    "purchase_date": "data_transacao",
    "annual_revenue": "faturamento",
    "fleet_size": "frota"
}

# Configurações Visuais
THEME_COLOR = "#58a6ff"
APP_TITLE = "Market Intelligence Framework"