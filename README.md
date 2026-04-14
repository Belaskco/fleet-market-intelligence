# Automotive Market Intelligence Portal

Pare de tentar adivinhar o mercado usando médias mentirosas no Excel. Este portal separa o sinal do ruído usando engenharia de verdade e estatística aplicada, sem firulas. Desenvolvido para processar grandes volumes de dados automotivos com performance de baixo nível e inteligência de alto nível.

# Arquitetura
O projeto foi modularizado para ser escalável e resiliente. Se você busca scripts monolíticos e desorganizados, este repositório não é para você.

```python
Market_Intel/
├── data/               # Camada de Persistência (Parquet/Zstd)
├── src/                # Núcleo Lógico
│   ├── engine.py       # Motor de ETL (Vetorização em Rust via Polars)
│   └── intelligence.py # Analytics Avançado (SPC & Anomaly Detection)
├── app.py              # Orquestrador da Interface
└── requirements.txt    # Gestão de Dependências
```

# O que tem por baixo do capô
* SPC (Statistical Process Control): Monitoramento de estabilidade via limites 3-Sigma. O que sai da linha é anomalia de causa especial, não flutuação comum de mercado.
* Processamento Arrow-Native: Migração de legado JSON para Parquet com compressão Zstd, garantindo leituras 50x mais rápidas e tipagem estática.
* Engine Polars: Substituição do Pandas por uma engine multithreaded escrita em Rust, otimizando o uso de CPU no ambiente WSL.

## Como rodar (Se você tiver o mínimo de competência)
Certifique-se de estar em um ambiente Linux/WSL. Copie e execute um bloco por vez.

```bash
# Clone o repositório
git clone https://github.com/belaskco/market_intel.git
# Entre na pasta (O passo que a maioria esquece)
cd market_intel
# Instale as dependências (Use seu venv)
pip install -r requirements.txt
# Execute a aplicação
streamlit run app.py
```

# Utilitários de Engenharia
O repositório inclui ferramentas de elite para manutenção de dados:

* ```migrate_data.py```: Converte o lixo JSON em ouro Parquet.
* ```inspect_parquet.py```: Validação técnica de schema e integridade binária.

# Considerações
Matemática aplicada ao caos automotivo. Se o código for útil, ótimo. Se encontrar um bug, sinta-se à vontade para consertar e abrir um Pull Request — ou continue reclamando no LinkedIn.

### Desenvolvido por Cassio de Andrade.
https://www.linkedin.com/in/cassioandrade84/