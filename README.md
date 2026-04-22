# Black Crow Intelligence | Fleet Market Intelligence Framework

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Polars](https://img.shields.io/badge/Engine-Polars-orange.svg)
![Status](https://img.shields.io/badge/Version-1.3.0--Platinum-brightgreen.svg)

> "Ceaseless watcher of anomalies. Tireless weaver of patterns." 

**Black Crow Intelligence** não é apenas mais um dashboard bonitinho feito em Power BI por alguém que não sabe o que é um desvio padrão. É um ecossistema preditivo de alta performance para o mercado de frotas (B2B), construído para quem prefere fatos estatísticos a palpites de corredor.

## 🚀 Por que Polars e não o "velho" Pandas?

Porque o tempo é curto e o processamento em Rust é soberano. Enquanto o Pandas ainda está tentando entender o que é um arquivo Parquet pesado, o **Polars** já entregou o resultado, lavou a louça e assistiu a um episódio de *Evangelion*. Performance aqui não é luxo, é pré-requisito.

## 🧠 Camadas de Inteligência

### 1. Core Analytics (O "Health Check" da Realidade)
* **Pareto Market Share:** Para você saber quem manda no parquinho.
* **SPC (Statistical Process Control):** Implementação de Z-Score para detectar anomalias. Se o gráfico deu um pulo, o Corvo avisa. Se for ruído estatístico, ele ignora (diferente do seu chefe).

### 2. Predictive Engine (O Oráculo Nixtla-ish)
* **Sales Radar:** Lead Scoring que separa o "provável comprador" do "curioso que só quer orçamento". Municiamos o time de vendas com alvos reais, não com esperança.
* **Trend Forecasting:** Projeções de fechamento de mês inspiradas na arquitetura da Nixtla. Prevemos o futuro para que você não precise de uma bola de cristal.

### 3. Logic Engine (A voz da consciência)
* Tradução de métricas complexas em **Drivers de Decisão**. O sistema mastiga o dado e cospe a ação. Menos reunião, mais execução.

## 🛠️ Stack Tecnológica (O Cinto de Utilidades)

* **Engine:** Polars & PyArrow (Data Processing on Steroids)
* **Brain:** Scikit-Learn (O básico bem feito vence o hype)
* **HUD:** Streamlit (Minority Report feelings, mas sem as luvas bregas)
* **Style:** Plotly Express (Porque gráficos feios deveriam ser crime)

## 📂 Estrutura do Projeto (Organizado por quem tem TOC)

```text
├── data/               # Onde os dados moram (Parquet/CSV)
├── src/                # Onde a mágica (e o suor) acontece
│   ├── analytics_service.py   # Estatística pura, sem filtros
│   ├── prediction_service.py  # Adivinhando o amanhã com math
│   ├── data_engine.py         # Ingestão em alta velocidade
│   └── app_interface.py       # A cara do Corvo
├── app.py              # O botão de "Ligar" (Root)
└── requirements.txt    # O que você precisa para não passar passar raiva
```

## ⚙️ Como rodar essa beleza?

```Bash
git clone (https://github.com/Belaskco/fleet-market-intelligence.git)
cd fleet-market-intelligence
```

Instalar o que importa:
```Bash
pip install -r requirements.txt
```

Rodar o Streamlit:
```Bash
streamlit run app.py
```

---
*Desenvolvido com café, ódio direcionado e precisão cirúrgica por Cássio Ferreira de Andrade.*