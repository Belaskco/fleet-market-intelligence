import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from src.engine import load_data, apply_filters
from src.intelligence import MarketIntelligence

# 1. Configuração e Estética
st.set_page_config(page_title="Black Crow Intelligence", layout="wide", page_icon="🐦‍⬛")

st.markdown("""
    <style>
    .stMetric { background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# 2. Carga com Cache (Performance Sênior)
@st.cache_data
def get_data():
    return load_data()

df_raw = get_data()

if df_raw.is_empty():
    st.error("Erro Crítico: Data Lake (Parquet) não detectado.")
    st.info("Execute 'python migrate_data.py' no terminal para inicializar os dados.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=100)
    st.title("Strategic Filters")
    st.divider()
    
    marcas = sorted(df_raw["marca"].unique().to_list())
    selected_marcas = st.multiselect("Marcas em Análise:", marcas, default=marcas[:5])
    
    ufs = sorted(df_raw["uf"].unique().to_list())
    selected_ufs = st.multiselect("Filtro Regional (UF):", ufs, default=ufs)
    
    selected_days = st.slider("Período de Análise:", 1, 31, (1, 31))

# 3. Orquestração
df_filt = apply_filters(df_raw, selected_marcas, selected_ufs, selected_days)

if not df_filt.is_empty():
    st.title("🐦‍⬛ Black Crow: Automotive Market Intelligence")
    st.markdown(f"### Performance Analytics | {len(df_filt):,} Registros Processados")
    
    # --- KPIs ---
    k1, k2, k3, k4 = st.columns(4)
    lider = df_filt.group_by("marca").len().sort("len", descending=True).head(1)
    
    k1.metric("Total Volume", f"{len(df_filt):,}")
    k2.metric("Market Leader", lider["marca"][0])
    k3.metric("Daily Avg", f"{len(df_filt)/31:.1f}")
    k4.metric("Active Regions", len(df_filt["uf"].unique()))
    st.divider()

    # --- VISUALIZAÇÕES ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Market Share (Pareto)")
        rank_df = MarketIntelligence.get_pareto_data(df_filt)
        fig_rank = px.bar(rank_df, x='vendas', y='marca', orientation='h', color='vendas', color_continuous_scale='Viridis')
        fig_rank.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_dark")
        st.plotly_chart(fig_rank, use_container_width=True)

    with col2:
        st.subheader("📈 Controle Estatístico (SPC)")
        v_dia, mean, std = MarketIntelligence.calculate_spc(df_filt)
        fig_spc = px.line(v_dia, x='dia_do_mes', y='vol', markers=True)
        for val, color, label in zip([mean, mean+3*std, mean-3*std], ["#58a6ff", "#f85149", "#f85149"], ["Mean", "UCL", "LCL"]):
            fig_spc.add_hline(y=val, line_dash="dot", line_color=color, annotation_text=label)
        fig_spc.update_layout(template="plotly_dark")
        st.plotly_chart(fig_spc, use_container_width=True)

    col3, col4 = st.columns([1, 1.5])
    with col3:
        st.subheader("📦 Consistência de Vendas")
        df_box = MarketIntelligence.prepare_boxplot_data(df_filt, selected_marcas, selected_days)
        st.plotly_chart(px.box(df_box, x='marca', y='vendas', color='marca', template="plotly_dark"), use_container_width=True)

    with col4:
        st.subheader("🧠 Drivers de Decisão (ML Tree)")
        rules = MarketIntelligence.get_decision_logic(df_filt)
        st.code(rules)
else:
    st.warning("Sem dados para os filtros selecionados.")