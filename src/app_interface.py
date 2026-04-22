import plotly.express as px
import streamlit as st
import polars as pl
import sys
import os

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_dashboard():
    """
    Orquestra a interface focado em Sales Intelligence e visões preditivas
    para suporte direto ao time comercial.
    """
    
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.markdown(f"""
        <style>
        .stMetric {{ 
            background-color: #161b22; 
            border: 1px solid #30363d; 
            padding: 15px; 
            border-radius: 10px; 
        }}
        div[data-testid="stExpander"] {{
            border: 1px solid #30363d;
            background-color: #0d1117;
        }}
        div[data-testid="stInfo"] {{
            background-color: #0d1117;
            border: 1px solid #30363d;
            color: {THEME_COLOR};
        }}
        </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def get_cached_data():
        return load_processed_data()

    df_raw = get_cached_data()

    if df_raw.is_empty():
        st.error("❌ Framework Error: Data source (fct_sales_history.parquet) not found.")
        st.stop()

    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=80)
        st.title("Market Control")
        st.divider()
        
        with st.popover("⚙️ Configurar Segmentação"):
            marcas_disp = sorted(df_raw["marca"].unique().to_list())
            sel_marcas = st.multiselect("Foco em Marcas:", marcas_disp, default=marcas_disp[:5])
            
            ufs_disp = sorted(df_raw["uf"].unique().to_list())
            sel_ufs = st.multiselect("Regiões Estratégicas:", ufs_disp, default=ufs_disp)
            
            setores_disp = sorted(df_raw["industry_sector"].unique().to_list())
            sel_setores = st.multiselect("Setores Industriais:", setores_disp, default=setores_disp)
        
        st.divider()
        sel_days = st.slider("Janela de Observação (Dias):", 1, 31, (1, 31))
        st.caption("Black Crow Intelligence | v1.3.0 Platinum")

    df_filt = apply_business_filters(df_raw, sel_marcas, sel_ufs, sel_days)
    if not sel_setores == setores_disp:
        df_filt = df_filt.filter(pl.col("industry_sector").is_in(sel_setores))

    if not df_filt.is_empty():
        st.title(f"{APP_TITLE}")
        
        # --- BLOCO 1: OPERATIONAL KPIs ---
        k1, k2, k3, k4 = st.columns(4)
        dist_data = AnalyticsService.get_pareto_distribution(df_filt)
        
        k1.metric("Volume Total", f"{len(df_filt):,}")
        k2.metric("Market Leader", dist_data["marca"][0] if not dist_data.is_empty() else "N/A")
        k3.metric("Média Diária", f"{len(df_filt)/31:.1f}")
        k4.metric("Market Breadth (UFs)", len(df_filt["uf"].unique()))
        
        st.divider()

        # --- BLOCO 2: SALES RADAR (PREDITIVO MICRO) ---
        st.subheader("🎯 Sales Radar: Próximas Compras Prováveis")
        # Aqui trazemos a visão por cliente para o time comercial
        opportunities = PredictionService.get_sales_opportunities(df_filt)
        
        col_list, col_chart = st.columns([2, 1])
        
        with col_list:
            st.write("Clientes com maior propensão de renovação/expansão de frota:")
            st.dataframe(
                opportunities, 
                column_config={
                    "marca": "Cliente/Marca",
                    "total_revenue": st.column_config.NumberColumn("Faturamento Est. ($)", format="$ %d"),
                    "avg_fleet_size": "Média de Frota",
                    "propensity_score": st.column_config.TextColumn("Score de Propensão")
                },
                hide_index=True,
                use_container_width=True
            )

        with col_chart:
            proj_vol, trend = PredictionService.get_market_trend(df_filt)
            st.metric("Forecast Fechamento Mês", f"{proj_vol} un.", delta=trend)
            st.info(f"O mercado apresenta tendência de **{trend}**. Foque nos clientes com propensão 'Alta'.")

        st.divider()

        # --- BLOCO 3: ANOMALIES & STABILITY ---
        st.subheader("📊 Health Check & Anomalias")
        anomalies = PredictionService.identify_anomalies(df_filt)
        if not anomalies.is_empty():
            st.warning(f"Atenção: Detectadas {len(anomalies)} oscilações atípicas no volume de vendas.")
        
        v_dia, m, s = AnalyticsService.calculate_spc_metrics(df_filt)
        fig_spc = px.line(v_dia, x='dia_do_mes', y='vol', markers=True, template="plotly_dark", height=300)
        st.plotly_chart(fig_spc, use_container_width=True)

        # --- BLOCO 4: ESTRATÉGIA ---
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏆 Market Share")
            fig_p = px.bar(dist_data, x='vendas', y='marca', orientation='h', color='vendas', template="plotly_dark")
            st.plotly_chart(fig_p, use_container_width=True)
        with c2:
            st.subheader("🏢 Penetração Setorial")
            sec_df = df_filt.group_by("industry_sector").len(name="quantidade").sort("quantidade", descending=True)
            fig_pie = px.pie(sec_df, values="quantidade", names="industry_sector", hole=0.4, template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()
        st.subheader("🧠 Drivers de Decisão (Logic Engine)")
        st.code(AnalyticsService.get_decision_rules(df_filt), language='python')

    else:
        st.warning("⚠️ Filtros restritivos demais. Não há dados para exibir.")