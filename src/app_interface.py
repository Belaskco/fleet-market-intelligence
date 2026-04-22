import plotly.express as px
import streamlit as st
import polars as pl
import sys
import os

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# Path setup para consistência de módulos em deploy
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_dashboard():
    """
    Orquestrador da interface. Foco em UX intuitiva: filtros pré-carregados 
    para análise imediata do market share consolidado.
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Override de CSS para fixar identidade visual Dark/GitHub Enterprise
    st.markdown(f"""
        <style>
        .stMetric {{ background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }}
        div[data-testid="stSidebar"] {{ background-color: #0d1117; }}
        div[data-testid="stInfo"] {{ background-color: #0d1117; border: 1px solid #30363d; color: {THEME_COLOR}; }}
        </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def get_cached_data():
        return load_processed_data()

    df_raw = get_cached_data()

    if df_raw.is_empty():
        st.error("❌ Framework Error: Fonte de dados (Parquet) inacessível.")
        st.stop()

    # --- SIDEBAR: SEGMENTAÇÃO COM DEFAULT SETTINGS (UX FIRST) ---
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=80)
        st.title("Market Control")
        st.divider()
        
        st.subheader("🎯 Filtros Estratégicos")
        
        # UX: Pré-seleção total para visão macro imediata
        marcas_disp = sorted(df_raw["marca"].unique().to_list())
        sel_marcas = st.multiselect("Foco em Marcas:", marcas_disp, default=marcas_disp)
        
        # UX: Default focado em 'Brasil' conforme necessidade do time de vendas
        ufs_disp = sorted(df_raw["uf"].unique().to_list())
        default_ufs = [uf for uf in ufs_disp if uf == "Brasil"] or ufs_disp
        sel_ufs = st.multiselect("Regiões Estratégicas:", ufs_disp, default=default_ufs)
        
        setores_disp = sorted(df_raw["industry_sector"].unique().to_list())
        sel_setores = st.multiselect("Setores Industriais:", setores_disp, default=setores_disp)
        
        st.divider()
        sel_days = st.slider("Janela de Observação (Dias):", 1, 31, (1, 31))
        st.caption("Black Crow Intelligence | v1.3.0 Platinum")

    # Pipeline de processamento e filtragem de dados
    df_filt = apply_business_filters(df_raw, sel_marcas, sel_ufs, sel_days)
    if not sel_setores == setores_disp:
        df_filt = df_filt.filter(pl.col("industry_sector").is_in(sel_setores))

    if not df_filt.is_empty():
        # Cálculo global de métricas para estabilidade visual entre gráficos
        v_dia, m, s = AnalyticsService.calculate_spc_metrics(df_filt)
        
        st.title(f"{APP_TITLE}")
        
        # --- BLOCO 1: OPERATIONAL KPIs ---
        k1, k2, k3, k4 = st.columns(4)
        dist_data = AnalyticsService.get_pareto_distribution(df_filt)
        
        k1.metric("Volume Total", f"{len(df_filt):,}")
        k2.metric("Market Leader", dist_data["marca"][0] if not dist_data.is_empty() else "N/A")
        k3.metric("Média Diária", f"{len(df_filt)/31:.1f}")
        k4.metric("Market Breadth (UFs)", len(df_filt["uf"].unique()))
        
        st.divider()

        # --- BLOCO 2: DIAGNÓSTICO DE TRAJETÓRIA ---
        st.subheader("📊 Diagnóstico de Trajetória")
        fig_area = px.area(v_dia, x='dia_do_mes', y='vol', 
                        template="plotly_dark", 
                        color_discrete_sequence=[THEME_COLOR])
        fig_area.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig_area, use_container_width=True)
        
        proj_vol, trend = PredictionService.get_market_trend(df_filt)
        st.info(f"Tendência de mercado: **{trend}**. Forecast de fechamento: **{proj_vol} unidades**.")

        st.divider()

        # --- BLOCO 3: ESTRATÉGIA & SPC ---
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("🏆 Market Share (Pareto)")
            fig_p = px.bar(dist_data, x='vendas', y='marca', orientation='h', 
                        color='vendas', template="plotly_dark")
            st.plotly_chart(fig_p, use_container_width=True)

        with c2:
            st.subheader("📈 Estabilidade (SPC - Carta I)")
            
            # Parametrização Sigma 2 e LCL mínimo para detecção de inatividade
            k_sigma = 2 
            ucl = m + k_sigma * s
            lcl = max(1, m - k_sigma * s)

            # Join preventivo para garantir metadados (Marca) no hover e alertas
            if "marca" not in v_dia.columns:
                v_dia = v_dia.join(
                    df_filt.select(["dia_do_mes", "marca"]).unique(subset=["dia_do_mes"]),
                    on="dia_do_mes", how="left"
                )

            v_dia = v_dia.with_columns(
                status=pl.when(pl.col("vol") > ucl).then(pl.lit("Expansão"))
                        .when(pl.col("vol") <= lcl).then(pl.lit("Retração"))
                        .otherwise(pl.lit("Estável"))
            )

            # Alertas nominais focados em conversão comercial
            exp_pts = v_dia.filter(pl.col("status") == "Expansão")
            ret_pts = v_dia.filter(pl.col("status") == "Retração")
            
            if not exp_pts.is_empty():
                nomes = ", ".join(exp_pts["marca"].unique().to_list())
                st.success(f"🚀 **Expansão Detectada:** Foco comercial em **{nomes}**.")
            if not ret_pts.is_empty():
                nomes = ", ".join(ret_pts["marca"].unique().to_list())
                st.error(f"⚠️ **Risco de Retração:** Investigar churn em **{nomes}**.")

            # Gráfico com hover blindado via custom_data (compatibilidade Streamlit Cloud)
            fig_spc = px.line(v_dia, x='dia_do_mes', y='vol', markers=True, 
                            custom_data=['marca', 'status'],
                            template="plotly_dark",
                            range_y=[-0.5, max(ucl, v_dia["vol"].max()) * 1.2])
            
            fig_spc.update_traces(
                hovertemplate="<br>".join([
                    "Dia: %{x}", "Volume: %{y}",
                    "Marca: %{customdata[0]}", "Status: %{customdata[1]}"
                ])
            )

            fig_spc.add_hline(y=ucl, line_dash="dash", line_color="#238636", annotation_text="UCL (Expansão)")
            fig_spc.add_hline(y=m, line_dash="solid", line_color="white", opacity=0.2)
            fig_spc.add_hline(y=lcl, line_dash="dash", line_color="#da3633", annotation_text="LCL (Retração)")

            color_map = {'Expansão': '#238636', 'Retração': '#da3633', 'Estável': THEME_COLOR}
            colors = [color_map[s] for s in v_dia["status"].to_list()]
            
            fig_spc.update_traces(marker=dict(color=colors, size=10, line=dict(width=1, color='white')))
            st.plotly_chart(fig_spc, use_container_width=True)

        st.divider()
        st.subheader("🧠 Drivers de Decisão (Logic Engine)")
        st.code(AnalyticsService.get_decision_rules(df_filt), language='python')

    else:
        st.sidebar.warning("⚠️ Contexto vazio. Revise os filtros de segmentação.")