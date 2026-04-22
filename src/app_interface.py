import plotly.express as px
import streamlit as st
import polars as pl
import sys
import os

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# Setup de ambiente: Garante a resolução de caminhos no Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def set_all_state(label, options, value):
    """
    CALLBACK DE ESTADO: Gerencia a seleção em massa dos filtros.
    Essencial para manter a sincronia entre botões e checkboxes no Streamlit.io.
    """
    for opt in options:
        st.session_state[f"chk_{label}_{opt}"] = value

def run_dashboard():
    """
    Interface Principal Black Crow Intel.
    Arquitetura voltada para diagnóstico estatístico (SPC) e predição nominal (Nixtla).
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Injeção de CSS para identidade visual Dark e alta densidade de informação
    st.markdown(f"""
        <style>
        [data-testid="stMetricValue"] {{ font-size: 1.6rem !important; }}
        .stMetric {{ background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }}
        div[data-testid="stSidebar"] {{ background-color: #0d1117; }}
        div[data-testid="stInfo"] {{ background-color: #0d1117; border: 1px solid #30363d; color: {THEME_COLOR}; }}
        .stButton button {{ width: 100%; height: 1.6rem; font-size: 0.7rem !important; font-weight: bold; }}
        </style>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def get_cached_data():
        return load_processed_data()

    df_raw = get_cached_data()

    if df_raw.is_empty():
        st.error("❌ Erro: Repositório fct_sales indisponível ou vazio.")
        st.stop()

    # --- SIDEBAR: SEGMENTAÇÃO ---
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=70)
        st.title("Market Control")
        st.divider()
        
        def create_smart_filter(label, options):
            with st.expander(label, expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, options, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, options, False))
                selected = [opt for opt in options if st.checkbox(opt, key=f"chk_{label}_{opt}", value=st.session_state.get(f"chk_{label}_{opt}", True))]
                return selected

        sel_marcas = create_smart_filter("Marcas", sorted(df_raw["marca"].unique().to_list()))
        sel_paises = create_smart_filter("Mercados", sorted(df_raw["uf"].unique().to_list()))
        sel_setores = create_smart_filter("Setores", sorted(df_raw["industry_sector"].unique().to_list()))

        st.divider()
        sel_days = st.slider("Janela de Observação (Dias):", 1, 31, (1, 31))
        st.caption("Black Crow Intelligence | v2.0.0 Platinum")

    # --- ENGINE DE DADOS ---
    df_filt = apply_business_filters(df_raw, sel_marcas, sel_paises, sel_days)
    if sel_setores:
        df_filt = df_filt.filter(pl.col("industry_sector").is_in(sel_setores))

    if not df_filt.is_empty():
        # Cálculos Analíticos e Predição Nixtla
        v_dia, m, s = AnalyticsService.calculate_spc_metrics(df_filt)
        dist_data = AnalyticsService.get_pareto_distribution(df_filt)
        total_vol = len(df_filt)
        proj_vol, trend = PredictionService.get_market_trend(df_filt)
        
        st.title(f"{APP_TITLE}")
        
        # --- BLOCO 1: KPIs (RESUMO EXECUTIVO) ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Volume Total", f"{total_vol:,}")
        if not dist_data.is_empty():
            k2.metric("Líder de Mix", f"{dist_data['marca'][0][:12]}...", delta=f"{(dist_data['vendas'][0]/total_vol):.1%} Share")
        k3.metric("Taxa Conversão", f"{total_vol/(sel_days[1]-sel_days[0]+1):.1f} un/dia")
        k4.metric("Abrangência", f"{len(df_filt['uf'].unique())} Mercados")
        k5.metric("Forecast Total", f"{proj_vol} un", delta=trend)

        st.divider()

        # --- BLOCO 2: DIAGNÓSTICO DE TRAJETÓRIA & ANTECIPAÇÃO (NIXTLA) ---
        st.subheader("📊 Diagnóstico de Trajetória & Predição Nominal")
        
        # Histórico Visual (Area Chart)
        fig_area = px.area(v_dia, x='dia_do_mes', y='vol', template="plotly_dark", color_discrete_sequence=[THEME_COLOR])
        fig_area.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig_area, use_container_width=True)
        
        # Divisão entre Tendência Macro e Detalhamento Micro (Por Cliente)
        c_trend, c_nixtla = st.columns([1, 2])
        
        with c_trend:
            st.info(f"Tendência: **{trend}**\n\nForecast: **{proj_vol} un.**")
            st.caption("Projeção baseada em volume histórico agregado via motor Nixtla.")

        with c_nixtla:
            # Integração Nixtla: Predição por cliente para a próxima janela
            try:
                df_forecast = PredictionService.get_client_predictions(df_filt)
                if not df_forecast.is_empty():
                    st.dataframe(
                        df_forecast, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Cliente": st.column_config.TextColumn("🎯 Próximas Compras"),
                            "Qtd_Prevista": st.column_config.NumberColumn("Pedidos Est."),
                            "Volume_Est": st.column_config.NumberColumn("Vol. Total")
                        }
                    )
                else:
                    st.warning("⚠️ Volume insuficiente para forecast nominal.")
            except Exception as e:
                st.error("⚠️ Falha ao processar predições Nixtla.")

        st.divider()

        # --- BLOCO 3: ESTABILIDADE E MIX ---
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏆 Distribuição de Mix")
            st.plotly_chart(px.bar(dist_data, x='vendas', y='marca', orientation='h', template="plotly_dark"), use_container_width=True)
        with c2:
            st.subheader("📈 Estabilidade Estatística (SPC)")
            ucl, lcl = m + 2*s, max(0, m - 2*s)
            v_dia = v_dia.join(df_filt.select(["dia_do_mes", "marca"]).unique(subset=["dia_do_mes"]), on="dia_do_mes", how="left")
            v_dia = v_dia.with_columns(status=pl.when(pl.col("vol") > ucl).then(pl.lit("Expansão")).when(pl.col("vol") <= lcl).then(pl.lit("Retração")).otherwise(pl.lit("Estável")))
            
            exp, ret = v_dia.filter(pl.col("status") == "Expansão"), v_dia.filter(pl.col("status") == "Retração")
            if not exp.is_empty(): st.success(f"🚀 **Expansão:** Picos em {', '.join(exp['marca'].unique().to_list())}")
            if not ret.is_empty(): st.error(f"⚠️ **Risco:** Quedas em {', '.join(ret['marca'].unique().to_list())}")

            fig_spc = px.line(v_dia, x='dia_do_mes', y='vol', markers=True, template="plotly_dark")
            fig_spc.add_hline(y=ucl, line_dash="dash", line_color="#238636")
            fig_spc.add_hline(y=lcl, line_dash="dash", line_color="#da3633")
            st.plotly_chart(fig_spc, use_container_width=True)

        st.divider()

        # --- BLOCO 4: LOGIC ENGINE (DIAGNÓSTICO ISENTO) ---
        st.subheader("🧠 Drivers de Decisão (Logic Engine)")
        hhi = (dist_data["vendas"] / total_vol).pow(2).sum()
        vol_cv = (s / m) if m > 0 else 0
        confianca = max(0, 100 - (vol_cv * 100))
        
        st.markdown(f"""
        ```python
        # Analytical Diagnosis - Logic Engine v2.0.0
        - Saúde da Carteira: Perfil {'CONCENTRADO' if hhi > 0.25 else 'DIVERSIFICADO'} (HHI: {hhi:.2f}).
        - Previsibilidade: Nota de Confiança em {confianca:.1f}% (Base estatística Sigma-2).
        - Insight de Compra: {len(df_forecast) if 'df_forecast' in locals() else 0} clientes com propensão detectada.
        - Estabilidade: {'Atenção: Alta oscilação detectada.' if vol_cv > 0.4 else 'Fluxo operacional em regime de normalidade.'}
        ```
        """, unsafe_allow_html=True)

    else:
        st.sidebar.warning("⚠️ Ajuste os filtros para gerar análise operacional.")