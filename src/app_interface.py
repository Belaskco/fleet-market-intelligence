import plotly.express as px
import streamlit as st
import polars as pl
import sys
import os

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# Setup de ambiente para garantir a resolução de módulos em produção
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def run_dashboard():
    # Interface Black Crow Intel. 
    
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Injeção de CSS para identidade visual e métricas compactas
    st.markdown(f"""
        <style>
        [data-testid="stMetricValue"] {{ font-size: 1.6rem !important; }}
        .stMetric {{ background-color: #161b22; border: 1px solid #30363d; padding: 15px; border-radius: 10px; }}
        div[data-testid="stSidebar"] {{ background-color: #0d1117; }}
        .stButton button {{ width: 100%; padding: 0px; height: 1.5rem; font-size: 0.7rem !important; }}
        </style>
    """, unsafe_allow_html=True)

    @st.cache_data
    def get_cached_data():
        return load_processed_data()

    df_raw = get_cached_data()

    if df_raw.is_empty():
        st.error("❌ Erro de Sistema: Falha na ingestão do repositório de dados.")
        st.stop()

    # --- SIDEBAR: SEGMENTAÇÃO COM CONTROLE DE MASSA ---
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=80)
        st.title("Market Control")
        st.divider()
        
        st.subheader("🎯 Filtros Estratégicos")

        def create_smart_filter(label, options):
            
            # Expander com lógica funcional de Select All / Clear All.
            with st.expander(label, expanded=False):
                c1, c2 = st.columns(2)
                
                # Chaves de controle
                state_key = f"master_{label}"
                if state_key not in st.session_state:
                    st.session_state[state_key] = True

                # Callbacks para forçar a atualização dos checkboxes individuais
                if c1.button("Todos", key=f"btn_all_{label}"):
                    st.session_state[state_key] = True
                    for opt in options:
                        st.session_state[f"chk_{label}_{opt}"] = True
                
                if c2.button("Nenhum", key=f"btn_none_{label}"):
                    st.session_state[state_key] = False
                    for opt in options:
                        st.session_state[f"chk_{label}_{opt}"] = False

                selected = []
                for opt in options:
                    # O checkbox amarrado diretamente à sua própria chave no session_state
                    chk_key = f"chk_{label}_{opt}"
                    if chk_key not in st.session_state:
                        st.session_state[chk_key] = st.session_state[state_key]
                    
                    if st.checkbox(opt, key=chk_key):
                        selected.append(opt)
                return selected

        # Renderização dos filtros
        marcas_disp = sorted(df_raw["marca"].unique().to_list())
        sel_marcas = create_smart_filter("Foco em Marcas", marcas_disp)

        paises_disp = sorted(df_raw["uf"].unique().to_list())
        sel_paises = create_smart_filter("Mercados Internacionais", paises_disp)

        setores_disp = sorted(df_raw["industry_sector"].unique().to_list())
        sel_setores = create_smart_filter("Setores Industriais", setores_disp)

        st.divider()
        sel_days = st.slider("Janela de Observação (Dias):", 1, 31, (1, 31))
        st.caption("Black Crow Intelligence | v1.6.5 Platinum")

    # --- PROCESSAMENTO DE DADOS ---
    df_filt = apply_business_filters(df_raw, sel_marcas, sel_paises, sel_days)
    if sel_setores:
        df_filt = df_filt.filter(pl.col("industry_sector").is_in(sel_setores))

    if not df_filt.is_empty():
        # PRÉ-CÁLCULO: Variáveis globais para estabilidade de renderização
        v_dia, m, s = AnalyticsService.calculate_spc_metrics(df_filt)
        dist_data = AnalyticsService.get_pareto_distribution(df_filt)
        total_vol = len(df_filt)
        
        st.title(f"{APP_TITLE}")
        
        # --- BLOCO 1: STRATEGIC MARKET KPIs ---
        k1, k2, k3, k4, k5 = st.columns(5)
        
        loc_col = "city" if "city" in df_filt.columns else "uf"
        dist_loc = df_filt.group_by(loc_col).agg(pl.len().alias("cnt")).sort("cnt", descending=True)
        top_loc = dist_loc[loc_col][0] if not dist_loc.is_empty() else "N/A"
        concentracao = (dist_loc["cnt"][0] / total_vol) if total_vol > 0 else 0

        k1.metric("Volume Total", f"{total_vol:,}")

        if not dist_data.is_empty():
            full_leader = dist_data["marca"][0]
            display_leader = (full_leader[:12] + "...") if len(full_leader) > 12 else full_leader
            share = (dist_data["vendas"][0] / total_vol) if total_vol > 0 else 0
            k2.metric("Market Leader", display_leader, delta=f"{share:.1%} Share", help=f"Líder: {full_leader}")
        else:
            k2.metric("Market Leader", "N/A")

        k3.metric("Taxa de Conversão", f"{total_vol/(sel_days[1]-sel_days[0]+1):.1f} un/dia")
        k4.metric("Capilaridade", f"{len(dist_loc)} Locais", help=f"Hub detectado: {top_loc}")
        k5.metric("Conc. Regional", f"{concentracao:.1%}", help=f"Localidade dominante: {top_loc}")

        st.divider()

        # --- BLOCO 2: DIAGNÓSTICO DE TRAJETÓRIA ---
        st.subheader("📊 Diagnóstico de Trajetória")
        fig_area = px.area(v_dia, x='dia_do_mes', y='vol', template="plotly_dark", color_discrete_sequence=[THEME_COLOR])
        fig_area.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig_area, use_container_width=True)
        
        proj_vol, trend = PredictionService.get_market_trend(df_filt)
        st.info(f"O mercado apresenta tendência **{trend}**. Forecast de fechamento: **{proj_vol} unidades**.")

        st.divider()

        # --- BLOCO 3: ESTRATÉGIA & ESTABILIDADE (SPC) ---
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("🏆 Market Share (Pareto)")
            fig_p = px.bar(dist_data, x='vendas', y='marca', orientation='h', color='vendas', template="plotly_dark")
            st.plotly_chart(fig_p, use_container_width=True)

        with c2:
            st.subheader("📈 Estabilidade (SPC - Carta I)")
            ucl = m + 2 * s
            lcl = max(1, m - 2 * s)

            if "marca" not in v_dia.columns:
                v_dia = v_dia.join(df_filt.select(["dia_do_mes", "marca"]).unique(subset=["dia_do_mes"]), on="dia_do_mes", how="left")

            v_dia = v_dia.with_columns(
                status=pl.when(pl.col("vol") > ucl).then(pl.lit("Expansão"))
                        .when(pl.col("vol") <= lcl).then(pl.lit("Retração"))
                        .otherwise(pl.lit("Estável"))
            )

            # Alertas Nominais: Conversão de sinais estatísticos em alertas de negócio
            exp_pts = v_dia.filter(pl.col("status") == "Expansão")
            ret_pts = v_dia.filter(pl.col("status") == "Retração")
            
            if not exp_pts.is_empty():
                st.success(f"🚀 **Expansão:** Picos detectados em **{', '.join(exp_pts['marca'].unique().to_list())}**.")
            if not ret_pts.is_empty():
                st.error(f"⚠️ **Risco:** Inatividade/Queda em **{', '.join(ret_pts['marca'].unique().to_list())}**.")

            fig_spc = px.line(v_dia, x='dia_do_mes', y='vol', markers=True, 
                            custom_data=['marca', 'status'], template="plotly_dark",
                            range_y=[-0.5, max(ucl, v_dia["vol"].max()) * 1.2])
            
            fig_spc.update_traces(
                hovertemplate="<br>".join(["Dia: %{x}", "Vol: %{y}", "Marca: %{customdata[0]}", "Status: %{customdata[1]}"])
            )

            fig_spc.add_hline(y=ucl, line_dash="dash", line_color="#238636", annotation_text="UCL (Expansão)")
            fig_spc.add_hline(y=m, line_dash="solid", line_color="white", opacity=0.2)
            fig_spc.add_hline(y=lcl, line_dash="dash", line_color="#da3633", annotation_text="LCL (Retração)")

            color_map = {'Expansão': '#238636', 'Retração': '#da3633', 'Estável': THEME_COLOR}
            colors = [color_map[v] for v in v_dia["status"].to_list()]
            
            fig_spc.update_traces(marker=dict(color=colors, size=10, line=dict(width=1, color='white')))
            st.plotly_chart(fig_spc, use_container_width=True)

        st.divider()
        st.subheader("🧠 Drivers de Decisão (Logic Engine)")
        st.code(AnalyticsService.get_decision_rules(df_filt), language='python')

    else:
        st.sidebar.warning("⚠️ Seleção sem dados. Revise os filtros de segmentação.")