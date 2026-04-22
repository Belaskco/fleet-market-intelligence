import plotly.express as px
import streamlit as st
import polars as pl
import sys
import os
import plotly.graph_objects as go

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# Setup de ambiente para garantir caminhos de módulos em produção
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def human_format(num):
    """Converte números grandes para formatos legíveis (K, M, B)."""
    if num is None or num == 0: return "0"
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{:.1f}{}'.format(num, ['', 'K', 'M', 'B', 'T'][magnitude])

def set_all_state(label, options, value):
    """Callback de controle de estado para seleção em massa."""
    for opt in options:
        st.session_state[f"chk_{label}_{opt}"] = value

def run_dashboard():
    """
    Interface Black Crow Intel v2.4.0.
    Design Adaptativo (Light/Dark), Filtro D-1 e Projeção Pontilhada Nixtla.
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # CSS Adaptativo: Remove fundos fixos, foca em bordas e sombras sutis
    st.markdown(f"""
        <style>
        [data-testid="stMetricValue"] {{ font-size: 1.6rem !important; }}
        .stMetric {{ 
            border: 1px solid rgba(128, 128, 128, 0.2); 
            padding: 15px; 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .stButton button {{ width: 100%; height: 1.6rem; font-size: 0.7rem !important; font-weight: bold; }}
        div[data-testid="stExpander"] {{ border: none !important; }}
        </style>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def get_cached_data(): 
        return load_processed_data()

    df_raw = get_cached_data()
    if df_raw.is_empty():
        st.error("❌ Base de dados indisponível."); st.stop()

    # --- SIDEBAR: FILTROS ESTRATÉGICOS ---
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=70)
        st.title("Market Control")
        st.divider()
        
        def create_smart_filter(label, options):
            with st.expander(label, expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, options, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, options, False))
                return [opt for opt in options if st.checkbox(opt, key=f"chk_{label}_{opt}", value=st.session_state.get(f"chk_{label}_{opt}", True))]

        sel_marcas = create_smart_filter("Resellers", sorted(df_raw["marca"].unique().to_list()))
        sel_paises = create_smart_filter("Territórios", sorted(df_raw["uf"].unique().to_list()))
        
        st.divider() 
        sel_days = st.slider("Janela Mensal (Ordens):", 1, 31, (1, 31))
        st.caption("v2.4.0 | Adaptive Nixtla")

    # --- MOTOR DE DADOS COM BLINDAGEM D-1 ---
    # Removemos o último dia do dataset para evitar quedas artificiais no gráfico
    max_date = df_raw["data_faturamento"].max()
    df_filt = apply_business_filters(df_raw, sel_marcas, sel_paises, sel_days)
    df_filt = df_filt.filter(pl.col("data_faturamento") < max_date)

    if not df_filt.is_empty():
        # Inteligência Analítica
        total_vol = len(df_filt)
        v_dia, m, s = AnalyticsService.calculate_spc_metrics(df_filt)
        # Pareto Invertido: Ordenamos de forma ascendente para o gráfico de barras horizontais (Líder no Topo)
        dist_data = AnalyticsService.get_pareto_distribution(df_filt).sort("vendas", descending=False)
        proj_vol, trend = PredictionService.get_market_trend(df_filt)
        v_future = PredictionService.get_daily_forecast(df_filt)
        
        st.title(f"{APP_TITLE}")
        
        # --- BLOCO 1: KPIs (Enterprise View) ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Ordens (D-1)", f"{total_vol:,}")
        if not dist_data.is_empty():
            lider_share = (dist_data.tail(1)['vendas'][0]/total_vol)
            k2.metric("Líder de Canal", f"{dist_data.tail(1)['marca'][0][:12]}...", delta=f"{lider_share:.1%} Share")
        k3.metric("Frequência", f"{total_vol/(sel_days[1]-sel_days[0]+1):.1f} ord/dia")
        k4.metric("Territórios", f"{len(df_filt['uf'].unique())}")
        # Valor monetário humanizado no KPI (Simulando 100x o volume em USD)
        k5.metric("Forecast (USD)", human_format(proj_vol * 1250), delta=trend)

        st.divider()

        # --- BLOCO 2: CICLOS E OPORTUNIDADES ---
        c_left, c_right = st.columns([1.5, 1])
        
        with c_left:
            st.subheader("📊 Ciclos de Venda (Histórico)")
            fig_area = px.area(v_dia, x='dia_do_mes', y='vol', color_discrete_sequence=[THEME_COLOR])
            fig_area.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_area, use_container_width=True)
        
        with c_right:
            st.subheader("🔮 Oportunidades Nixtla")
            df_forecast = PredictionService.get_client_predictions(df_filt)
            if not df_forecast.is_empty():
                # Formatação simplificada para Valor Previsto
                df_view = df_forecast.with_columns(
                    pl.col("Valor_Est").map_elements(human_format, return_dtype=pl.String).alias("Valor Humanizado")
                )
                st.dataframe(
                    df_view.select(["Cliente", "Qtd_Prevista", "Valor Humanizado", "Probabilidade"]), 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Probabilidade": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=1),
                        "Valor Humanizado": "Valor Est."
                    }
                )
            else:
                st.info("💡 Volume insuficiente para predição nominal.")

        st.divider()

        # --- BLOCO 3: PARETO E SPC COM FORECAST ---
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("🏆 Pareto de Líderes")
            # Gráfico de barras com o líder no topo
            fig_bar = px.bar(dist_data.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader("📈 Estabilidade + Forecast Nixtla")
            ucl, lcl = m + 2*s, max(0, m - 2*s)
            
            fig_spc = go.Figure()
            # Série Histórica (Linha Cheia)
            fig_spc.add_trace(go.Scatter(x=v_dia['dia_do_mes'], y=v_dia['vol'], mode='lines+markers', name='Histórico', line=dict(color=THEME_COLOR, width=2)))
            
            # Série Preditiva (Linha Pontilhada)
            if not v_future.is_empty():
                # Conectamos o último ponto real ao primeiro ponto predito
                last_real = v_dia.tail(1)
                v_future_conn = pl.concat([last_real, v_future])
                fig_spc.add_trace(go.Scatter(x=v_future_conn['dia_do_mes'], y=v_future_conn['vol'], mode='lines', name='Forecast', line=dict(color=THEME_COLOR, dash='dot', width=2)))
            
            fig_spc.add_hline(y=ucl, line_dash="dash", line_color="#238636", annotation_text="UCL")
            fig_spc.add_hline(y=lcl, line_dash="dash", line_color="#da3633", annotation_text="LCL")
            fig_spc.update_layout(height=350, margin=dict(t=10, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_spc, use_container_width=True)

        st.divider()

        # --- BLOCO 4: SPC & MIX ---
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏆 Mix de Clientes (Pareto)")
            st.plotly_chart(px.bar(dist_data, x='vendas', y='marca', orientation='h', template="plotly_dark"), use_container_width=True)
        with c2:
            st.subheader("📈 Estabilidade Estatística (SPC)")
            ucl, lcl = m + 2*s, max(0, m - 2*s)
            v_dia_spc = v_dia.join(df_filt.select(["dia_do_mes", "marca"]).unique(subset=["dia_do_mes"]), on="dia_do_mes", how="left")
            v_dia_spc = v_dia_spc.with_columns(status=pl.when(pl.col("vol") > ucl).then(pl.lit("Exp")).when(pl.col("vol") <= lcl).then(pl.lit("Ret")).otherwise(pl.lit("Est")))
            
            exp, ret = v_dia_spc.filter(pl.col("status") == "Exp"), v_dia_spc.filter(pl.col("status") == "Ret")
            if not exp.is_empty(): st.success(f"🚀 **Expansão:** {', '.join(exp['marca'].unique().to_list())}")
            if not ret.is_empty(): st.error(f"⚠️ **Retração:** {', '.join(ret['marca'].unique().to_list())}")
            
            fig_spc = px.line(v_dia_spc, x='dia_do_mes', y='vol', markers=True, template="plotly_dark")
            fig_spc.add_hline(y=ucl, line_dash="dash", line_color="#238636")
            fig_spc.add_hline(y=lcl, line_dash="dash", line_color="#da3633")
            st.plotly_chart(fig_spc, use_container_width=True)

        st.divider()

        # --- BLOCO 5: LOGIC ENGINE ---
        st.subheader("🧠 Drivers de Decisão (Logic Engine)")
        hhi = (dist_data["vendas"] / total_vol).pow(2).sum() if total_vol > 0 else 0
        vol_cv = (s / m) if m > 0 else 0
        confianca = max(0, 100-(vol_cv*100))
        
        st.markdown(f"""
        ```python
        # Strategic Insights - Logic Engine v2.2.8
        - Saúde da Carteira: Perfil {'CONCENTRADO' if hhi > 0.25 else 'DIVERSIFICADO'} (HHI: {hhi:.2f}).
        - Previsibilidade: Nota de Confiança em {confianca:.1f}% (Base Sigma-2).
        - Insight Comercial: {len(df_forecast) if 'df_forecast' in locals() else 0} oportunidades de faturamento nominal detectadas.
        - Estabilidade: {'Flutuação atípica detectada.' if vol_cv > 0.4 else 'Fluxo operacional em regime de normalidade.'}
        ```
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros para gerar análise operacional.")