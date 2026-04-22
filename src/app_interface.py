import plotly.express as px
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from datetime import timedelta

# Importação dos motores de inteligência do framework Black Crow
from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

def human_format(num):
    """
    Converte números brutos em escalas de fácil leitura executiva (K, M, B).
    Exemplo: 1.200.000 -> 1.2M
    """
    if num is None or num == 0: return "0"
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{:.1f}{}'.format(num, ['', 'K', 'M', 'B', 'T'][magnitude])

def set_all_state(label, options, value):
    """Gerenciador de estado para filtros globais (Seleção em Massa)."""
    for opt in options:
        st.session_state[f"chk_{label}_{opt}"] = value

def run_dashboard():
    """
    Interface Black Crow Intelligence v4.6.0 - Full Content Edition.
    Dashboard executivo com foco em Previsibilidade, Estabilidade e Antecipação.
    """
    st.set_page_config(
        page_title=APP_TITLE, 
        layout="wide", 
        initial_sidebar_state="expanded"
    )

    # CSS Enterprise Customizado (Conserva o visual em Dark/Light Mode)
    st.markdown(f"""
        <style>
        [data-testid="stMetricValue"] {{ font-size: 1.8rem !important; font-weight: 700; }}
        .stMetric {{ 
            border: 1px solid rgba(128, 128, 128, 0.2); 
            padding: 20px; 
            border-radius: 12px; 
            background-color: rgba(128, 128, 128, 0.05);
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        }}
        .stButton button {{ 
            width: 100%; 
            height: 1.8rem; 
            font-size: 0.75rem !important; 
            font-weight: 600; 
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .main-header {{ font-size: 2.2rem; font-weight: 800; margin-bottom: 0.5rem; }}
        </style>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def get_cached_data(): 
        return load_processed_data()

    df_raw = get_cached_data()
    if df_raw.is_empty(): 
        st.error("❌ Base de dados Fleet não localizada no repositório."); st.stop()

    # --- MAPEAMENTO DINÂMICO DE SCHEMA (Blindagem de Erros) ---
    date_col = "purchase_date" if "purchase_date" in df_raw.columns else "data_faturamento"
    id_col = "company_name" if "company_name" in df_raw.columns else "marca"
    val_col = "order_value" if "order_value" in df_raw.columns else "faturamento"

    # --- SIDEBAR: CONTROLES DE SEGMENTAÇÃO ---
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=85)
        st.markdown("<h2 style='margin-top:0;'>Market Control</h2>", unsafe_allow_html=True)
        st.divider()
        
        def create_smart_filter(label, col_name):
            options = sorted(df_raw[col_name].unique().to_list())
            with st.expander(f"Filtro: {label}", expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, options, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, options, False))
                return [opt for opt in options if st.checkbox(opt, key=f"chk_{label}_{opt}", value=st.session_state.get(f"chk_{label}_{opt}", True))]

        sel_marcas = create_smart_filter("Marcas", id_col)
        sel_paises = create_smart_filter("Países", "hq_country" if "hq_country" in df_raw.columns else "uf")
        sel_setores = create_smart_filter("Setores", "industry_sector")
        
        st.divider()
        sel_days = st.slider("Janela de Observação (Dia do Mês):", 1, 31, (1, 31))
        st.caption("Ajuste a janela para focar em ciclos específicos do faturamento.")

    # --- ENGINE DE PROCESSAMENTO ---
    max_date = df_raw[date_col].max()
    df_filt = apply_business_filters(df_raw, sel_marcas, sel_paises, sel_days)
    
    if sel_setores:
        df_filt = df_filt.filter(pl.col("industry_sector").is_in(sel_setores))
    
    # Blindagem Rapha (D-1): Remove dados parciais do dia atual para não derrubar a média
    df_filt = df_filt.filter(pl.col(date_col) < max_date)

    if not df_filt.is_empty():
        # Chamadas aos Motores de Inteligência (Prediction & Analytics)
        v_semanal = df_filt.with_columns(
            pl.col(date_col).dt.truncate("1w").alias("semana")
        ).group_by("semana").len(name="vol").sort("semana")
        
        m_week, s_week = v_semanal["vol"].mean(), v_semanal["vol"].std()
        
        dist_data = AnalyticsService.get_pareto_distribution(df_filt).sort("vendas", descending=False)
        proj_vol, trend = PredictionService.get_market_trend(df_filt)
        v_future = PredictionService.get_daily_forecast(df_filt, horizon=4)
        insights = PredictionService.get_strategic_insights(df_filt)
        
        # Título Principal
        st.markdown(f"<div class='main-header'>{APP_TITLE}</div>", unsafe_allow_html=True)
        
        # --- BLOCO 1: KPI SCORECARD ---
        k1, k2, k3, k4, k5 = st.columns(5)
        
        k1.metric("Volume Acumulado", f"{len(df_filt):,}", 
                  help="Total de pedidos faturados no período selecionado (Excluíndo D-0).")
        
        if not dist_data.is_empty():
            lider = dist_data.tail(1)
            share = (lider['vendas'][0]/len(df_filt))
            k2.metric("Líder de Canal", f"{lider['marca'][0][:14]}", 
                      delta=f"{share:.1%} Share", help="Marca com maior volume de pedidos.")
        
        k3.metric("Média Semanal", f"{m_week:.1f} un/sem", 
                  help="Volume médio transacionado por ciclo de 7 dias.")
        
        k4.metric("Previsibilidade", f"{insights.get('confianca', 0):.1f}%", 
                  help="Nota de Confiança baseada na estabilidade do canal (Logic Engine v4.0).")
        
        k5.metric("Forecast Próx. Ciclo", human_format(proj_vol * 125000), 
                  delta=trend, help="Projeção financeira estimada para as próximas 4 semanas.")

        st.divider()

        # --- BLOCO 2: DIAGNÓSTICO E OPORTUNIDADES ---
        c_left, c_right = st.columns([1.6, 1])
        
        with c_left:
            st.subheader("📊 Diagnóstico de Trajetória Operacional")
            # Gráfico de área para volume semanal histórico
            fig_area = px.area(v_semanal, x='semana', y='vol', 
                               color_discrete_sequence=[THEME_COLOR])
            fig_area.update_layout(
                height=300, 
                margin=dict(t=10, b=10, l=10, r=10), 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title=None, yaxis_title="Pedidos/Semana"
            )
            st.plotly_chart(fig_area, use_container_width=True)
            
            st.info(f"O mercado apresenta trajetória de **{trend.upper()}**. Projeção estimada para o próximo ciclo: **{proj_vol} unidades**.")
        
        with c_right:
            st.subheader("🔮 Antecipação Nominal (Nixtla)")
            df_forecast = PredictionService.get_client_predictions(df_filt)
            
            if not df_forecast.is_empty():
                # Formatação financeira para a tabela de oportunidades
                df_view = df_forecast.with_columns(
                    pl.col("Valor_Est").map_elements(human_format, return_dtype=pl.String).alias("Valor Est.")
                )
                st.dataframe(
                    df_view.select(["Cliente", "Qtd_Prevista", "Valor Est.", "Probabilidade"]), 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Probabilidade": st.column_config.ProgressColumn(
                            "Confiança", min_value=0, max_value=1, format="%.2f"
                        ),
                        "Qtd_Prevista": st.column_config.NumberColumn("Qtd (un)")
                    }
                )
            else:
                st.warning("Volume de dados insuficiente para gerar antecipação por cliente.")

        st.divider()

        # --- BLOCO 3: CONTROLE E ESTRUTURA ---
        c_spc, c_pareto = st.columns([1.6, 1])
        
        with c_spc:
            # ALERTAS HIERÁRQUICOS (EXECUÇÃO REAL)
            ucl, lcl = m_week + 2*s_week, max(0, m_week - 2*s_week)
            last_week_vol = v_semanal.tail(1)["vol"][0]
            
            if last_week_vol > ucl:
                st.success(f"🚀 **Expansão Crítica:** Última semana ({last_week_vol}) rompeu o limite superior UCL ({ucl:.1f}).")
            elif last_week_vol < lcl:
                st.error(f"⚠️ **Retração Crítica:** Última semana ({last_week_vol}) abaixo do limite inferior LCL ({lcl:.1f}).")
            else:
                st.info(f"✅ **Estabilidade:** Fluxo semanal em regime nominal ({last_week_vol} pedidos).")

            # GRÁFICO SPC + FORECAST PONTILHADO
            st.subheader("📈 Controle Estatístico + Forecast")
            fig_spc = go.Figure()
            
            # Série Real
            fig_spc.add_trace(go.Scatter(
                x=v_semanal['semana'], y=v_semanal['vol'], 
                mode='lines+markers', name='Histórico', 
                line=dict(color=THEME_COLOR, width=3)
            ))
            
            # Série Preditiva (Pontilhada)
            if not v_future.is_empty():
                last_week_date = v_semanal.tail(1)["semana"][0]
                v_fut_plot = v_future.with_columns([
                    pl.Series([last_week_date + timedelta(weeks=i+1) for i in range(len(v_future))]).alias("semana")
                ]).select(["semana", "vol"])
                
                # Conexão entre as linhas
                hist_tail = v_semanal.select(["semana", "vol"]).tail(1).with_columns([
                    pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)
                ])
                v_fut_plot = v_fut_plot.with_columns([
                    pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)
                ])
                conn = pl.concat([hist_tail, v_fut_plot])
                
                fig_spc.add_trace(go.Scatter(
                    x=conn['semana'], y=conn['vol'], 
                    mode='lines', name='Forecast', 
                    line=dict(color=THEME_COLOR, dash='dot', width=3)
                ))
            
            fig_spc.add_hline(y=ucl, line_dash="dash", line_color="#238636", annotation_text="UCL")
            fig_spc.add_hline(y=lcl, line_dash="dash", line_color="#da3633", annotation_text="LCL")
            
            fig_spc.update_layout(
                height=350, margin=dict(t=10, b=10), 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                showlegend=False, xaxis_title=None
            )
            st.plotly_chart(fig_spc, use_container_width=True)

            # ALERTAS PREDITIVOS (ABAIXO DA CARTA)
            if not v_future.is_empty():
                fut_outlier = v_future.filter(pl.col("vol") > ucl)
                if not fut_outlier.is_empty():
                    st.warning(f"🔮 **Alerta Preditivo:** Possível pico de demanda detectado nas próximas 4 semanas. Prepare o estoque.")

        with c_pareto:
            st.subheader("🏆 Pareto de Líderes")
            # Gráfico de barras horizontais com líder no topo
            fig_bar = px.bar(
                dist_data.tail(10), x='vendas', y='marca', 
                orientation='h', color_discrete_sequence=[THEME_COLOR]
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                margin=dict(t=0, b=0), 
                height=450,
                xaxis_title="Volume de Pedidos", yaxis_title=None
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        
        # --- LOGIC ENGINE STRATEGIC FOOTER ---
        st.markdown(f"""
        ```python
        # Strategic Insights - Black Crow Logic Engine v4.6.0
        - Saúde da Carteira: Perfil {insights.get('perfil').upper()} (Índice HHI: {insights.get('hhi', 0):.2f}).
        - Previsibilidade: Nota de Confiança em {insights.get('confianca', 0):.1f}% (Nível Profissional Enterprise).
        - Estabilidade: {insights.get('estabilidade').upper()}.
        - Coeficiente de Variação (CV): {insights.get('cv', 0):.2f} (Base semanal filtrada).
        ```
        """)
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros para gerar análise operacional.")