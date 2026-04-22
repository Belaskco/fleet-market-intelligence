import plotly.express as px
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from datetime import timedelta

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# --- UI COMPONENTS (COMPONENTES MODULARES) ---

def apply_enterprise_styles():
    """
    Aplica a paleta 'Executive Clarity'. 
    Foco em legibilidade absoluta, contraste de alto nível e conforto visual.
    """
    st.markdown(f"""
        <style>
        /* 1. FUNDO DA PÁGINA (CINZA NEUTRO PROFISSIONAL) */
        [data-testid="stAppViewContainer"] {{
            background-color: #F1F5F9 !important;
        }}

        /* FUNDO DA SIDEBAR */
        [data-testid="stSidebar"] {{
            background-color: #FFFFFF !important;
            border-right: 2px solid #E2E8F0;
        }}

        /* 2. AJUSTE DE LAYOUT (CORREÇÃO DO TOPO) */
        .block-container {{
            max-width: 98% !important;
            padding-left: 3rem !important;
            padding-right: 3rem !important;
            padding-top: 4rem !important; /* Aumentado para o cabeçalho não fugir */
        }}

        /* 3. DESIGN DOS PAINÉIS (SOFT WHITE COM CONTRASTE) */
        .stMetric {{ 
            border: 1px solid #CBD5E1; 
            padding: 25px; 
            border-radius: 12px; 
            background-color: #FFFFFF !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}

        /* TEXTOS DAS MÉTRICAS (PRETO PROFISSIONAL) */
        [data-testid="stMetricValue"] {{ 
            font-size: 2.2rem !important; 
            font-weight: 800 !important; 
            color: #0F172A !important; 
        }}
        
        [data-testid="stMetricLabel"] {{
            color: #334155 !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
        }}

        /* BOTÕES (ALTA VISIBILIDADE) */
        .stButton button {{ 
            width: 100%; 
            font-weight: 700; 
            text-transform: uppercase; 
            border-radius: 8px;
            background-color: #FFFFFF !important;
            border: 2px solid #475569 !important;
            color: #0F172A !important;
            transition: all 0.3s ease;
        }}
        
        .stButton button:hover {{
            border-color: {THEME_COLOR} !important;
            color: {THEME_COLOR} !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        
        /* HEADER PRINCIPAL (CORREÇÃO DE POSIÇÃO E COR) */
        .header-title {{ 
            font-size: 2.5rem; 
            font-weight: 900; 
            color: #0F172A; 
            border-bottom: 5px solid {THEME_COLOR}; 
            padding-bottom: 12px; 
            margin-bottom: 35px; 
            letter-spacing: -1px;
        }}

        /* SUBHEADERS E TÍTULOS DE SECÇÃO */
        h1, h2, h3, .section-header {{
            color: #0F172A !important;
            font-weight: 800 !important;
            letter-spacing: -0.5px;
        }}

        /* TEXTOS GERAIS (FORÇAR LEITURA) */
        p, span, label, li {{
            color: #1E293B !important;
            font-weight: 500 !important;
        }}
        
        .section-header {{
            color: #475569;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            margin-top: 1.5rem;
        }}

        /* CORREÇÃO PARA DATAFRAMES / TABELAS */
        [data-testid="stTable"], .stDataFrame {{
            background-color: #FFFFFF !important;
            border-radius: 8px;
            padding: 10px;
        }}
        
        /* CORRIGE TEXTO EM INPUTS E FILTROS */
        .stSelectbox label, .stSlider label {{
            color: #0F172A !important;
            font-weight: 700 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    """Gerencia todos os filtros de segmentação na barra lateral."""
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=95)
        st.markdown("<h1 style='font-size: 1.8rem; margin-top:10px; color:#0F172A;'>Black Crow</h1>", unsafe_allow_html=True)
        st.caption("Intelligence Unit | v4.8.4")
        st.divider()
        
        def smart_filter(label, col):
            opts = sorted(df[col].unique().to_list())
            with st.expander(f"Filtro: {label}", expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, opts, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, opts, False))
                return [o for o in opts if st.checkbox(o, key=f"chk_{label}_{o}", value=st.session_state.get(f"chk_{label}_{o}", True))]

        id_col = "company_name" if "company_name" in df.columns else "marca"
        geo_col = "hq_country" if "hq_country" in df.columns else "uf"
        
        filtros = {
            "marcas": smart_filter("Marcas / Empresas", id_col),
            "paises": smart_filter("Países / Regiões", geo_col),
            "setores": smart_filter("Setores Industriais", "industry_sector"),
            "dias": st.slider("Recorte por Dia do Mês:", 1, 31, (1, 31))
        }
        return filtros

def render_spc_chart(v_semanal, v_future, m, s):
    """Renderiza a Carta de Controle com Forecast Pontilhado e Alertas."""
    ucl, lcl = m + 2*s, max(0, m - 2*s)
    
    # 1. Alertas de Execução (Topo)
    last_vol = v_semanal.tail(1)["vol"][0]
    if last_vol > ucl: 
        st.success(f"🚀 **Expansão Crítica:** Volume semanal ({last_vol}) rompeu o limite superior UCL.")
    elif last_vol < lcl: 
        st.error(f"⚠️ **Retração Crítica:** Volume semanal ({last_vol}) abaixo do limite inferior LCL.")
    else: 
        st.info(f"✅ **Fluxo Estável:** Operação em regime nominal ({last_vol} pedidos).")

    # 2. Gráfico Plotly SPC
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=v_semanal['semana'], y=v_semanal['vol'], 
        mode='lines+markers', name='Histórico Real', 
        line=dict(color=THEME_COLOR, width=3),
        marker=dict(size=9, color=THEME_COLOR, line=dict(width=2, color='white'))
    ))
    
    if not v_future.is_empty():
        last_date = v_semanal.tail(1)["semana"][0]
        v_fut_dates = v_future.with_columns([
            pl.Series([last_date + timedelta(weeks=i+1) for i in range(len(v_future))]).alias("semana")
        ])
        
        h_tail = v_semanal.select(["semana", "vol"]).tail(1).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        f_tail = v_fut_dates.select(["semana", "vol"]).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        
        conn = pl.concat([h_tail, f_tail])
        fig.add_trace(go.Scatter(
            x=conn['semana'], y=conn['vol'], 
            mode='lines', name='Forecast Nixtla', 
            line=dict(color=THEME_COLOR, dash='dot', width=3)
        ))
    
    fig.add_hline(y=ucl, line_dash="dash", line_color="#10B981", annotation_text="UCL", annotation_position="top left")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#EF4444", annotation_text="LCL", annotation_position="bottom left")
    
    fig.update_layout(
        height=380, 
        margin=dict(t=30, b=10), 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='white', 
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, linecolor='#94A3B8', tickfont=dict(color='#0F172A', size=12)),
        yaxis=dict(showgrid=True, gridcolor='#E2E8F0', zeroline=False, tickfont=dict(color='#0F172A', size=12))
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if not v_future.is_empty() and v_future.filter(pl.col("vol") > ucl).height > 0:
        st.warning("🔮 **Alerta de Tendência:** O motor detectou picos de demanda prováveis no próximo ciclo.")

def render_scorecard(total, m_week, share_lider, confianca, forecast_val, trend):
    """Exibe a grade de KPIs principais."""
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Volume Acumulado", f"{total:,}", help="Pedidos processados (excluíndo D-0).")
    k2.metric("Líder de Canal", f"{share_lider[0][:12]}", delta=f"{share_lider[1]:.1%} Share")
    k3.metric("Média Semanal", f"{m_week:.1f} un/sem")
    k4.metric("Previsibilidade", f"{confianca:.1f}%", help="Nota Logic Engine v4.0 (Enterprise)")
    k5.metric("Forecast Próx. Ciclo", human_format(forecast_val * 125000), delta=trend)

# --- UTILS & ENGINE ---

def set_all_state(label, options, value):
    for opt in options: 
        st.session_state[f"chk_{label}_{opt}"] = value

def human_format(num):
    if num is None or num == 0: return "0"
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0: return f"{num:.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"

def run_dashboard():
    """Ponto de entrada do Dashboard Black Crow v4.8.4."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    
    apply_enterprise_styles()
    df_raw = load_processed_data()
    if df_raw.is_empty(): 
        st.error("❌ Base de dados Fleet não carregada corretamente."); st.stop()

    date_col = "purchase_date" if "purchase_date" in df_raw.columns else "data_faturamento"
    f = render_sidebar(df_raw)
    
    df = apply_business_filters(df_raw, f['marcas'], f['paises'], f['dias'])
    if f['setores']: 
        df = df.filter(pl.col("industry_sector").is_in(f['setores']))
    
    df = df.filter(pl.col(date_col) < df_raw[date_col].max()) # Filtro D-1

    if not df.is_empty():
        # Motores
        v_sem = df.with_columns(pl.col(date_col).dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        proj_vol, trend = PredictionService.get_market_trend(df)
        v_fut = PredictionService.get_daily_forecast(df, horizon=4)
        ins = PredictionService.get_strategic_insights(df)
        
        # Header Principal
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        
        # KPIs
        lider_info = (dist.tail(1)['marca'][0], dist.tail(1)['vendas'][0]/len(df))
        render_scorecard(len(df), m_w, lider_info, ins.get('confianca', 0), proj_vol, trend)
        st.divider()

        # Grid Principal
        c_l, c_r = st.columns([1.6, 1])
        with c_l:
            st.subheader("📊 Diagnóstico de Trajetória")
            fig_area = px.area(v_sem, x='semana', y='vol', color_discrete_sequence=[THEME_COLOR])
            fig_area.update_layout(
                height=300, margin=dict(t=0,b=0), 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                xaxis_title=None, yaxis_title="Volume Acumulado",
                xaxis=dict(tickfont=dict(color='#0F172A')),
                yaxis=dict(tickfont=dict(color='#0F172A'))
            )
            st.plotly_chart(fig_area, use_container_width=True)
            
            st.markdown("<div class='section-header'>📈 Estabilidade e Controle Estatístico</div>", unsafe_allow_html=True)
            render_spc_chart(v_sem, v_fut, m_w, s_w)

        with c_r:
            st.subheader("🔮 Antecipação Nominal (Próx. Ciclo)")
            df_p = PredictionService.get_client_predictions(df)
            if not df_p.is_empty():
                df_view = df_p.with_columns(pl.col("Valor_Est").map_elements(human_format, return_dtype=pl.String).alias("Potencial Est."))
                st.dataframe(
                    df_view.select(["Cliente", "Qtd_Prevista", "Potencial Est.", "Probabilidade"]), 
                    use_container_width=True, hide_index=True, 
                    column_config={"Probabilidade": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=1, color="green")}
                )
            else:
                st.warning("Volume de dados insuficiente.")
            
            st.markdown("<div class='section-header'>🏆 Pareto de Líderes (Share)</div>", unsafe_allow_html=True)
            fig_bar = px.bar(dist.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR])
            fig_bar.update_layout(
                height=380, margin=dict(t=0,b=0), 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                xaxis_title="Volume de Pedidos", yaxis_title=None,
                xaxis=dict(tickfont=dict(color='#0F172A')),
                yaxis=dict(tickfont=dict(color='#0F172A'))
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()
        st.markdown(f"""
        <div style="background-color:#0F172A; padding:20px; border-radius:10px; color:white; font-family:monospace;">
        # Strategic Insights v4.8.4 | Perfil {ins.get('perfil').upper()} | HHI: {ins.get('hhi',0):.2f} | CV: {ins.get('cv',0):.2f}<br>
        - Estabilidade: {ins.get('estabilidade').upper()}<br>
        - Nota de Confiança: {ins.get('confianca', 0):.1f}% (Nível Profissional Enterprise)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros.")