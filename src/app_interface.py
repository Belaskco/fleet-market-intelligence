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
    """Aplica o DNA visual da Black Crow para um visual executivo."""
    st.markdown(f"""
        <style>
        [data-testid="stMetricValue"] {{ font-size: 1.9rem !important; font-weight: 800; color: {THEME_COLOR}; }}
        .stMetric {{ 
            border: 1px solid rgba(128, 128, 128, 0.2); 
            padding: 20px; 
            border-radius: 12px; 
            background: rgba(128,128,128,0.03); 
        }}
        .stButton button {{ 
            width: 100%; 
            font-weight: 700; 
            text-transform: uppercase; 
            border-radius: 8px; 
        }}
        .header-title {{ 
            font-size: 2.2rem; 
            font-weight: 800; 
            color: {THEME_COLOR}; 
            border-bottom: 2px solid {THEME_COLOR}; 
            padding-bottom: 10px; 
            margin-bottom: 20px; 
        }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    """Gerencia todos os filtros de segmentação na barra lateral."""
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=80)
        st.title("Market Control")
        st.divider()
        
        def smart_filter(label, col):
            opts = sorted(df[col].unique().to_list())
            with st.expander(f"Filtro: {label}"):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, opts, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, opts, False))
                return [o for o in opts if st.checkbox(o, key=f"chk_{label}_{o}", value=st.session_state.get(f"chk_{label}_{o}", True))]

        # Mapeamento de colunas do Parquet
        id_col = "company_name" if "company_name" in df.columns else "marca"
        geo_col = "hq_country" if "hq_country" in df.columns else "uf"
        
        filtros = {
            "marcas": smart_filter("Marcas", id_col),
            "paises": smart_filter("Países", geo_col),
            "setores": smart_filter("Setores", "industry_sector"),
            "dias": st.slider("Janela Mensal (Dias):", 1, 31, (1, 31))
        }
        return filtros

def render_spc_chart(v_semanal, v_future, m, s):
    """Renderiza a Carta de Controle com Forecast Pontilhado e Alertas Hierárquicos."""
    ucl, lcl = m + 2*s, max(0, m - 2*s)
    
    # Alerta de Status Real (Acima da Carta)
    last_vol = v_semanal.tail(1)["vol"][0]
    if last_vol > ucl: 
        st.success(f"🚀 **Expansão:** Volume semanal ({last_vol}) rompeu o limite superior.")
    elif last_vol < lcl: 
        st.error(f"⚠️ **Retração:** Volume semanal ({last_vol}) abaixo do limite inferior.")
    else: 
        st.info(f"✅ **Estabilidade:** Operação em regime nominal ({last_vol} pedidos).")

    fig = go.Figure()
    # Histórico
    fig.add_trace(go.Scatter(x=v_semanal['semana'], y=v_semanal['vol'], mode='lines+markers', name='Real', line=dict(color=THEME_COLOR, width=3)))
    
    # Forecast Pontilhado
    if not v_future.is_empty():
        last_date = v_semanal.tail(1)["semana"][0]
        v_fut_dates = v_future.with_columns([pl.Series([last_date + timedelta(weeks=i+1) for i in range(len(v_future))]).alias("semana")])
        conn = pl.concat([v_semanal.select(["semana", "vol"]).tail(1), v_fut_dates.select(["semana", "vol"])])
        fig.add_trace(go.Scatter(x=conn['semana'], y=conn['vol'], mode='lines', name='Forecast', line=dict(color=THEME_COLOR, dash='dot', width=3)))
    
    # Linhas de Controle
    fig.add_hline(y=ucl, line_dash="dash", line_color="#238636", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#da3633", annotation_text="LCL")
    
    fig.update_layout(
        height=350, 
        margin=dict(t=10, b=10), 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(128,128,128,0.02)', 
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Alerta Preditivo (Abaixo da Carta)
    if not v_future.is_empty() and v_future.filter(pl.col("vol") > ucl).height > 0:
        st.warning("🔮 **Alerta Preditivo:** Possível pico de demanda detectado no forecast para as próximas semanas.")

def render_scorecard(total, m_week, share_lider, confianca, forecast_val, trend):
    """Exibe a grade de métricas principais no topo do dashboard."""
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Volume Acumulado", f"{total:,}")
    k2.metric("Líder de Canal", f"{share_lider[0][:12]}", delta=f"{share_lider[1]:.1%} Share")
    k3.metric("Média Semanal", f"{m_week:.1f} un/sem")
    k4.metric("Previsibilidade", f"{confianca:.1f}%", help="Logic Engine v4.0 (Enterprise)")
    k5.metric("Forecast Próx. Mês", human_format(forecast_val * 125000), delta=trend)

# --- UTILS & ENGINE ---

def set_all_state(label, options, value):
    """Ação para os botões 'Todos/Nenhum' nos filtros."""
    for opt in options: 
        st.session_state[f"chk_{label}_{opt}"] = value

def human_format(num):
    """Converte números grandes para o formato executivo (K, M, B)."""
    if num is None or num == 0: return "0"
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0: return f"{num:.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"

def run_dashboard():
    """Função mestre que orquestra a interface v4.8.0."""
    apply_enterprise_styles()
    df_raw = load_processed_data()
    if df_raw.is_empty(): 
        st.error("❌ Base de dados Fleet não encontrada."); st.stop()

    date_col = "purchase_date" if "purchase_date" in df_raw.columns else "data_faturamento"
    f = render_sidebar(df_raw)
    
    # Filtros e Processamento D-1
    df = apply_business_filters(df_raw, f['marcas'], f['paises'], f['dias'])
    if f['setores']: 
        df = df.filter(pl.col("industry_sector").is_in(f['setores']))
    
    # Remoção do último dia incompleto (Blindagem Rapha)
    df = df.filter(pl.col(date_col) < df_raw[date_col].max()) 

    if not df.is_empty():
        # Cálculos e Motores
        v_sem = df.with_columns(pl.col(date_col).dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        proj_vol, trend = PredictionService.get_market_trend(df)
        v_fut = PredictionService.get_daily_forecast(df, horizon=4)
        ins = PredictionService.get_strategic_insights(df)
        
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        
        # Top Scorecard
        lider_info = (dist.tail(1)['marca'][0], dist.tail(1)['vendas'][0]/len(df))
        render_scorecard(len(df), m_w, lider_info, ins.get('confianca', 0), proj_vol, trend)
        st.divider()

        # Dashboard Grid
        c_l, c_r = st.columns([1.6, 1])
        with c_l:
            st.subheader("📊 Diagnóstico de Trajetória")
            # Histórico de Área
            st.plotly_chart(px.area(v_sem, x='semana', y='vol', color_discrete_sequence=[THEME_COLOR]).update_layout(
                height=280, margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            ), use_container_width=True)
            
            # Carta de Controle
            st.subheader("📈 Controle Estatístico")
            render_spc_chart(v_sem, v_fut, m_w, s_w)

        with c_r:
            # Tabela de Antecipação
            st.subheader("🔮 Antecipação Nominal (Nixtla)")
            df_p = PredictionService.get_client_predictions(df)
            if not df_p.is_empty():
                st.dataframe(
                    df_p.with_columns(pl.col("Valor_Est").map_elements(human_format, return_dtype=pl.String).alias("Valor Est.")).select(["Cliente", "Qtd_Prevista", "Valor Est.", "Probabilidade"]), 
                    use_container_width=True, hide_index=True, 
                    column_config={"Probabilidade": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=1)}
                )
            else:
                st.warning("Dados insuficientes para antecipação por cliente.")
            
            # Pareto
            st.subheader("🏆 Pareto de Líderes")
            st.plotly_chart(px.bar(dist.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR]).update_layout(
                height=350, margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
            ), use_container_width=True)

        # Strategic Insights Footer
        st.divider()
        st.markdown(f"""
        ```python
        # Strategic Insights v4.8.0 | Perfil {ins.get('perfil').upper()} | Índice HHI: {ins.get('hhi',0):.2f} | Coef. Variação: {ins.get('cv',0):.2f}
        - Estabilidade: {ins.get('estabilidade').upper()}
        - Nota de Confiança: {ins.get('confianca', 0):.1f}% (Logic Engine Enterprise)
        ```
        """)
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros para gerar análise operacional.")