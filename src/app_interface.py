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
    # Aplica o DNA visual da Black Crow para um visual executivo de alto nível.
    st.markdown(f"""
        <style>
        /* 1. FUNDO DA PÁGINA (SOFT GRAY) */
        [data-testid="stAppViewContainer"] {{
            background-color: #F1F5F9;
        }}

        /* FUNDO DA SIDEBAR */
        [data-testid="stSidebar"] {{
            background-color: #FFFFFF;
            border-right: 1px solid #CBD5E1;
        }}

        /* 2. LARGURA DO CONTEÚDO */
        .block-container {{
            max-width: 98% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 1rem !important;
        }}

        /* 3. DESIGN DOS PAINÉIS / CARDS (ELEVATION) */
        .stMetric {{ 
            border: 1px solid #CBD5E1; 
            padding: 20px; 
            border-radius: 10px; 
            background-color: #FFFFFF;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        /* CORREÇÃO DE TEXTO NAS MÉTRICAS */
        [data-testid="stMetricValue"] {{ 
            font-size: 1.8rem !important; 
            font-weight: 800; 
            color: #0F172A !important; 
        }}
        
        [data-testid="stMetricLabel"] {{
            color: #475569 !important;
            font-weight: 600 !important;
        }}

        /* BOTÕES PROFISSIONAIS */
        .stButton button {{ 
            width: 100%; 
            font-weight: 700; 
            text-transform: uppercase; 
            border-radius: 6px;
            background-color: #FFFFFF;
            border: 1px solid #94A3B8;
            color: #1E293B;
            transition: all 0.2s;
        }}
        
        .stButton button:hover {{
            border-color: {THEME_COLOR};
            color: {THEME_COLOR};
            background-color: #F8FAFC;
        }}
        
        /* HEADER EXECUTIVO */
        .header-title {{ 
            font-size: 2.2rem; 
            font-weight: 800; 
            color: #0F172A; 
            border-bottom: 4px solid {THEME_COLOR}; 
            padding-bottom: 10px; 
            margin-bottom: 25px; 
            letter-spacing: -0.5px;
        }}

        /* SUBHEADERS */
        h3, h2, h1 {{
            color: #1E293B !important;
            font-weight: 700 !important;
        }}

        /* TEXTO GERAL E TABELAS */
        p, span, label {{
            color: #1E293B !important;
        }}
        
        .section-header {{
            color: #475569;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            margin-top: 1.2rem;
            margin-bottom: 0.4rem;
        }}

        /* CORREÇÃO PARA DATAFRAMES (EVITAR TEXTO INVISÍVEL) */
        .stDataFrame div {{
            color: #0F172A !important;
        }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    """Gerencia todos os filtros de segmentação e controles na barra lateral."""
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=90)
        st.title("Market Control")
        st.caption("Engine: Black Crow v4.8 | Logic: Weekly Cycle")
        st.divider()
        
        def smart_filter(label, col):
            opts = sorted(df[col].unique().to_list())
            with st.expander(f"Filtro: {label}", expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, opts, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, opts, False))
                return [o for o in opts if st.checkbox(o, key=f"chk_{label}_{o}", value=st.session_state.get(f"chk_{label}_{o}", True))]

        # Mapeamento dinâmico baseado na inspeção do Parquet
        id_col = "company_name" if "company_name" in df.columns else "marca"
        geo_col = "hq_country" if "hq_country" in df.columns else "uf"
        
        filtros = {
            "marcas": smart_filter("Marcas / Empresas", id_col),
            "paises": smart_filter("Países / Mercados", geo_col),
            "setores": smart_filter("Setores Industriais", "industry_sector"),
            "dias": st.slider("Recorte por Dia do Mês:", 1, 31, (1, 31))
        }
        st.info("💡 Filtros D-1 aplicados automaticamente para garantir integridade dos dados.")
        return filtros

def render_spc_chart(v_semanal, v_future, m, s):
    """Renderiza a Carta de Controle com Forecast Pontilhado e Alertas Estatísticos."""
    ucl, lcl = m + 2*s, max(0, m - 2*s)
    
    # 1. Alerta de Execução Real (Topo)
    last_vol = v_semanal.tail(1)["vol"][0]
    if last_vol > ucl: 
        st.success(f"🚀 **Expansão Crítica:** Último ciclo ({last_vol}) rompeu o limite superior UCL ({ucl:.1f}).")
    elif last_vol < lcl: 
        st.error(f"⚠️ **Retração Crítica:** Último ciclo ({last_vol}) abaixo do limite inferior LCL ({lcl:.1f}).")
    else: 
        st.info(f"✅ **Estabilidade:** Fluxo operacional em regime nominal ({last_vol} pedidos/sem).")

    # 2. Gráfico Plotly SPC
    fig = go.Figure()
    # Série Histórica Real (Sólida)
    fig.add_trace(go.Scatter(
        x=v_semanal['semana'], y=v_semanal['vol'], 
        mode='lines+markers', name='Histórico', 
        line=dict(color=THEME_COLOR, width=3),
        marker=dict(size=8)
    ))
    
    # Série Preditiva Nixtla (Pontilhada)
    if not v_future.is_empty():
        last_date = v_semanal.tail(1)["semana"][0]
        
        # Sincronização de Timeline para o Forecast
        v_fut_dates = v_future.with_columns([
            pl.Series([last_date + timedelta(weeks=i+1) for i in range(len(v_future))]).alias("semana")
        ])
        
        # BLINDAGEM DE SCHEMA: Garante que ambos os DataFrames tenham as mesmas colunas e tipos antes do concat
        # O erro acontecia porque o histórico é Int e o Forecast é Float.
        hist_tail = v_semanal.select(["semana", "vol"]).tail(1).with_columns([
            pl.col("semana").cast(pl.Date), 
            pl.col("vol").cast(pl.Float64)
        ])
        
        fut_tail = v_fut_dates.select(["semana", "vol"]).with_columns([
            pl.col("semana").cast(pl.Date), 
            pl.col("vol").cast(pl.Float64)
        ])
        
        # Agora a concatenação é segura entre tipos idênticos
        conn = pl.concat([hist_tail, fut_tail])
        
        fig.add_trace(go.Scatter(
            x=conn['semana'], y=conn['vol'], 
            mode='lines', name='Forecast', 
            line=dict(color=THEME_COLOR, dash='dot', width=3)
        ))
    
    # Limites Estáticos (SPC)
    fig.add_hline(y=ucl, line_dash="dash", line_color="#238636", annotation_text="UCL (Limite Sup.)")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#da3633", annotation_text="LCL (Limite Inf.)")
    
    fig.update_layout(
        height=350, 
        margin=dict(t=10, b=10), 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(128,128,128,0.02)', 
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(128,128,128,0.1)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Alerta Preditivo (Base)
    if not v_future.is_empty() and v_future.filter(pl.col("vol") > ucl).height > 0:
        st.warning("🔮 **Alerta Preditivo:** O motor detectou possíveis picos de demanda nas próximas 4 semanas.")

def render_scorecard(total, m_week, share_lider, confianca, forecast_val, trend):
    """Exibe a grade de KPIs de nível executivo no topo do painel."""
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Volume Acumulado", f"{total:,}", help="Total de pedidos faturados no período (Excluindo D-0).")
    k2.metric("Líder de Canal", f"{share_lider[0][:12]}", delta=f"{share_lider[1]:.1%} Share")
    k3.metric("Média Semanal", f"{m_week:.1f} un/sem")
    k4.metric("Previsibilidade", f"{confianca:.1f}%", help="Logic Engine v4.0 (Calibração Enterprise)")
    k5.metric("Forecast Próx. Ciclo", human_format(forecast_val * 125000), delta=trend)

# --- UTILS & CORE ENGINE ---

def set_all_state(label, options, value):
    """Controlador de estado para botões de filtro em massa."""
    for opt in options: 
        st.session_state[f"chk_{label}_{opt}"] = value

def human_format(num):
    """Formata valores financeiros para escala executiva (K, M, B)."""
    if num is None or num == 0: return "0"
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0: return f"{num:.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"

def run_dashboard():
    # Maestro que orquestra todos os componentes do Dashboard
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_enterprise_styles()

    df_raw = load_processed_data()
    if df_raw.is_empty(): 
        st.error("❌ Base de dados Fleet não carregada corretamente."); st.stop()

    date_col = "purchase_date" if "purchase_date" in df_raw.columns else "data_faturamento"
    f = render_sidebar(df_raw)
    
    # Processamento de Filtros e Blindagem D-1
    df = apply_business_filters(df_raw, f['marcas'], f['paises'], f['dias'])
    if f['setores']: 
        df = df.filter(pl.col("industry_sector").is_in(f['setores']))
    
    # Blindagem D-1 (Remoção do dia atual incompleto)
    df = df.filter(pl.col(date_col) < df_raw[date_col].max()) 

    if not df.is_empty():
        # --- CÁLCULOS E MOTORES DE IA ---
        v_sem = df.with_columns(pl.col(date_col).dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        proj_vol, trend = PredictionService.get_market_trend(df)
        v_fut = PredictionService.get_daily_forecast(df, horizon=4)
        ins = PredictionService.get_strategic_insights(df)
        
        # Header Principal
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        
        # Top Metrics
        lider_info = (dist.tail(1)['marca'][0], dist.tail(1)['vendas'][0]/len(df))
        render_scorecard(len(df), m_w, lider_info, ins.get('confianca', 0), proj_vol, trend)
        st.divider()

        # Layout Principal (Grid)
        c_l, c_r = st.columns([1.6, 1])
        
        with c_l:
            st.subheader("📊 Diagnóstico de Trajetória Semanal")
            # Histórico de Área
            fig_area = px.area(v_sem, x='semana', y='vol', color_discrete_sequence=[THEME_COLOR])
            fig_area.update_layout(
                height=280, margin=dict(t=0,b=0), 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title=None, yaxis_title="Volume"
            )
            st.plotly_chart(fig_area, use_container_width=True)
            
            # Carta de Controle (SPC)
            st.markdown("<div class='section-header'>📈 Controle Estatístico e Estabilidade</div>", unsafe_allow_html=True)
            render_spc_chart(v_sem, v_fut, m_w, s_w)

        with c_r:
            # --- TABELA DE ANTECIPAÇÃO NOMINAL (NIXTLA) ---
            st.subheader("🔮 Antecipação Nominal (Próx. Ciclo)")
            df_p = PredictionService.get_client_predictions(df)
            
            if not df_p.is_empty():
                # Aplicamos formatação financeira na tabela preditiva
                df_view = df_p.with_columns(
                    pl.col("Valor_Est").map_elements(human_format, return_dtype=pl.String).alias("Potencial Est.")
                )
                st.dataframe(
                    df_view.select(["Cliente", "Qtd_Prevista", "Potencial Est.", "Probabilidade"]), 
                    use_container_width=True, 
                    hide_index=True,
                    column_config={
                        "Probabilidade": st.column_config.ProgressColumn(
                            "Confiança", min_value=0, max_value=1, format="%.2f", color="green"
                        ),
                        "Qtd_Prevista": st.column_config.NumberColumn("Previsão (un)")
                    }
                )
            else:
                st.warning("Volume de dados insuficiente para gerar antecipação por cliente nesta segmentação.")
            
            # Pareto de Líderes
            st.markdown("<div class='section-header'>🏆 Pareto de Líderes (Market Share)</div>", unsafe_allow_html=True)
            fig_bar = px.bar(
                dist.tail(10), x='vendas', y='marca', 
                orientation='h', color_discrete_sequence=[THEME_COLOR]
            )
            fig_bar.update_layout(
                height=350, margin=dict(t=0,b=0), 
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Volume", yaxis_title=None
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Strategic Insights Footer (Logic Engine)
        st.divider()
        st.markdown(f"""
        ```python
        # Strategic Insights v4.8.0 | Perfil {ins.get('perfil').upper()} | Índice HHI: {ins.get('hhi',0):.2f} | Coef. Variação: {ins.get('cv',0):.2f}
        - Estabilidade: {ins.get('estabilidade').upper()}
        - Nota de Confiança: {ins.get('confianca', 0):.1f}% (Nível Profissional Enterprise)
        - Fonte: Black Crow Intelligence Unit
        ```
        """)
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros para gerar análise operacional.")