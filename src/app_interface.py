import plotly.express as px
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from datetime import timedelta

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# Definição da cor padrão de legenda solicitada
LEGEND_COLOR = "#162945"

# --- UI COMPONENTS ---

def apply_enterprise_styles():
    """
    Layout 'Midnight Platinum' v5.5.2.
    Padronização cromática total para legibilidade máxima.
    """
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
        
        /* 1. ESTRUTURA GLOBAL */
        [data-testid="stAppViewContainer"] {{ 
            background-color: #F8FAFC !important; 
            font-family: 'Inter', sans-serif;
        }}
        
        /* 2. SIDEBAR DARK LUXURY */
        [data-testid="stSidebar"] {{ 
            background-color: #0F172A !important; 
            border-right: none;
            box-shadow: 4px 0 15px rgba(0,0,0,0.1);
        }}
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {{
            color: #E2E8F0 !important;
            font-weight: 500 !important;
        }}
        
        /* Sidebar Expanders */
        .stExpander {{
            border: 1px solid rgba(255,255,255,0.1) !important;
            border-radius: 12px !important;
            background-color: rgba(255,255,255,0.03) !important;
            margin-bottom: 0.8rem !important;
        }}

        /* 3. CONTEÚDO PRINCIPAL */
        .block-container {{ 
            max-width: 94% !important; 
            padding-top: 3rem !important; 
        }}
        
        .header-title {{ 
            font-size: 2.8rem; 
            font-weight: 800; 
            color: {LEGEND_COLOR}; 
            letter-spacing: -2px;
            margin-bottom: 0.2rem;
        }}
        
        .header-subtitle {{
            font-size: 0.9rem;
            color: #64748B;
            margin-bottom: 3rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 600;
        }}

        /* 4. CARDS DE MÉTRICAS (ELEVADOS) */
        .stMetric {{ 
            border: 1px solid #FFFFFF; 
            padding: 28px; 
            border-radius: 20px; 
            background-color: #FFFFFF !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.04);
        }}
        [data-testid="stMetricValue"] {{ 
            font-size: 2.4rem !important; 
            font-weight: 800; 
            color: {LEGEND_COLOR} !important; 
            letter-spacing: -1px;
        }}
        [data-testid="stMetricLabel"] {{ 
            color: #94A3B8 !important; 
            font-weight: 700; 
            text-transform: uppercase; 
            letter-spacing: 1px; 
            font-size: 0.75rem !important; 
        }}
        
        /* 5. TABS CUSTOMIZADAS */
        .stTabs [data-baseweb="tab-list"] {{ gap: 30px; }}
        .stTabs [data-baseweb="tab"] {{
            color: #94A3B8;
            font-weight: 700;
            padding-bottom: 12px;
        }}
        .stTabs [aria-selected="true"] {{
            color: {THEME_COLOR} !important;
            border-bottom: 3px solid {THEME_COLOR} !important;
        }}

        /* 6. BOTÕES DA SIDEBAR */
        .stButton button {{ 
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: #F8FAFC !important;
            border-radius: 8px;
            font-size: 0.7rem !important;
        }}
        .stButton button:hover {{ 
            background: {THEME_COLOR} !important; 
            border-color: {THEME_COLOR} !important;
        }}

        /* 7. INSIGHTS BOX */
        .insights-card {{
            background: #FFFFFF;
            padding: 35px;
            border-radius: 24px;
            border: 1px solid #E2E8F0;
            margin-top: 3rem;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05);
        }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=60)
        st.markdown(f"<div style='margin-top: 15px; margin-bottom: 30px;'><span style='font-size: 1.6rem; font-weight: 800; color: #F8FAFC; letter-spacing: -1px;'>Black Crow</span><br><span style='color: #64748B; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;'>Intelligence Unit v5.5.2</span></div>", unsafe_allow_html=True)
        
        def smart_filter(label, col):
            opts = sorted(df[col].unique().to_list())
            with st.expander(f"Filtro: {label}"):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, opts, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, opts, False))
                return [o for o in opts if st.checkbox(o, key=f"chk_{label}_{o}", value=st.session_state.get(f"chk_{label}_{o}", True))]

        filtros = {
            "marcas": smart_filter("Empresas", "marca"),
            "paises": smart_filter("Mercados", "uf"),
            "setores": smart_filter("Segmentos", "industry_sector"),
            "dias": st.slider("Janela de Observação:", 1, 31, (1, 31))
        }
        return filtros

def render_periodicity_heatmap(df):
    """Heatmap com tipografia padronizada em #162945."""
    df_heat = df.with_columns([
        pl.col("data_faturamento").dt.weekday().alias("dow"),
        pl.col("data_faturamento").dt.day().map_elements(lambda d: (d-1)//7 + 1, return_dtype=pl.Int64).alias("semana_mes")
    ]).group_by(["semana_mes", "dow"]).len().sort(["semana_mes", "dow"])

    heat_matrix = df_heat.to_pandas().pivot(index="semana_mes", columns="dow", values="len").fillna(0)
    dias_map = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sab", 7: "Dom"}
    heat_matrix.columns = [dias_map.get(c, c) for c in heat_matrix.columns]
    heat_matrix.index = [f"Semana {i}" for i in heat_matrix.index]

    fig = px.imshow(
        heat_matrix,
        color_continuous_scale="Viridis",
        aspect="auto",
        text_auto=True
    )
    fig.update_layout(
        height=320, margin=dict(t=10, b=10, l=10, r=10), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color=LEGEND_COLOR, size=11)),
        yaxis=dict(tickfont=dict(color=LEGEND_COLOR, size=11))
    )
    st.plotly_chart(fig, use_container_width=True)

def render_spc_chart(v_semanal, v_future, m, s):
    ucl, lcl = m + 3*s, max(0, m - 3*s)
    last_vol = v_semanal.tail(1)["vol"][0]
    
    if last_vol > ucl: st.success(f"🚀 **Expansão Detectada:** Volume ({last_vol}) rompeu UCL.")
    elif last_vol < lcl: st.error(f"⚠️ **Retração Crítica:** Volume ({last_vol}) abaixo de LCL.")
    else: st.info(f"✅ **Fluxo Nominal:** Estabilidade estatística confirmada.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v_semanal['semana'], y=v_semanal['vol'], mode='lines+markers', name='Real', line=dict(color=THEME_COLOR, width=4), marker=dict(size=10, color='white', line=dict(width=3, color=THEME_COLOR))))
    
    if not v_future.is_empty():
        last_d = v_semanal.tail(1)["semana"][0]
        v_f = v_future.with_columns([pl.Series([last_d + timedelta(weeks=i+1) for i in range(len(v_future))]).alias("semana")])
        h_t = v_semanal.select(["semana", "vol"]).tail(1).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        f_t = v_f.select(["semana", "vol"]).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        conn = pl.concat([h_t, f_t])
        fig.add_trace(go.Scatter(x=conn['semana'], y=conn['vol'], mode='lines', name='Forecast', line=dict(color=THEME_COLOR, dash='dot', width=4)))
    
    fig.add_hline(y=ucl, line_dash="dash", line_color="#10B981", opacity=0.3)
    fig.add_hline(y=lcl, line_dash="dash", line_color="#EF4444", opacity=0.3)
    fig.update_layout(
        height=400, margin=dict(t=20, b=10), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', 
        showlegend=True,
        legend=dict(font=dict(color=LEGEND_COLOR)),
        xaxis=dict(showgrid=False, linecolor='#E2E8F0', tickfont=dict(color=LEGEND_COLOR, size=11)),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', zeroline=False, tickfont=dict(color=LEGEND_COLOR, size=11))
    )
    st.plotly_chart(fig, use_container_width=True)

def human_format(num):
    if num is None or num == 0: return "0"
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0: return f"{num:.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"

def set_all_state(label, options, value):
    for opt in options: st.session_state[f"chk_{label}_{opt}"] = value

def run_dashboard():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_enterprise_styles()
    df_raw = load_processed_data()
    if df_raw.is_empty(): st.error("❌ Erro na carga."); st.stop()
    
    f = render_sidebar(df_raw)
    df = apply_business_filters(df_raw, f['marcas'], f['paises'], f['dias'])
    df = df.filter(pl.col("data_faturamento") < df_raw["data_faturamento"].max())

    if not df.is_empty():
        v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        proj_vol, trend = PredictionService.get_market_trend(df)
        v_fut = PredictionService.get_daily_forecast(df, horizon=4)
        ins = PredictionService.get_strategic_insights(df)
        
        # Header
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='header-subtitle'>Market Intelligence & Predictive Analytics Engine</div>", unsafe_allow_html=True)
        
        # Scorecard
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Volume Total", f"{len(df):,}")
        k2.metric("Dominância", f"{dist.tail(1)['marca'][0][:14]}", delta=f"{(dist.tail(1)['vendas'][0]/len(df)):.1%} Share")
        k3.metric("Média Semanal", f"{m_w:.1f}")
        k4.metric("Previsibilidade", f"{ins.get('confianca', 0):.1f}%")
        k5.metric("Target Forecast", human_format(proj_vol * 125000), delta=trend)
        
        st.divider()

        cl, cr = st.columns([1.7, 1])
        with cl:
            st.markdown(f"<h3 style='color: {LEGEND_COLOR}; margin-bottom:1rem;'>Diagnóstico Dinâmico</h3>", unsafe_allow_html=True)
            t1, t2 = st.tabs(["Performance Semanal", "Periodicidade"])
            with t1:
                st.plotly_chart(px.area(v_sem, x='semana', y='vol', color_discrete_sequence=[THEME_COLOR]).update_layout(
                    height=340, margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', 
                    xaxis=dict(showgrid=False, tickfont=dict(color=LEGEND_COLOR)), 
                    yaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=None, tickfont=dict(color=LEGEND_COLOR))
                ), use_container_width=True)
            with t2:
                render_periodicity_heatmap(df)
            
            st.markdown(f"<h3 style='color: {LEGEND_COLOR}; margin-top:3rem; margin-bottom:1rem;'>Controle de Estabilidade</h3>", unsafe_allow_html=True)
            render_spc_chart(v_sem, v_fut, m_w, s_w)

        with cr:
            st.markdown(f"<h3 style='color: {LEGEND_COLOR}; margin-bottom:1rem;'>Antecipação Nominal</h3>", unsafe_allow_html=True)
            df_p = PredictionService.get_client_predictions(df)
            if not df_p.is_empty():
                st.dataframe(df_p.with_columns(pl.col("Valor_Est").map_elements(human_format, return_dtype=pl.String).alias("Valor")).select(["Cliente", "Qtd_Prevista", "Valor", "Probabilidade"]), use_container_width=True, hide_index=True, column_config={"Probabilidade": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=1, color="blue")})
            
            st.markdown(f"<h3 style='color: {LEGEND_COLOR}; margin-top:3rem; margin-bottom:1rem;'>Share de Mercado</h3>", unsafe_allow_html=True)
            st.plotly_chart(px.bar(dist.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR]).update_layout(
                height=400, margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', 
                xaxis=dict(showgrid=True, gridcolor='#F1F5F9', title=None, tickfont=dict(color=LEGEND_COLOR)), 
                yaxis=dict(showgrid=False, tickfont=dict(color=LEGEND_COLOR, size=12))
            ), use_container_width=True)

        # Insights
        st.markdown(f"""
        <div class="insights-card">
            <h3 style="margin-top:0; color:{LEGEND_COLOR}; font-size:1.3rem; letter-spacing:-0.5px;">Strategic Insights v5.5.2</h3>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; border-top: 1px solid #F1F5F9; padding-top: 25px; margin-top: 15px;">
                <div>
                    <span style="font-size:0.7rem; color:#94A3B8; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Perímetro</span><br>
                    <span style="font-size:1.1rem; color:{LEGEND_COLOR}; font-weight:800;">{ins.get('perfil').upper()}</span>
                </div>
                <div>
                    <span style="font-size:0.7rem; color:#94A3B8; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Índice HHI</span><br>
                    <span style="font-size:1.1rem; color:{LEGEND_COLOR}; font-weight:800;">{ins.get('hhi',0):.2f}</span>
                </div>
                <div>
                    <span style="font-size:0.7rem; color:#94A3B8; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Regime</span><br>
                    <span style="font-size:1.1rem; color:{LEGEND_COLOR}; font-weight:800;">{ins.get('estabilidade').upper()}</span>
                </div>
                <div>
                    <span style="font-size:0.7rem; color:#94A3B8; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Motor IA</span><br>
                    <span style="font-size:1.1rem; color:{LEGEND_COLOR}; font-weight:800;">{ins.get('confianca', 0):.1f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Filtros vazios.")