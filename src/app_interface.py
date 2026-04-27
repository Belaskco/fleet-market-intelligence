import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import polars as pl

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR
from datetime import timedelta

# --- UI COMPONENTS ---

def apply_enterprise_styles():
    # Layout
    
    st.markdown(f"""
        <style>
        /* 1. FUNDO E LAYOUT */
        [data-testid="stAppViewContainer"] {{ background-color: #F1F5F9 !important; }}
        .block-container {{ max-width: 96% !important; padding-top: 3rem !important; }}

        /* 2. SIDEBAR - CORREÇÃO DE CONTRASTE TOTAL */
        [data-testid="stSidebar"] {{ 
            background-color: #FFFFFF !important; 
            border-right: 2px solid #E2E8F0; 
        }}
        /* Força textos, labels e expanders da Sidebar a serem escuros */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] summary p,
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] span {{
            color: #0F172A !important;
            font-weight: 600 !important;
        }}
        [data-testid="stSidebar"] .stExpander {{
            background-color: #F8FAFC !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 8px;
        }}

        /* 3. CARDS DE MÉTRICAS */
        .stMetric {{ 
            border: 1px solid #CBD5E1; 
            padding: 22px; 
            border-radius: 12px; 
            background-color: #FFFFFF !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }}
        [data-testid="stMetricValue"] {{ font-size: 2rem !important; font-weight: 800; color: #0F172A !important; }}
        [data-testid="stMetricLabel"] {{ color: #475569 !important; font-weight: 700; }}
        
        /* HEADER */
        .header-title {{ 
            font-size: 2.3rem; font-weight: 900; color: #0F172A; 
            border-bottom: 5px solid {THEME_COLOR}; padding-bottom: 12px; margin-bottom: 30px; 
        }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    # Sidebar com filtros.
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=85)
        st.markdown("<h2 style='color:#0F172A; font-weight:900;'>Black Crow</h2>", unsafe_allow_html=True)
        st.caption("Intelligence Unit | v5.3.0")
        st.divider()
        
        def smart_filter(label, col):
            opts = sorted(df[col].unique().to_list())
            with st.expander(f"Filtro: {label}"):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, opts, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, opts, False))
                return [o for o in opts if st.checkbox(o, key=f"chk_{label}_{o}", value=st.session_state.get(f"chk_{label}_{o}", True))]

        return {
            "marcas": smart_filter("Marcas", "marca"),
            "paises": smart_filter("Geografia", "uf"),
            "setores": smart_filter("Setores", "industry_sector"),
            "dias": st.slider("Recorte Mensal (Dias):", 1, 31, (1, 31))
        }

def render_periodicity_heatmap(df):
    
    # Preparação dos dados para o Heatmap
    df_heat = df.with_columns([
        pl.col("data_faturamento").dt.weekday().alias("dow"),
        pl.col("data_faturamento").dt.day().map_elements(lambda d: (d-1)//7 + 1, return_dtype=pl.Int64).alias("semana_mes")
    ]).group_by(["semana_mes", "dow"]).len().sort(["semana_mes", "dow"])

    # Pivot para o formato de matriz do Plotly
    heat_matrix = df_heat.to_pandas().pivot(index="semana_mes", columns="dow", values="len").fillna(0)
    
    # Nomes dos dias para o eixo X
    dias_map = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sab", 7: "Dom"}
    heat_matrix.columns = [dias_map.get(c, c) for c in heat_matrix.columns]
    heat_matrix.index = [f"Semana {i}" for i in heat_matrix.index]

    fig = px.imshow(
        heat_matrix,
        labels=dict(x="Dia da Semana", y="Semana do Mês", color="Volume"),
        color_continuous_scale="RdBu_r", # coolwarm invertido (Vermelho=Quente/Alto)
        aspect="auto",
        text_auto=True
    )
    fig.update_layout(height=300, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def render_spc_chart(v_semanal, v_future, m, s):
    ucl, lcl = m + 3*s, max(0, m - 3*s)
    last_vol = v_semanal.tail(1)["vol"][0]
    
    if last_vol > ucl: st.success(f"🚀 **Alerta:** Volume ({last_vol}) acima do limite superior.")
    elif last_vol < lcl: st.error(f"⚠️ **Alerta:** Volume ({last_vol}) abaixo do limite inferior.")
    else: st.info(f"✅ **Estabilidade:** Fluxo operacional nominal.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=v_semanal['semana'], y=v_semanal['vol'], mode='lines+markers', name='Real', line=dict(color=THEME_COLOR, width=3)))
    
    if not v_future.is_empty():
        last_d = v_semanal.tail(1)["semana"][0]
        v_f = v_future.with_columns([pl.Series([last_d + timedelta(weeks=i+1) for i in range(len(v_future))]).alias("semana")])
        h_t = v_semanal.select(["semana", "vol"]).tail(1).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        f_t = v_f.select(["semana", "vol"]).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        conn = pl.concat([h_t, f_t])
        fig.add_trace(go.Scatter(x=conn['semana'], y=conn['vol'], mode='lines', name='Forecast', line=dict(color=THEME_COLOR, dash='dot', width=3)))
    
    fig.add_hline(y=ucl, line_dash="dash", line_color="#10B981", annotation_text="UCL")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#EF4444", annotation_text="LCL")
    fig.update_layout(height=380, margin=dict(t=30, b=10), plot_bgcolor='white', showlegend=False)
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
        # Motores
        v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        proj_vol, trend = PredictionService.get_market_trend(df)
        v_fut = PredictionService.get_daily_forecast(df, horizon=4)
        ins = PredictionService.get_strategic_insights(df)
        
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        
        # KPI Scorecard
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Volume Acumulado", f"{len(df):,}")
        k2.metric("Líder de Canal", f"{dist.tail(1)['marca'][0][:14]}", delta=f"{(dist.tail(1)['vendas'][0]/len(df)):.1%} Share")
        k3.metric("Média Semanal", f"{m_w:.1f} un/sem")
        k4.metric("Previsibilidade", f"{ins.get('confianca', 0):.1f}%", help="Logic Engine v5.0")
        k5.metric("Forecast Próx. Ciclo", human_format(proj_vol * 125000), delta=trend)
        
        st.divider()

        cl, cr = st.columns([1.6, 1])
        with cl:
            st.subheader("📊 Trajetória e Periodicidade")
            tab1, tab2 = st.tabs(["Fluxo Semanal", "Mapa de Calor Sazonal"])
            with tab1:
                st.plotly_chart(px.area(v_sem, x='semana', y='vol', color_discrete_sequence=[THEME_COLOR]).update_layout(height=280, margin=dict(t=0,b=0), plot_bgcolor='white'), use_container_width=True)
            with tab2:
                render_periodicity_heatmap(df)
            
            st.markdown("<h3 style='margin-top:20px;'>📈 Estabilidade e Controle Estatístico</h3>", unsafe_allow_html=True)
            render_spc_chart(v_sem, v_fut, m_w, s_w)

        with cr:
            st.subheader("🔮 Antecipação Nominal (Nixtla)")
            df_p = PredictionService.get_client_predictions(df)
            if not df_p.is_empty():
                st.dataframe(df_p.with_columns(pl.col("Valor_Est").map_elements(human_format, return_dtype=pl.String).alias("Potencial")).select(["Cliente", "Qtd_Prevista", "Potencial", "Probabilidade"]), use_container_width=True, hide_index=True, column_config={"Probabilidade": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=1, color="green")})
            
            st.markdown("<h3 style='margin-top:25px;'>🏆 Mix de Liderança</h3>", unsafe_allow_html=True)
            st.plotly_chart(px.bar(dist.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR]).update_layout(height=380, margin=dict(t=0,b=0), plot_bgcolor='white'), use_container_width=True)

        st.divider()
        st.markdown(f"""
        <div style="background-color:white; padding:25px; border:1px solid #E2E8F0; border-radius:12px; color:#0F172A;">
            <strong>Strategic Insights v5.3.0</strong> | Perfil: {ins.get('perfil').upper()} | HHI: {ins.get('hhi',0):.2f} | CV: {ins.get('cv',0):.2f}<br>
            Estabilidade: {ins.get('estabilidade').upper()} | Confiança do Forecast: {ins.get('confianca', 0):.1f}%
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros.")