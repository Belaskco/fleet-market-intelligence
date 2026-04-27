import plotly.express as px
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from datetime import timedelta

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# Definição da cor de autoridade (Navy Profundo / Preto para Contraste Máximo)
LEGEND_COLOR = "#0F172A"
AXIS_COLOR = "#000000" # Preto absoluto para eixos e textos pequenos

# --- UI COMPONENTS ---

def apply_enterprise_styles():
    """
    Layout 'The Everything Edition' v5.9.9.
    Organiza todos os componentes em um fluxo de storytelling sem ocultar dados.
    """
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
        
        /* 1. FUNDO E FONTE GLOBAL */
        [data-testid="stAppViewContainer"] {{ 
            background-color: #F8FAFC !important; 
            font-family: 'Inter', sans-serif;
        }}
        
        /* 2. SIDEBAR - CONTRASTE MÁXIMO */
        [data-testid="stSidebar"] {{ 
            background-color: #0F172A !important; 
            box-shadow: 5px 0 15px rgba(0,0,0,0.5);
        }}
        
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] span, 
        [data-testid="stSidebar"] p {{
            color: #FFFFFF !important;
            font-weight: 800 !important;
            opacity: 1 !important;
        }}

        /* 3. CABEÇALHOS */
        .header-title {{ 
            font-size: 3rem; font-weight: 900; color: {LEGEND_COLOR} !important; 
            letter-spacing: -2px; margin-bottom: 0.2rem;
        }}
        
        h3, .stMarkdown h3, [data-testid="stMarkdownContainer"] h3 {{
            color: {LEGEND_COLOR} !important;
            font-weight: 900 !important;
            opacity: 1 !important;
            margin-top: 2rem !important;
            margin-bottom: 1.5rem !important;
        }}

        /* 4. SCORECARD (MÉTRICAS) */
        .stMetric {{ 
            border: 2px solid #CBD5E1; 
            padding: 25px; 
            border-radius: 20px; 
            background-color: #FFFFFF !important;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
            min-height: 150px;
        }}
        
        [data-testid="stMetricValue"] {{ 
            font-weight: 900 !important; 
            color: {LEGEND_COLOR} !important; 
            font-size: clamp(1.2rem, 2.2vw, 2.4rem) !important;
        }}
        
        [data-testid="stMetricLabel"] p, .stMetric label {{ 
            color: #0F172A !important; 
            font-weight: 800 !important;
            text-transform: uppercase !important;
            opacity: 1 !important;
            letter-spacing: 1.2px !important;
        }}

        /* 5. DATAFRAME */
        .stDataFrame {{ border: 2px solid #E2E8F0; border-radius: 12px; background: white; }}
        
        /* 6. INSIGHTS CARD */
        .insights-card {{
            background: #FFFFFF;
            border: 2px solid #0F172A;
            border-radius: 20px;
            padding: 40px;
            margin-top: 30px;
            box-shadow: 0 20px 40px rgba(15, 23, 42, 0.1);
        }}
        .insights-card span {{
            color: #475569 !important;
            font-weight: 900 !important;
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 1.5px;
        }}
        .insights-card .val-text {{
            color: #0F172A !important;
            font-weight: 900 !important;
            font-size: 1.6rem;
            margin-top: 5px;
        }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=70)
        st.markdown(f"<div style='margin-bottom: 30px;'><span style='font-size: 1.8rem; font-weight: 900; color: #FFFFFF; letter-spacing: -1px;'>Black Crow</span><br><span style='color: #94A3B8; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px;'>Intelligence v5.9.9</span></div>", unsafe_allow_html=True)
        
        def smart_filter(label, col):
            opts = sorted(df[col].unique().to_list())
            with st.expander(f"Filtro: {label}", expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, opts, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, opts, False))
                return [o for o in opts if st.checkbox(o, key=f"chk_{label}_{o}", value=st.session_state.get(f"chk_{label}_{o}", True))]

        filtros = {
            "marcas": smart_filter("Empresas", "marca"),
            "paises": smart_filter("UFs / Mercados", "uf"),
            "setores": smart_filter("Segmentos", "industry_sector"),
            "dias": st.slider("Recorte de Dias:", 1, 31, (1, 31))
        }
        
        st.divider()
        if st.button("🔄 RECARREGAR INTELIGÊNCIA"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
            
        return filtros

def render_periodicity_heatmap(df):
    if df.is_empty(): return
    try:
        df_heat = df.with_columns([
            pl.col("data_faturamento").dt.weekday().alias("dow"),
            pl.col("data_faturamento").dt.day().map_elements(lambda d: (d-1)//7 + 1, return_dtype=pl.Int64).alias("semana_mes")
        ]).group_by(["semana_mes", "dow"]).len().sort(["semana_mes", "dow"])
        
        heat_matrix = df_heat.to_pandas().pivot(index="semana_mes", columns="dow", values="len").fillna(0)
        dias_map = {1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sab", 7: "Dom"}
        heat_matrix.columns = [dias_map.get(c, c) for c in heat_matrix.columns]
        heat_matrix.index = [f"Semana {i}" for i in heat_matrix.index]
        
        fig = px.imshow(heat_matrix, color_continuous_scale="Viridis", aspect="auto", text_auto=True)
        fig.update_layout(
            template="plotly_white",
            height=350, margin=dict(t=10, b=10, l=40, r=20),
            paper_bgcolor='white', plot_bgcolor='white', 
            coloraxis_showscale=False,
            font=dict(color=AXIS_COLOR, size=12, family="Inter")
        )
        fig.update_xaxes(tickfont=dict(color=AXIS_COLOR, size=12))
        fig.update_yaxes(tickfont=dict(color=AXIS_COLOR, size=12))
        st.plotly_chart(fig, use_container_width=True)
    except: st.info("Massa de dados insuficiente.")

def render_spc_chart(v_semanal, v_future, m, s):
    if v_semanal.is_empty(): return
    ucl, lcl = m + 3*s, max(0, m - 3*s)
    fig = go.Figure()
    
    # Linha Real
    fig.add_trace(go.Scatter(x=v_semanal['semana'], y=v_semanal['vol'], mode='lines+markers', name='Volume Real', line=dict(color=THEME_COLOR, width=5), marker=dict(size=12, color='white', line=dict(width=3, color=THEME_COLOR))))
    
    # Projeção IA
    if not v_future.is_empty():
        last_d = v_semanal.tail(1)["semana"][0]
        v_f = v_future.with_columns([pl.Series([last_d + timedelta(weeks=i+1) for i in range(len(v_future))]).alias("semana")])
        h_t = v_semanal.select(["semana", "vol"]).tail(1).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        f_t = v_f.select(["semana", "vol"]).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        conn = pl.concat([h_t, f_t])
        fig.add_trace(go.Scatter(x=conn['semana'], y=conn['vol'], mode='lines', name='Forecast (IA)', line=dict(color=THEME_COLOR, dash='dot', width=4)))
    
    # Limites
    fig.add_hline(y=ucl, line_dash="dash", line_color="#10B981", opacity=0.8)
    fig.add_hline(y=lcl, line_dash="dash", line_color="#EF4444", opacity=0.8)
    
    fig.add_annotation(x=v_semanal['semana'][0], y=ucl, text="<b>LSC (LIMITE SUPERIOR)</b>", showarrow=False, yshift=25, font=dict(color="#10B981", size=12))
    fig.add_annotation(x=v_semanal['semana'][0], y=lcl, text="<b>LIC (LIMITE INFERIOR)</b>", showarrow=False, yshift=-25, font=dict(color="#EF4444", size=12))

    fig.update_layout(
        template="plotly_white",
        height=550, margin=dict(t=60, b=40, l=60, r=40), 
        paper_bgcolor='white', plot_bgcolor='white', 
        showlegend=True, 
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12)),
        font=dict(color=AXIS_COLOR, size=12, family="Inter")
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(color=AXIS_COLOR, size=12), linecolor=AXIS_COLOR, linewidth=2)
    fig.update_yaxes(showgrid=True, gridcolor='#F1F5F9', tickfont=dict(color=AXIS_COLOR, size=12), linecolor=AXIS_COLOR, linewidth=2)
    
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
    if df_raw.is_empty(): st.error("❌ Erro de Carga."); st.stop()
    
    f = render_sidebar(df_raw)
    df = apply_business_filters(df_raw, f['marcas'], f['paises'], f['dias'])
    max_d = df_raw["data_faturamento"].max()
    df = df.filter(pl.col("data_faturamento") < max_d)

    if not df.is_empty():
        # Motores
        v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        proj_vol, trend = PredictionService.get_market_trend(df)
        v_fut = PredictionService.get_daily_forecast(df, horizon=4)
        ins = PredictionService.get_strategic_insights(df)
        
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #475569; font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 900; margin-bottom: 2.5rem;'>Executive Market Intelligence Engine</div>", unsafe_allow_html=True)
        
        # --- ROW 1: SCORECARD ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Volume Total", f"{len(df):,}")
        if not dist.is_empty():
            lider_nome = dist.tail(1)['marca'][0]
            lider_share = (dist.tail(1)['vendas'][0]/len(df))
            k2.metric(label="Líder de Share", value=lider_nome, delta=f"{lider_share:.1%} Share")
        k3.metric("Média Semanal", f"{m_w:.1f}")
        k4.metric("Previsibilidade", f"{ins.get('confianca', 0):.1f}%")
        k5.metric("Target Forecast", human_format(proj_vol * 125000), delta=trend)
        
        st.divider()

        # --- ROW 2: ESTABILIDADE (CORE) ---
        st.markdown(f"<h3>📈 1. Controle de Estabilidade Operacional (SPC)</h3>", unsafe_allow_html=True)
        render_spc_chart(v_sem, v_fut, m_w, s_w)

        st.divider()

        # --- ROW 3: DINÂMICA DE MERCADO ---
        st.markdown(f"<h3>📊 2. Dinâmica e Concentração de Mercado</h3>", unsafe_allow_html=True)
        c_left, c_right = st.columns([1, 1.2])
        
        with c_left:
            st.markdown(f"<p style='font-weight:800; font-size:0.9rem; text-transform:uppercase;'>Principais Líderes de Share</p>", unsafe_allow_html=True)
            if not dist.is_empty():
                fig_bar = px.bar(dist.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR], text_auto=True)
                fig_bar.update_layout(
                    template="plotly_white", height=400, margin=dict(t=10, b=10, l=120, r=20),
                    paper_bgcolor='white', plot_bgcolor='white', font=dict(color=AXIS_COLOR, size=11)
                )
                fig_bar.update_xaxes(showticklabels=False, title=None)
                st.plotly_chart(fig_bar, use_container_width=True)

        with c_right:
            st.markdown(f"<p style='font-weight:800; font-size:0.9rem; text-transform:uppercase;'>Trajetória de Volume Acumulado</p>", unsafe_allow_html=True)
            fig_area = px.area(v_sem, x='semana', y='vol', color_discrete_sequence=[THEME_COLOR])
            fig_area.update_layout(
                template="plotly_white", height=400, margin=dict(t=10, b=10, l=40, r=20),
                paper_bgcolor='white', plot_bgcolor='white', font=dict(color=AXIS_COLOR, size=11)
            )
            st.plotly_chart(fig_area, use_container_width=True)

        st.divider()

        # --- ROW 4: CALENDÁRIO OPERACIONAL ---
        st.markdown(f"<h3>📅 3. Periodicidade e Janelas de Faturamento</h3>", unsafe_allow_html=True)
        render_periodicity_heatmap(df)

        st.divider()

        # --- ROW 5: EXECUÇÃO (NOMINAL) ---
        st.markdown(f"<h3>🔮 4. Antecipação Nominal de Faturamento (Próximo Ciclo)</h3>", unsafe_allow_html=True)
        df_p = PredictionService.get_client_predictions(df)
        if not df_p.is_empty():
            last_buy_stats = df.group_by("marca").agg(pl.col("data_faturamento").max().alias("ultima_compra"))
            df_view = df_p.join(last_buy_stats, left_on="Cliente", right_on="marca", how="left")
            df_final = df_view.with_columns([
                ((max_d - pl.col("ultima_compra")).dt.total_days()).alias("Recência (Dias)"),
                (pl.col("Valor_Est") / pl.col("Qtd_Prevista")).alias("Ticket Médio")
            ]).select([
                pl.col("Cliente"),
                pl.col("Qtd_Prevista").alias("Volume Previsto"),
                pl.col("Ticket Médio"),
                pl.col("Valor_Est").alias("Faturamento Est."),
                pl.col("Recência (Dias)")
            ]).sort("Faturamento Est.", descending=True).head(15)
            
            st.dataframe(
                df_final, use_container_width=True, hide_index=True, height=450,
                column_config={
                    "Ticket Médio": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Faturamento Est.": st.column_config.NumberColumn(format="R$ %.2f"),
                    "Recência (Dias)": st.column_config.NumberColumn(format="%d d")
                }
            )
        else:
            st.info("Aguardando massa crítica para projeção nominal.")

        # --- ROW 6: FECHAMENTO (STRATEGIC INSIGHTS) ---
        st.markdown(f"""
        <div class="insights-card">
            <h3 style="margin-top:0; color:{LEGEND_COLOR}; font-size:1.4rem; border-bottom: 2px solid #F1F5F9; padding-bottom:15px;">Strategic Intelligence Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 30px; margin-top:20px;">
                <div><span>ESTRUTURA</span><br><div class="val-text">{ins.get('perfil').upper()}</div></div>
                <div><span>CONCENTRAÇÃO HHI</span><br><div class="val-text">{ins.get('hhi',0):.2f}</div></div>
                <div><span>REGIME OPERACIONAL</span><br><div class="val-text">{ins.get('estabilidade').upper()}</div></div>
                <div><span>CONFIANÇA PREDITIVA</span><br><div class="val-text">{ins.get('confianca', 0):.1f}%</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros para processar a inteligência.")