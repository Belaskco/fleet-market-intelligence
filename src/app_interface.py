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
    Layout 'Visibility Protocol' v5.9.6.
    Força legibilidade absoluta e remove qualquer transparência de fontes.
    Correção de erros de compatibilidade com Plotly Engine.
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
        }}

        /* 4. SCORECARD (MÉTRICAS) */
        .stMetric {{ 
            border: 2px solid #CBD5E1; 
            padding: 25px; 
            border-radius: 20px; 
            background-color: #FFFFFF !important;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.05);
            min-height: 180px;
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

        /* 5. TABS E DATAFRAME */
        .stTabs [data-baseweb="tab-list"] {{ gap: 24px; }}
        .stTabs [data-baseweb="tab"] {{ color: #1E293B !important; font-weight: 900 !important; opacity: 1 !important; }}
        .stTabs [aria-selected="true"] {{ color: {THEME_COLOR} !important; border-bottom: 4px solid {THEME_COLOR} !important; }}
        .stDataFrame {{ border: 2px solid #E2E8F0; border-radius: 12px; background: white; }}
        
        /* 6. INSIGHTS CARD */
        .insights-card {{
            background: white;
            border: 2px solid #E2E8F0;
            border-radius: 20px;
            padding: 35px;
        }}
        .insights-card span {{
            color: #0F172A !important;
            font-weight: 900 !important;
            opacity: 1 !important;
        }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=70)
        st.markdown(f"<div style='margin-bottom: 30px;'><span style='font-size: 1.8rem; font-weight: 900; color: #FFFFFF; letter-spacing: -1px;'>Black Crow</span><br><span style='color: #94A3B8; font-size: 0.75rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1.5px;'>Intelligence v5.9.6</span></div>", unsafe_allow_html=True)
        
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
        if st.button("🔄 LIMPAR TUDO E REINICIAR"):
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
            height=400, margin=dict(t=30, b=30, l=40, r=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
            coloraxis_showscale=False,
            font=dict(color=AXIS_COLOR, size=12, family="Inter")
        )
        # Forçar visibilidade dos eixos com PRETO ABSOLUTO
        fig.update_xaxes(tickfont=dict(color=AXIS_COLOR, size=12))
        fig.update_yaxes(tickfont=dict(color=AXIS_COLOR, size=12))
        st.plotly_chart(fig, use_container_width=True)
    except: st.info("Massa de dados insuficiente.")

def render_spc_chart(v_semanal, v_future, m, s):
    if v_semanal.is_empty(): return
    ucl, lcl = m + 3*s, max(0, m - 3*s)
    fig = go.Figure()
    
    # Linha Real
    fig.add_trace(go.Scatter(x=v_semanal['semana'], y=v_semanal['vol'], mode='lines+markers', name='Volume Real', line=dict(color=THEME_COLOR, width=4), marker=dict(size=10, color='white', line=dict(width=3, color=THEME_COLOR))))
    
    # Projeção Nixtla
    if not v_future.is_empty():
        last_d = v_semanal.tail(1)["semana"][0]
        v_f = v_future.with_columns([pl.Series([last_d + timedelta(weeks=i+1) for i in range(len(v_future))]).alias("semana")])
        h_t = v_semanal.select(["semana", "vol"]).tail(1).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        f_t = v_f.select(["semana", "vol"]).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        conn = pl.concat([h_t, f_t])
        fig.add_trace(go.Scatter(x=conn['semana'], y=conn['vol'], mode='lines', name='Forecast (IA)', line=dict(color=THEME_COLOR, dash='dot', width=4)))
    
    # Limites Estatísticos
    fig.add_hline(y=ucl, line_dash="dash", line_color="#10B981", opacity=0.8)
    fig.add_hline(y=lcl, line_dash="dash", line_color="#EF4444", opacity=0.8)
    
    # Anotações para LSC e LIC - Texto em negrito via tags HTML
    fig.add_annotation(x=v_semanal['semana'][0], y=ucl, text="<b>LIMITE SUPERIOR (LSC)</b>", showarrow=False, yshift=20, font=dict(color="#10B981", size=11))
    fig.add_annotation(x=v_semanal['semana'][0], y=lcl, text="<b>LIMITE INFERIOR (LIC)</b>", showarrow=False, yshift=-20, font=dict(color="#EF4444", size=11))

    fig.update_layout(
        template="plotly_white",
        height=450, margin=dict(t=40, b=40, l=40, r=20), 
        paper_bgcolor='white', plot_bgcolor='white', 
        showlegend=True, 
        legend=dict(
            font=dict(color=AXIS_COLOR, size=12),
            bgcolor="white", 
            bordercolor=AXIS_COLOR, 
            borderwidth=2,
            opacity=1
        ),
        font=dict(color=AXIS_COLOR, size=12, family="Inter")
    )
    # Forçar visibilidade total dos eixos com PRETO
    fig.update_xaxes(showgrid=False, tickfont=dict(color=AXIS_COLOR, size=12), linecolor=AXIS_COLOR, linewidth=2)
    fig.update_yaxes(showgrid=True, gridcolor='#E2E8F0', tickfont=dict(color=AXIS_COLOR, size=12), linecolor=AXIS_COLOR, linewidth=2)
    
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
        v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        proj_vol, trend = PredictionService.get_market_trend(df)
        v_fut = PredictionService.get_daily_forecast(df, horizon=4)
        ins = PredictionService.get_strategic_insights(df)
        
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color: #475569; font-size: 1rem; text-transform: uppercase; letter-spacing: 2px; font-weight: 900; margin-bottom: 2.5rem;'>Executive Market Intelligence Engine</div>", unsafe_allow_html=True)
        
        # --- SCORECARD ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Volume Total", f"{len(df):,}")
        
        if not dist.is_empty():
            lider_nome = dist.tail(1)['marca'][0]
            lider_share = (dist.tail(1)['vendas'][0]/len(df))
            k2.metric(
                label="Líder de Share",
                value=lider_nome,
                delta=f"{lider_share:.1%} Share",
                help=f"**IDENTIFICAÇÃO COMPLETA:**\n\n{lider_nome}" 
            )
        
        k3.metric("Média Semanal", f"{m_w:.1f}")
        k4.metric("Previsibilidade", f"{ins.get('confianca', 0):.1f}%")
        k5.metric("Target Forecast", human_format(proj_vol * 125000), delta=trend)
        
        st.divider()

        # --- SEÇÃO 1: ANTECIPAÇÃO NOMINAL ---
        st.markdown(f"<h3>🔮 Antecipação Nominal de Faturamento</h3>", unsafe_allow_html=True)
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
                pl.col("Valor_Est").alias("Venda Estimada"),
                pl.col("Recência (Dias)")
            ]).sort("Venda Estimada", descending=True)
            st.dataframe(df_final, use_container_width=True, hide_index=True, height=450)
        else:
            st.info("Aguardando massa crítica para projeção.")

        st.divider()

        # --- SEÇÃO 2: DIAGNÓSTICO ---
        st.markdown(f"<h3>📊 Diagnóstico Dinâmico de Performance</h3>", unsafe_allow_html=True)
        tab_area, tab_heat = st.tabs(["Trajetória Histórica", "Periodicidade"])
        with tab_area:
            fig_area = px.area(v_sem, x='semana', y='vol', color_discrete_sequence=[THEME_COLOR])
            fig_area.update_layout(
                template="plotly_white",
                height=500, margin=dict(t=10,b=10), 
                paper_bgcolor='white', plot_bgcolor='white', 
                font=dict(color=AXIS_COLOR, size=12, family="Inter")
            )
            fig_area.update_xaxes(tickfont=dict(color=AXIS_COLOR), linecolor=AXIS_COLOR)
            fig_area.update_yaxes(tickfont=dict(color=AXIS_COLOR), linecolor=AXIS_COLOR)
            st.plotly_chart(fig_area, use_container_width=True)
        with tab_heat:
            render_periodicity_heatmap(df)

        st.divider()

        # --- SEÇÃO 3: ESTABILIDADE E SHARE ---
        c_left, c_right = st.columns([1.5, 1])
        with c_left:
            st.markdown(f"<h3>📈 Controle de Estabilidade (SPC)</h3>", unsafe_allow_html=True)
            render_spc_chart(v_sem, v_fut, m_w, s_w)
        with c_right:
            st.markdown(f"<h3>🏆 Share de Mercado (Líderes)</h3>", unsafe_allow_html=True)
            if not dist.is_empty():
                fig_bar = px.bar(dist.tail(12), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR], text_auto=True)
                fig_bar.update_layout(
                    template="plotly_white",
                    height=450, margin=dict(t=20, b=20, l=160, r=20), 
                    paper_bgcolor='white', plot_bgcolor='white', 
                    font=dict(color=AXIS_COLOR, size=13)
                )
                fig_bar.update_xaxes(tickfont=dict(color=AXIS_COLOR), linecolor=AXIS_COLOR)
                fig_bar.update_yaxes(tickfont=dict(color=AXIS_COLOR), linecolor=AXIS_COLOR)
                st.plotly_chart(fig_bar, use_container_width=True)

        # --- RODAPÉ ---
        st.markdown(f"""
        <div class="insights-card">
            <h3 style="margin-top:0; color:{LEGEND_COLOR}; font-size:1.4rem;">Strategic Insights v5.9.6</h3>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; border-top: 2px solid #F1F5F9; padding-top: 25px; margin-top:15px;">
                <div><span>PERÍMETRO</span><br><div class="val-text">{ins.get('perfil').upper()}</div></div>
                <div><span>ÍNDICE HHI</span><br><div class="val-text">{ins.get('hhi',0):.2f}</div></div>
                <div><span>REGIME</span><br><div class="val-text">{ins.get('estabilidade').upper()}</div></div>
                <div><span>CONFIANÇA IA</span><br><div class="val-text">{ins.get('confianca', 0):.1f}%</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros para processar a inteligência.")