import plotly.express as px
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from datetime import timedelta

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# --- CONSTANTES DE DESIGN SISTÊMICO (PERIWINKLE THEME) ---
LEGEND_COLOR = "#0F172A"
PANEL_BG = "#d1d7f0"    
AXIS_COLOR = "#1E293B"  
SUCCESS_COLOR = "#059669"
DANGER_COLOR = "#DC2626"

# --- UI COMPONENTS ---

def apply_enterprise_styles():
    """
    Layout 'Panel Identity' v6.0.6.
    Aplica fundo #d1d7f0 nos painéis e tipografia Marinho Profundo.
    """
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
        
        /* 1. FUNDO GLOBAL (PÁGINA) */
        [data-testid="stAppViewContainer"] {{ 
            background-color: #F1F5F9 !important; 
            font-family: 'Inter', sans-serif;
        }}
        
        /* 2. SIDEBAR */
        [data-testid="stSidebar"] {{ 
            background-color: #0F172A !important; 
        }}
        
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
            color: #FFFFFF !important;
            font-weight: 800 !important;
        }}

        /* 3. CABEÇALHOS */
        .header-title {{ 
            font-size: 3.5rem; font-weight: 900; color: {LEGEND_COLOR} !important; 
            letter-spacing: -2px;
        }}
        
        h3 {{
            color: {LEGEND_COLOR} !important;
            font-weight: 900 !important;
            border-left: 8px solid {THEME_COLOR};
            padding-left: 18px;
            text-transform: uppercase;
            font-size: 1.4rem !important;
            margin-bottom: 20px !important;
        }}

        /* 4. SCORECARDS (MÉTRICAS) - FUNDO #d1d7f0 --- */
        .stMetric {{ 
            border: 1px solid #B4BEDA; 
            padding: 25px; 
            border-radius: 20px; 
            background-color: {PANEL_BG} !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }}
        
        /* Valor da Métrica (Marinho Profundo) */
        [data-testid="stMetricValue"], 
        [data-testid="stMetricValue"] > div,
        [data-testid="stMetricValue"] span {{ 
            color: {LEGEND_COLOR} !important;
            font-weight: 950 !important;
            font-size: 2.6rem !important;
        }}
        
        /* Rótulo da Métrica */
        [data-testid="stMetricLabel"] p,
        [data-testid="stMetricLabel"] span,
        .stMetric label {{ 
            color: #334155 !important; 
            font-weight: 800 !important;
            text-transform: uppercase !important;
            font-size: 0.9rem !important;
        }}

        /* 5. PAINEL DE INSIGHTS - FUNDO #d1d7f0 */
        .insights-card {{
            background: {PANEL_BG};
            border: 2px solid #B4BEDA;
            border-radius: 24px;
            padding: 40px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }}
        
        .insights-card span {{
            color: #475569 !important;
            font-weight: 800 !important;
        }}
        
        .insights-card .val-text {{
            color: {LEGEND_COLOR} !important;
            font-weight: 900 !important;
        }}

        /* 6. DATAFRAME */
        .stDataFrame {{ 
            border: 2px solid #B4BEDA; 
            border-radius: 16px; 
        }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=70)
        st.markdown(f"### Black Crow\n**Intelligence v6.0.6**")
        
        def smart_filter(label, col):
            opts = sorted(df[col].unique().to_list())
            with st.expander(f"Filtro: {label}", expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, opts, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, opts, False))
                return [o for o in opts if st.checkbox(o, key=f"chk_{label}_{o}", value=st.session_state.get(f"chk_{label}_{o}", True))]

        filtros = {
            "marcas": smart_filter("Empresas", "marca"),
            "paises": smart_filter("UFs", "uf"),
            "setores": smart_filter("Segmentos", "industry_sector"),
            "dias": st.slider("Janela Mensal (Dias):", 1, 31, (1, 31))
        }
        
        if st.button("🔄 RESTART ENGINE"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
            
        return filtros

def render_spc_chart(v_semanal, v_future, m, s):
    if v_semanal.is_empty(): return
    ucl, lcl = m + 3*s, max(0, m - 3*s)
    
    fig = go.Figure()
    
    # Volume Real
    fig.add_trace(go.Scatter(
        x=v_semanal['semana'], y=v_semanal['vol'], 
        mode='lines+markers', name='<b>Real</b>',
        line=dict(color=THEME_COLOR, width=5),
        marker=dict(size=12, color='white', line=dict(width=3, color=THEME_COLOR))
    ))
    
    # Forecast
    if not v_future.is_empty():
        last_val = v_semanal.tail(1)
        hist_part = last_val.select([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        pred_part = v_future.select([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        conn = pl.concat([hist_part, pred_part])
        
        fig.add_trace(go.Scatter(
            x=conn['semana'], y=conn['vol'], 
            mode='lines', name='<b>Forecast</b>',
            line=dict(color=THEME_COLOR, dash='dot', width=4)
        ))
    
    # Limites
    fig.add_hline(y=ucl, line_dash="dash", line_color=SUCCESS_COLOR, line_width=3, 
                  annotation_text="<b>LSC</b>", annotation_position="top left", 
                  annotation_font=dict(color=SUCCESS_COLOR, size=14))
    
    fig.add_hline(y=lcl, line_dash="dash", line_color=DANGER_COLOR, line_width=3, 
                  annotation_text="<b>LIC</b>", annotation_position="bottom left",
                  annotation_font=dict(color=DANGER_COLOR, size=14))

    fig.update_layout(
        paper_bgcolor=PANEL_BG, # Fundo do painel solicitado
        plot_bgcolor=PANEL_BG,  # Fundo do gráfico solicitado
        height=550, margin=dict(t=50, b=50, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1, font=dict(size=14, color=LEGEND_COLOR)),
        font=dict(color=LEGEND_COLOR, family="Inter", size=14)
    )
    
    fig.update_xaxes(title="<b>LINHA DO TEMPO</b>", tickfont=dict(size=13, color=LEGEND_COLOR, weight="bold"), showgrid=False, linecolor=LEGEND_COLOR, linewidth=2)
    fig.update_yaxes(title="<b>VOLUME</b>", tickfont=dict(size=13, color=LEGEND_COLOR, weight="bold"), gridcolor='#B4BEDA', linecolor=LEGEND_COLOR, linewidth=2)
    
    st.plotly_chart(fig, use_container_width=True)

def run_dashboard():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_enterprise_styles()
    
    df_raw = load_processed_data()
    if df_raw.is_empty(): 
        st.error("Erro na base Parquet.")
        st.stop()
    
    f = render_sidebar(df_raw)
    df = apply_business_filters(df_raw, f['marcas'], f['paises'], f['dias'])
    
    max_date = df_raw["data_faturamento"].max()
    df = df.filter(pl.col("data_faturamento") < max_date)

    if not df.is_empty():
        # --- CALC ENGINE ---
        v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        v_fut = PredictionService.get_daily_forecast(df)
        ins = PredictionService.get_strategic_insights(df)
        
        # --- HEADER ---
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#475569; font-weight:900; text-transform:uppercase; letter-spacing:2px; font-size:1.1rem;'>Intelligence Engine • v6.0.6</p>", unsafe_allow_html=True)
        
        # --- ROW 1: METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("VOLUME TOTAL", f"{len(df):,}")
        
        lider_text = "N/A"
        if not dist.is_empty():
            lider_text = dist.tail(1).to_series(0)[0] if 'marca' not in dist.columns else dist.tail(1)['marca'][0]
        
        m2.metric("LÍDER DE CANAL", lider_text)
        m3.metric("VARIABILIDADE (CV)", f"{ins.get('cv', 0):.2f}")
        m4.metric("CONFIANÇA PREDITIVA", f"{ins.get('confianca', 0):.1f}%")
        
        st.divider()

        # --- ROW 2: SPC HERO ---
        # Container Azulado para o SPC
        st.markdown("<h3>📈 1. Análise de Estabilidade</h3>", unsafe_allow_html=True)
        render_spc_chart(v_sem, v_fut, m_w, s_w)

        # --- ROW 3: DYNAMICS ---
        st.divider()
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("<h3>🏆 2. Market Share</h3>", unsafe_allow_html=True)
            if not dist.is_empty():
                y_col = 'marca' if 'marca' in dist.columns else dist.columns[0]
                fig_share = px.bar(dist.tail(10), x='vendas', y=y_col, orientation='h', color_discrete_sequence=[THEME_COLOR], text_auto=True)
                fig_share.update_layout(
                    paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
                    height=450, margin=dict(t=10, b=10, l=120, r=20),
                    font=dict(size=13, color=LEGEND_COLOR),
                    xaxis=dict(showticklabels=False, title=None),
                    yaxis=dict(tickfont=dict(size=13, color=LEGEND_COLOR, weight="bold"), title=None)
                )
                st.plotly_chart(fig_share, use_container_width=True)

        with c2:
            st.markdown("<h3>📅 3. Sazonalidade Operacional</h3>", unsafe_allow_html=True)
            try:
                df_heat = df.with_columns([
                    pl.col("data_faturamento").dt.weekday().alias("dow"),
                    pl.col("data_faturamento").dt.day().map_elements(lambda d: (d-1)//7 + 1, return_dtype=pl.Int64).alias("sem")
                ]).group_by(["sem", "dow"]).len().to_pandas()
                heat = df_heat.pivot(index="sem", columns="dow", values="len").fillna(0)
                fig_heat = px.imshow(heat, color_continuous_scale="Viridis", text_auto=True)
                fig_heat.update_layout(
                    paper_bgcolor=PANEL_BG, plot_bgcolor=PANEL_BG,
                    height=450, margin=dict(t=10, b=10), font=dict(size=13, color=LEGEND_COLOR)
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            except Exception:
                st.info("Aguardando massa de dados...")

        # --- ROW 4: PIPELINE NOMINAL ---
        st.divider()
        st.markdown("<h3>🔮 4. Antecipação de Faturamento Nominal</h3>", unsafe_allow_html=True)
        df_p = PredictionService.get_client_predictions(df)
        if not df_p.is_empty():
            st.dataframe(
                df_p.head(20), use_container_width=True, hide_index=True,
                column_config={
                    "Valor_Est": st.column_config.ProgressColumn(
                        "POTENCIAL FINANCEIRO", 
                        format="R$ %.2f", 
                        min_value=0, 
                        max_value=float(df_p["Valor_Est"].max())
                    ),
                    "Qtd_Prevista": st.column_config.NumberColumn("VOLUME", format="%d un")
                }
            )
        else:
            st.warning("Aguardando massa crítica.")

        # --- FOOTER: ESTRATÉGICO ---
        st.markdown(f"""
        <div class="insights-card">
            <h4 style="margin-top:0; color:{LEGEND_COLOR}; font-weight:900; text-transform:uppercase; border-bottom:2px solid #B4BEDA; padding-bottom:15px;">Executive Strategy Summary</h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 30px; margin-top:25px;">
                <div><span style="font-size:0.8rem;">ESTRUTURA</span><br><div class="val-text">{ins.get('perfil').upper()}</div></div>
                <div><span style="font-size:0.8rem;">HHI INDEX</span><br><div class="val-text">{ins.get('hhi',0):.3f}</div></div>
                <div><span style="font-size:0.8rem;">ESTADO OPERACIONAL</span><br><div class="val-text">{ins.get('estabilidade').upper()}</div></div>
                <div><span style="font-size:0.8rem;">VOLATILIDADE (CV)</span><br><div class="val-text">{(ins.get('cv',0)*100):.1f}%</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Aguardando seleção de filtros.")

def set_all_state(label, options, value):
    for opt in options: st.session_state[f"chk_{label}_{opt}"] = value

if __name__ == "__main__":
    run_dashboard()