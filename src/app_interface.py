import plotly.express as px
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from datetime import timedelta

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# --- CONSTANTES DE DESIGN SISTÊMICO ---
LEGEND_COLOR = "#0F172A"
AXIS_COLOR = "#000000"
SUCCESS_COLOR = "#10B981"
DANGER_COLOR = "#EF4444"

# --- UI COMPONENTS ---

def apply_enterprise_styles():
    """
    Layout 'Strict Schema' v6.0.2.
    Estabilização de tipos de dados para evitar erros de concatenação no Polars.
    """
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
        
        [data-testid="stAppViewContainer"] {{ 
            background-color: #F8FAFC !important; 
            font-family: 'Inter', sans-serif;
        }}
        
        [data-testid="stSidebar"] {{ 
            background-color: #0F172A !important; 
            box-shadow: 5px 0 15px rgba(0,0,0,0.5);
        }}
        
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
        }}

        .header-title {{ 
            font-size: 3rem; font-weight: 900; color: {LEGEND_COLOR} !important; 
            letter-spacing: -2px; margin-bottom: 0.2rem;
        }}
        
        h3 {{
            color: {LEGEND_COLOR} !important;
            font-weight: 900 !important;
            margin-top: 2.5rem !important;
            border-left: 6px solid {THEME_COLOR};
            padding-left: 15px;
            text-transform: uppercase;
            font-size: 1.2rem;
            letter-spacing: 1px;
        }}

        .stMetric {{ 
            border: 1px solid #E2E8F0; 
            padding: 20px; 
            border-radius: 16px; 
            background-color: #FFFFFF !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }}
        
        [data-testid="stMetricValue"] {{ 
            font-weight: 900 !important; 
            color: {LEGEND_COLOR} !important;
        }}

        .insights-card {{
            background: #FFFFFF;
            border: 2px solid {LEGEND_COLOR};
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }}
        
        /* Ajuste de contraste para textos pequenos */
        .stMarkdown p, .stMarkdown span {{
            color: #1E293B !important;
            font-weight: 500;
        }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=60)
        st.markdown(f"### Black Crow\n**Intelligence v6.0.2**")
        
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
            "dias": st.slider("Janela de Faturamento (Dias):", 1, 31, (1, 31))
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
    
    # Real
    fig.add_trace(go.Scatter(
        x=v_semanal['semana'], y=v_semanal['vol'], 
        mode='lines+markers', name='Volume Real',
        line=dict(color=THEME_COLOR, width=4),
        marker=dict(size=10, color='white', line=dict(width=2, color=THEME_COLOR))
    ))
    
    # Forecast
    if not v_future.is_empty():
        last_val = v_semanal.tail(1)
        # CORREÇÃO CRÍTICA: Garantindo que os schemas batam (Date e Float64) para o concat do Polars
        hist_part = last_val.select([
            pl.col("semana").cast(pl.Date), 
            pl.col("vol").cast(pl.Float64)
        ])
        pred_part = v_future.select([
            pl.col("semana").cast(pl.Date), 
            pl.col("vol").cast(pl.Float64)
        ])
        
        conn = pl.concat([hist_part, pred_part])
        
        fig.add_trace(go.Scatter(
            x=conn['semana'], y=conn['vol'], 
            mode='lines', name='IA Forecast',
            line=dict(color=THEME_COLOR, dash='dot', width=3)
        ))
    
    # Linhas de Controle
    fig.add_hline(y=ucl, line_dash="dash", line_color=SUCCESS_COLOR, annotation_text="LSC", annotation_position="top left")
    fig.add_hline(y=m, line_dash="solid", line_color="#94A3B8", opacity=0.3)
    fig.add_hline(y=lcl, line_dash="dash", line_color=DANGER_COLOR, annotation_text="LIC", annotation_position="bottom left")

    fig.update_layout(
        template="plotly_white", height=500, margin=dict(t=30, b=30, l=40, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(color=AXIS_COLOR, family="Inter")
    )
    st.plotly_chart(fig, use_container_width=True)

def run_dashboard():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_enterprise_styles()
    
    df_raw = load_processed_data()
    if df_raw.is_empty(): 
        st.error("Erro na carga de dados.")
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
        
        # Chamada correta conforme o schema do PredictionService
        ins = PredictionService.get_strategic_insights(df)
        
        # --- HEADER ---
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748b; font-weight:700; text-transform:uppercase;'>Business Monitoring & Intelligence v6.0.2</p>", unsafe_allow_html=True)
        
        # --- ROW 1: METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Volume Total", f"{len(df):,}")
        m2.metric("Líder de Canal", dist.tail(1)['marca'][0] if not dist.is_empty() else "N/A")
        m3.metric("Coef. de Variação", f"{ins.get('cv', 0):.2f}")
        m4.metric("Acurácia IA", f"{ins.get('confianca', 0):.1f}%")
        
        st.divider()

        # --- ROW 2: SPC HERO ---
        st.markdown("<h3>📈 Estabilidade Operacional e Projeção</h3>", unsafe_allow_html=True)
        render_spc_chart(v_sem, v_fut, m_w, s_w)

        # --- ROW 3: DYNAMICS ---
        st.divider()
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("<h3>🏆 Dominância de Mercado (Share)</h3>", unsafe_allow_html=True)
            if not dist.is_empty():
                fig_share = px.bar(dist.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR], text_auto=True)
                fig_share.update_layout(template="plotly_white", height=400, margin=dict(t=10, b=10, l=100, r=20))
                st.plotly_chart(fig_share, use_container_width=True)

        with c2:
            st.markdown("<h3>📅 Sazonalidade (Heatmap)</h3>", unsafe_allow_html=True)
            try:
                df_heat = df.with_columns([
                    pl.col("data_faturamento").dt.weekday().alias("dow"),
                    pl.col("data_faturamento").dt.day().map_elements(lambda d: (d-1)//7 + 1, return_dtype=pl.Int64).alias("sem")
                ]).group_by(["sem", "dow"]).len().to_pandas()
                heat = df_heat.pivot(index="sem", columns="dow", values="len").fillna(0)
                fig_heat = px.imshow(heat, color_continuous_scale="Viridis", text_auto=True)
                fig_heat.update_layout(height=400, margin=dict(t=10, b=10))
                st.plotly_chart(fig_heat, use_container_width=True)
            except Exception:
                st.info("Aguardando massa de dados sazonal.")

        # --- ROW 4: PIPELINE NOMINAL ---
        st.divider()
        st.markdown("<h3>🔮 Pipeline de Antecipação de Vendas</h3>", unsafe_allow_html=True)
        df_p = PredictionService.get_client_predictions(df)
        if not df_p.is_empty():
            st.dataframe(
                df_p.head(20), use_container_width=True, hide_index=True,
                column_config={
                    "Valor_Est": st.column_config.ProgressColumn(
                        "Potencial de Faturamento", 
                        format="R$ %.2f", 
                        min_value=0, 
                        max_value=float(df_p["Valor_Est"].max())
                    ),
                    "Qtd_Prevista": st.column_config.NumberColumn("Volume Sugerido", format="%d un")
                }
            )
        else:
            st.warning("Massa crítica insuficiente para previsões nominais.")

        # --- FOOTER ---
        st.markdown(f"""
        <div class="insights-card">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                <div><span>PERFIL DA CARTEIRA</span><br><div style="color:{LEGEND_COLOR}; font-weight:900; font-size:1.5rem;">{ins.get('perfil').upper()}</div></div>
                <div><span>CONCENTRAÇÃO (HHI)</span><br><div style="color:{LEGEND_COLOR}; font-weight:900; font-size:1.5rem;">{ins.get('hhi',0):.3f}</div></div>
                <div><span>REGIME OPERACIONAL</span><br><div style="color:{LEGEND_COLOR}; font-weight:900; font-size:1.5rem;">{ins.get('estabilidade').upper()}</div></div>
                <div><span>VARIABILIDADE</span><br><div style="color:{LEGEND_COLOR}; font-weight:900; font-size:1.5rem;">{(ins.get('cv',0)*100):.1f}%</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Aplique os filtros na sidebar para gerar os insights.")

def set_all_state(label, options, value):
    for opt in options: st.session_state[f"chk_{label}_{opt}"] = value

if __name__ == "__main__":
    run_dashboard()