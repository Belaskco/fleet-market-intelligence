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
    Layout 'The Final Review' v6.0.0.
    Otimização de contraste, tipografia e blindagem de componentes.
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
            border-left: 5px solid {THEME_COLOR};
            padding-left: 15px;
        }}

        .stMetric {{ 
            border: 1px solid #E2E8F0; 
            padding: 20px; 
            border-radius: 16px; 
            background-color: #FFFFFF !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=60)
        st.markdown(f"### Black Crow\n**Intelligence v6.0.0**")
        
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
            "dias": st.slider("Recorte Mensal (Dias):", 1, 31, (1, 31))
        }
        
        if st.button("🔄 REFRESH ENGINE"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
            
        return filtros

def render_spc_chart(v_semanal, v_future, m, s):
    """Gera o Gráfico de Estabilidade com anotações dinâmicas."""
    if v_semanal.is_empty(): return
    ucl, lcl = m + 3*s, max(0, m - 3*s)
    
    fig = go.Figure()
    
    # Volume Real
    fig.add_trace(go.Scatter(
        x=v_semanal['semana'], y=v_semanal['vol'], 
        mode='lines+markers', name='Real',
        line=dict(color=THEME_COLOR, width=4),
        marker=dict(size=10, color='white', line=dict(width=2, color=THEME_COLOR))
    ))
    
    # Forecast
    if not v_future.is_empty():
        last_val = v_semanal.tail(1)
        conn = pl.concat([last_val.select(["semana", "vol"]), v_future.select(["semana", "vol"])])
        fig.add_trace(go.Scatter(
            x=conn['semana'], y=conn['vol'], 
            mode='lines', name='Forecast IA',
            line=dict(color=THEME_COLOR, dash='dot', width=3)
        ))
    
    # Limites
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
        st.error("Falha crítica na conexão com a base Parquet.")
        st.stop()
    
    f = render_sidebar(df_raw)
    df = apply_business_filters(df_raw, f['marcas'], f['paises'], f['dias'])
    
    # Proteção de Dados Incompletos (D-1)
    max_date = df_raw["data_faturamento"].max()
    df = df.filter(pl.col("data_faturamento") < max_date)

    if not df.is_empty():
        # --- CALC ENGINE ---
        v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        v_fut = PredictionService.get_daily_forecast(df)
        ins = AnalyticsService.get_strategic_insights(df) # Alterado para AnalyticsService para consistência
        
        # --- HEADER ---
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:1px;'>Business Monitoring & Predictive Engine</p>", unsafe_allow_html=True)
        
        # --- ROW 1: METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Volume Total", f"{len(df):,}")
        m2.metric("Líder de Canal", dist.tail(1)['marca'][0] if not dist.is_empty() else "N/A")
        m3.metric("Estabilidade (CV)", f"{ins.get('cv', 0):.2f}")
        m4.metric("Confiança IA", f"{ins.get('confianca', 0):.1f}%")
        
        st.divider()

        # --- ROW 2: SPC HERO ---
        st.markdown("<h3>📈 Controle de Estabilidade e Forecast Semanal</h3>", unsafe_allow_html=True)
        render_spc_chart(v_sem, v_fut, m_w, s_w)

        # --- ROW 3: MARKET DYNAMICS ---
        st.divider()
        c1, c2 = st.columns([1, 1])
        
        with c1:
            st.markdown("<h3>🏆 Market Share (Top 10)</h3>", unsafe_allow_html=True)
            if not dist.is_empty():
                fig_share = px.bar(dist.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR], text_auto=True)
                fig_share.update_layout(template="plotly_white", height=400, margin=dict(t=10, b=10, l=100, r=20))
                st.plotly_chart(fig_share, use_container_width=True)

        with c2:
            st.markdown("<h3>📅 Periodicidade de Faturamento</h3>", unsafe_allow_html=True)
            # Reaproveitando a lógica do heatmap mas garantindo robustez
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
                st.info("Aguardando dados para mapa sazonal.")

        # --- ROW 4: TÁTICA NOMINAL (COM FORMATAÇÃO) ---
        st.divider()
        st.markdown("<h3>🔮 Pipeline de Antecipação Nominal</h3>", unsafe_allow_html=True)
        df_p = PredictionService.get_client_predictions(df)
        if not df_p.is_empty():
            # Melhoria Sênior: Estilização da tabela para focar no faturamento
            st.dataframe(
                df_p.head(20), use_container_width=True, hide_index=True,
                column_config={
                    "Valor_Est": st.column_config.ProgressColumn("Faturamento Previsto", format="R$ %.2f", min_value=0, max_value=float(df_p["Valor_Est"].max())),
                    "Qtd_Prevista": st.column_config.NumberColumn("Volume", format="%d un")
                }
            )
        else:
            st.warning("Massa crítica insuficiente para projeção nominal individualizada.")

        # --- FOOTER: ESTRATÉGICO ---
        st.markdown(f"""
        <div class="insights-card">
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px;">
                <div><span>ESTRUTURA DE CARTEIRA</span><br><div class="val-text">{ins.get('perfil').upper()}</div></div>
                <div><span>CONCENTRAÇÃO (HHI)</span><br><div class="val-text">{ins.get('hhi',0):.3f}</div></div>
                <div><span>SAÚDE OPERACIONAL</span><br><div class="val-text">{ins.get('estabilidade').upper()}</div></div>
                <div><span>RISCO DE VOLATILIDADE</span><br><div class="val-text">{(ins.get('cv',0)*100):.1f}%</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Aplique os filtros na barra lateral para iniciar a análise.")

def set_all_state(label, options, value):
    for opt in options: st.session_state[f"chk_{label}_{opt}"] = value

if __name__ == "__main__":
    run_dashboard()