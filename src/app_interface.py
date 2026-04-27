import plotly.express as px
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from datetime import timedelta

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# --- CONSTANTES DE DESIGN PREMIUM ---
NAVY_DEEP = "#0F172A"      # Azul Marinho Profundo
ACCENT_BLUE = "#3B82F6"    # Azul de Destaque (Interativo)
CARD_BG = "#FFFFFF"        # Fundo dos Cards (Branco para Limpeza)
SECTION_BG = "#F8FAFC"     # Fundo da Página (Slate 50)
PANEL_TINT = "#f0f2f9"     # O "Novo Periwinkle" - Suave e Profissional
SUCCESS_COLOR = "#10B981"
DANGER_COLOR = "#EF4444"
WARNING_COLOR = "#F59E0B"

# --- UI COMPONENTS ---

def apply_enterprise_styles():
    # Layout
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
        
        /* 1. ESTRUTURA GLOBAL */
        [data-testid="stAppViewContainer"] {{ 
            background-color: {SECTION_BG} !important; 
            font-family: 'Inter', sans-serif;
        }}
        
        /* 2. SIDEBAR MODERNA */
        [data-testid="stSidebar"] {{ 
            background-color: {NAVY_DEEP} !important; 
            border-right: 1px solid rgba(255,255,255,0.1);
        }}
        
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {{
            color: #F1F5F9 !important;
            font-weight: 600 !important;
            letter-spacing: 0.3px;
        }}

        /* 3. TIPOGRAFIA DE TÍTULO */
        .header-title {{ 
            font-size: 3.2rem; 
            font-weight: 900; 
            color: {NAVY_DEEP} !important; 
            letter-spacing: -2.5px;
            line-height: 1;
            margin-bottom: 0.5rem;
        }}
        
        h3 {{
            color: {NAVY_DEEP} !important;
            font-weight: 800 !important;
            text-transform: uppercase;
            font-size: 1.1rem !important;
            letter-spacing: 1.5px;
            margin-top: 3rem !important;
            margin-bottom: 1.5rem !important;
            display: flex;
            align-items: center;
        }}
        
        h3::before {{
            content: "";
            display: inline-block;
            width: 4px;
            height: 24px;
            background-color: {ACCENT_BLUE};
            margin-right: 12px;
            border-radius: 2px;
        }}

        /* 4. SCORECARDS (MÉTRICAS) REFINADOS --- */
        .stMetric {{ 
            border: 1px solid #E2E8F0; 
            padding: 24px !important; 
            border-radius: 16px !important; 
            background-color: {CARD_BG} !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
            transition: transform 0.2s ease;
        }}
        
        .stMetric:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        }}
        
        [data-testid="stMetricValue"] {{ 
            font-weight: 800 !important; 
            color: {NAVY_DEEP} !important;
            font-size: 2.4rem !important;
            letter-spacing: -1px;
        }}
        
        [data-testid="stMetricLabel"] p {{ 
            color: #64748B !important; 
            font-weight: 600 !important;
            text-transform: uppercase !important;
            font-size: 0.8rem !important;
            letter-spacing: 1px !important;
        }}

        /* 5. INSIGHTS SUMMARY CARD */
        .insights-card {{
            background: {CARD_BG};
            border-radius: 20px;
            padding: 35px;
            border: 1px solid #E2E8F0;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }}
        
        .insights-card span {{
            color: #94A3B8 !important;
            font-weight: 700 !important;
            font-size: 0.75rem;
            letter-spacing: 1.2px;
        }}
        
        .insights-card .val-text {{
            color: {NAVY_DEEP} !important;
            font-weight: 800 !important;
            font-size: 1.4rem;
        }}

        /* 6. DATA TABLES */
        .stDataFrame {{ 
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid #E2E8F0;
        }}
        
        /* 7. RISK BADGES (MODERN PILLS) */
        .risk-pill {{
            display: inline-flex;
            align-items: center;
            padding: 2px 10px;
            border-radius: 99px;
            font-size: 0.7rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-top: 8px;
        }}
        .pill-low {{ background-color: #DCFCE7; color: #15803D; border: 1px solid #BBF7D0; }}
        .pill-med {{ background-color: #FEF9C3; color: #A16207; border: 1px solid #FEF08A; }}
        .pill-high {{ background-color: #FEE2E2; color: #B91C1C; border: 1px solid #FECACA; }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=60)
        st.markdown(f"### Black Crow\n<span style='opacity:0.6; font-size:0.8rem;'>Segmentação de Dados</span>", unsafe_allow_html=True)
        
        def smart_filter(label, col):
            opts = sorted(df[col].unique().to_list())
            with st.expander(f"Filtro: {label}", expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, opts, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, opts, False))
                return [o for o in opts if st.checkbox(o, key=f"chk_{label}_{o}", value=st.session_state.get(f"chk_{label}_{o}", True))]

        filtros = {
            "marcas": smart_filter("Empresas", "marca"),
            "paises": smart_filter("Países", "uf"),
            "setores": smart_filter("Segmentos", "industry_sector"),
            "dias": st.slider("Janela Mensal (Dias):", 1, 31, (1, 31))
        }
        
        if st.button("🔄 RESTART CORE ENGINE"):
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
        mode='lines+markers', name='Faturamento Real',
        line=dict(color=ACCENT_BLUE, width=4),
        marker=dict(size=10, color='white', line=dict(width=3, color=ACCENT_BLUE))
    ))
    
    # IA Forecast
    if not v_future.is_empty():
        last_val = v_semanal.tail(1)
        hist_part = last_val.select([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        pred_part = v_future.select([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        conn = pl.concat([hist_part, pred_part])
        
        fig.add_trace(go.Scatter(
            x=conn['semana'], y=conn['vol'], 
            mode='lines', name='IA Projection',
            line=dict(color=ACCENT_BLUE, dash='dot', width=3)
        ))
    
    # Limites Estatísticos (LSC/LIC)
    fig.add_hline(y=ucl, line_dash="dash", line_color=SUCCESS_COLOR, line_width=2, 
                  annotation_text="LSC (99.7%)", annotation_position="top left", 
                  annotation_font=dict(color=SUCCESS_COLOR, size=11))
    
    fig.add_hline(y=lcl, line_dash="dash", line_color=DANGER_COLOR, line_width=2, 
                  annotation_text="LIC (99.7%)", annotation_position="bottom left",
                  annotation_font=dict(color=DANGER_COLOR, size=11))

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500, margin=dict(t=30, b=50, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color=NAVY_DEEP)),
        font=dict(color=NAVY_DEEP, family="Inter")
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#B1B3B5",
        griddash='dash',
        tickfont=dict(size=12,
            color=NAVY_DEEP),
        linecolor='#3B82F6'
    )
    fig.update_yaxes(
        gridcolor="#6A6A6A",
        griddash='dot',
        tickfont=dict(size=12,
            color=NAVY_DEEP),
        linecolor='#3B82F6'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def run_dashboard():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_enterprise_styles()
    
    df_raw = load_processed_data()
    if df_raw.is_empty(): 
        st.error("Erro na base de dados processada.")
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
        
        # Risk Logic
        cv_val = ins.get('cv', 0)
        risk_pill = "pill-low" if cv_val < 0.20 else "pill-med" if cv_val < 0.40 else "pill-high"
        risk_lbl = "Estável" if cv_val < 0.20 else "Moderado" if cv_val < 0.40 else "Crítico"
        cv_clr = SUCCESS_COLOR if cv_val < 0.20 else WARNING_COLOR if cv_val < 0.40 else DANGER_COLOR
        
        # --- HEADER ---
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#64748B; font-weight:600; text-transform:uppercase; letter-spacing:2px; font-size:0.85rem;'> • Executive Intelligence Engine • </p>", unsafe_allow_html=True)
        
        # --- ROW 1: CORE METRICS ---
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("VOLUME TOTAL", f"{len(df):,}")
        
        l_text = dist.tail(1).to_series(0)[0] if not dist.is_empty() else "N/A"
        m2.metric("LÍDER DE CANAL", l_text)
        
        # Custom Metric for Volatility with Modern Pill
        with m3:
            st.markdown(f"""
                <div class="stMetric">
                    <div data-testid="stMetricLabel"><p>VOLATILIDADE (CV)</p></div>
                    <div style="display:flex; flex-direction:column;">
                        <span style="font-weight:800; font-size:2.4rem; color:{NAVY_DEEP}; letter-spacing:-1px;">{cv_val:.1%}</span>
                        <div class="risk-pill {risk_pill}">{risk_lbl}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        m4.metric("ACURÁCIA IA", f"{ins.get('confianca', 0):.1f}%")
        
        # --- ROW 2: ESTABILIDADE HERO ---
        st.markdown("<h3>Análise de Estabilidade Estocástica</h3>", unsafe_allow_html=True)
        render_spc_chart(v_sem, v_fut, m_w, s_w)

        # --- ROW 3: DYNAMICS ---
        c1, c2 = st.columns([1, 1.2])
        
        with c1:
            st.markdown("<h3>Dominância de Share</h3>", unsafe_allow_html=True)
            if not dist.is_empty():
                y_c = 'marca' if 'marca' in dist.columns else dist.columns[0]
                fig_s = px.bar(dist.tail(10), x='vendas', y=y_c, orientation='h', color_discrete_sequence=[ACCENT_BLUE], text_auto=True)
                fig_s.update_layout(
                    template="plotly_white", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=450, margin=dict(t=10, b=10, l=120, r=20),
                    font=dict(size=12, color=NAVY_DEEP),
                    xaxis=dict(showticklabels=False, title=None),
                    yaxis=dict(tickfont=dict(size=12, color=NAVY_DEEP), title=None)
                )
                st.plotly_chart(fig_s, use_container_width=True)

        with c2:
            st.markdown("<h3>Sazonalidade e Fluxo</h3>", unsafe_allow_html=True)
            try:
                df_h = df.with_columns([
                    pl.col("data_faturamento").dt.weekday().alias("dow"),
                    pl.col("data_faturamento").dt.day().map_elements(lambda d: (d-1)//7 + 1, return_dtype=pl.Int64).alias("sem")
                ]).group_by(["sem", "dow"]).len().to_pandas()
                h_m = df_h.pivot(index="sem", columns="dow", values="len").fillna(0)
                fig_h = px.imshow(h_m, color_continuous_scale="Blues", text_auto=True)
                fig_h.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    height=450, margin=dict(t=10, b=10), font=dict(size=12, color=NAVY_DEEP)
                )
                st.plotly_chart(fig_h, use_container_width=True)
            except:
                st.info("Calibrando sazonalidade...")

        # --- ROW 4: PIPELINE NOMINAL ---
        st.markdown("<h3>Antecipação Nominal de Próximo Ciclo</h3>", unsafe_allow_html=True)
        df_p = PredictionService.get_client_predictions(df)
        if not df_p.is_empty():
            st.dataframe(
                df_p.head(15), use_container_width=True, hide_index=True,
                column_config={
                    "Valor_Est": st.column_config.ProgressColumn(
                        "POTENCIAL FINANCEIRO", format="R$ %.2f", 
                        min_value=0, max_value=float(df_p["Valor_Est"].max())
                    ),
                    "Qtd_Prevista": st.column_config.NumberColumn("VOLUME", format="%d un")
                }
            )

        # --- FOOTER: STRATEGIC INSIGHTS ---
        st.markdown(f"""
        <div class="insights-card">
            <h4 style="margin-top:0; color:{NAVY_DEEP}; font-weight:800; text-transform:uppercase; border-bottom:1px solid #E2E8F0; padding-bottom:15px; font-size:1rem;">Executive Strategic Interpretation</h4>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 30px; margin-top:25px;">
                <div><span>PERFIL DE CARTEIRA</span><br><div class="val-text">{ins.get('perfil').upper()}</div></div>
                <div><span>CONCENTRAÇÃO HHI</span><br><div class="val-text">{ins.get('hhi',0):.3f}</div></div>
                <div><span>SAÚDE OPERACIONAL</span><br><div class="val-text">{ins.get('estabilidade').upper()}</div></div>
                <div><span>TENDÊNCIA DE RISCO</span><br><div class="val-text" style="color:{cv_clr} !important;">{risk_lbl.upper()}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Utilize os filtros laterais para processar a inteligência de mercado.")

def set_all_state(label, options, value):
    for opt in options: st.session_state[f"chk_{label}_{opt}"] = value

if __name__ == "__main__":
    run_dashboard()