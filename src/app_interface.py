import plotly.express as px
import streamlit as st
import polars as pl
import plotly.graph_objects as go
from datetime import timedelta

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

# --- UI COMPONENTS (COMPONENTES MODULARES DE ALTA DENSIDADE) ---

def apply_enterprise_styles():
    # Aplica o DNA visual.
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        /* Configuração de Fundo e Sidebar */
        [data-testid="stAppViewContainer"] {{ background-color: #E9F3F7 !important; font-family: 'Inter', sans-serif; }}
        [data-testid="stSidebar"] {{ background-color: #A0A0A0 !important; border-right: 2px solid #E2E8F0; }}

        /* Layout Adaptativo Profissional */
        .block-container {{
            max-width: 96% !important;
            padding-top: 3.5rem !important;
            padding-bottom: 2rem !important;
        }}

        /* Cards de Métricas (Elevation Style) */
        .stMetric {{ 
            border: 1px solid #CBD5E1; 
            padding: 24px; 
            border-radius: 12px; 
            background-color: #FFFFFF !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s;
        }}
        .stMetric:hover {{ transform: translateY(-2px); }}
        
        [data-testid="stMetricValue"] {{ font-size: 2.1rem !important; font-weight: 800; color: #0F172A !important; }}
        [data-testid="stMetricLabel"] {{ color: #475569 !important; font-weight: 700; font-size: 0.95rem !important; }}
        
        /* Tipografia de Cabeçalho e Títulos */
        .header-title {{ 
            font-size: 2.4rem; 
            font-weight: 900; 
            color: #0F172A; 
            border-bottom: 5px solid {THEME_COLOR}; 
            padding-bottom: 12px;
            margin-bottom: 30px; 
            letter-spacing: -1px;
        }}
        
        h3, .section-header {{ 
            color: #1E293B !important; 
            font-weight: 800 !important; 
            letter-spacing: -0.5px; 
            margin-bottom: 15px !important;
        }}

        /* Tabelas Nixtla e Dataframes */
        .stDataFrame {{ border: 1px solid #CBD5E1; border-radius: 10px; background: white; }}
        
        /* Botões de Ação */
        .stButton button {{ 
            width: 100%; font-weight: 700; text-transform: uppercase; 
            border-radius: 8px; border: 2px solid #475569 !important;
            color: #0F172A !important; background: white !important;
        }}
        .stButton button:hover {{ border-color: {THEME_COLOR} !important; color: {THEME_COLOR} !important; }}
        </style>
    """, unsafe_allow_html=True)

def render_sidebar(df):
    """Barra lateral com controles granulares e identidade visual."""
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=85)
        st.markdown("<h2 style='color:#0F172A; margin-top:10px; font-weight:900;'>Black Crow</h2>", unsafe_allow_html=True)
        st.caption("Intelligence Unit | v5.1.0")
        st.divider()
        
        def smart_filter(label, col):
            opts = sorted(df[col].unique().to_list())
            with st.expander(f"Filtro: {label}", expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, opts, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, opts, False))
                return [o for o in opts if st.checkbox(o, key=f"chk_{label}_{o}", value=st.session_state.get(f"chk_{label}_{o}", True))]

        filtros = {
            "marcas": smart_filter("Marcas / Grupos", "marca"),
            "paises": smart_filter("Mercados / UFs", "uf"),
            "setores": smart_filter("Setores Industriais", "industry_sector"),
            "dias": st.slider("Janela Mensal (Observação):", 1, 31, (1, 31))
        }
        st.info("💡 Dados normalizados D-1 para integridade preditiva.")
        return filtros

def render_scorecard(total, m_week, share_lider, confianca, forecast_val, trend):
    # Grade de KPIs de alto impacto com tooltips executivas.
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Volume Acumulado", f"{total:,}", help="Total de pedidos faturados no período (Excluindo D-0).")
    k2.metric("Líder de Canal", f"{share_lider[0][:14]}", delta=f"{share_lider[1]:.1%} Share", help="Empresa com maior fatia de volume no mercado selecionado.")
    k3.metric("Média Semanal", f"{m_week:.1f} un/sem", help="Volume médio transacionado em ciclos de 7 dias.")
    k4.metric("Previsibilidade", f"{confianca:.1f}%", help="Nota de Confiança do Logic Engine v5.0 baseada em Estabilidade Pura.")
    k5.metric("Forecast Próx. Mês", human_format(forecast_val * 125000), delta=trend, help="Projeção financeira baseada no volume Nixtla para as próximas 4 semanas.")

def render_spc_chart(v_semanal, v_future, m, s):
    # Gráfico SPC com 3 Sigma e projeção pontilhada Nixtla.
    ucl, lcl = m + 3*s, max(0, m - 3*s)
    
    # Alerta de Status Real (Topo do Gráfico)
    last_vol = v_semanal.tail(1)["vol"][0]
    if last_vol > ucl: st.success(f"🚀 **Alerta de Expansão:** Último ciclo ({last_vol}) rompeu o limite superior UCL ({ucl:.1f}).")
    elif last_vol < lcl: st.error(f"⚠️ **Alerta de Retração:** Último ciclo ({last_vol}) abaixo do limite inferior LCL ({lcl:.1f}).")
    else: st.info(f"✅ **Operação Estável:** Volume semanal ({last_vol}) operando dentro da normalidade estatística.")

    fig = go.Figure()
    # Série Histórica Real
    fig.add_trace(go.Scatter(
        x=v_semanal['semana'], y=v_semanal['vol'], 
        mode='lines+markers', name='Histórico Real', 
        line=dict(color=THEME_COLOR, width=3),
        marker=dict(size=8, color=THEME_COLOR, line=dict(width=2, color='white'))
    ))
    
    # Conexão e Forecast Pontilhado
    if not v_future.is_empty():
        last_date = v_semanal.tail(1)["semana"][0]
        v_fut_dates = v_future.with_columns([pl.Series([last_date + timedelta(weeks=i+1) for i in range(len(v_future))]).alias("semana")])
        h_tail = v_semanal.select(["semana", "vol"]).tail(1).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        f_tail = v_fut_dates.select(["semana", "vol"]).with_columns([pl.col("semana").cast(pl.Date), pl.col("vol").cast(pl.Float64)])
        conn = pl.concat([h_tail, f_tail])
        fig.add_trace(go.Scatter(x=conn['semana'], y=conn['vol'], mode='lines', name='Forecast Nixtla', line=dict(color=THEME_COLOR, dash='dot', width=3)))
    
    # Linhas de Controle Z-3
    fig.add_hline(y=ucl, line_dash="dash", line_color="#10B981", annotation_text="UCL (3σ)", annotation_position="top left")
    fig.add_hline(y=lcl, line_dash="dash", line_color="#EF4444", annotation_text="LCL (3σ)", annotation_position="bottom left")
    
    fig.update_layout(
        height=380, margin=dict(t=30, b=10), 
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white', 
        showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, linecolor='#94A3B8', tickfont=dict(color='#0F172A')),
        yaxis=dict(showgrid=True, gridcolor='#F1F5F9', zeroline=False, tickfont=dict(color='#0F172A'))
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Alerta Preditivo (Base do Gráfico)
    if not v_future.is_empty() and v_future.filter(pl.col("vol") > ucl).height > 0:
        st.warning("🔮 **Atenção:** O motor detectou picos de demanda prováveis para as próximas 4 semanas.")

# --- ENGINE ---

def set_all_state(label, options, value):
    for opt in options: st.session_state[f"chk_{label}_{opt}"] = value

def human_format(num):
    if num is None or num == 0: return "0"
    for unit in ['', 'K', 'M', 'B']:
        if abs(num) < 1000.0: return f"{num:.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}T"

def run_dashboard():
    """Função mestre que orquestra a interface v5.1.0."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    apply_enterprise_styles()
    
    df_raw = load_processed_data()
    if df_raw.is_empty(): st.error("❌ Base de dados Fleet não carregada corretamente."); st.stop()

    f = render_sidebar(df_raw)
    df = apply_business_filters(df_raw, f['marcas'], f['paises'], f['dias'])
    
    # Blindagem D-1: Remoção de dados parciais do dia atual
    max_d = df_raw["data_faturamento"].max()
    df = df.filter(pl.col("data_faturamento") < max_d)

    if not df.is_empty():
        # Inteligência Analítica e Preditiva
        v_sem = df.with_columns(pl.col("data_faturamento").dt.truncate("1w").alias("semana")).group_by("semana").len(name="vol").sort("semana")
        m_w, s_w = v_sem["vol"].mean(), v_sem["vol"].std()
        dist = AnalyticsService.get_pareto_distribution(df).sort("vendas")
        proj_vol, trend = PredictionService.get_market_trend(df)
        v_fut = PredictionService.get_daily_forecast(df, horizon=4)
        ins = PredictionService.get_strategic_insights(df)
        
        st.markdown(f"<div class='header-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
        
        # Bloco de Métricas Superiores
        lider_info = (dist.tail(1)['marca'][0], dist.tail(1)['vendas'][0]/len(df))
        render_scorecard(len(df), m_w, lider_info, ins.get('confianca', 0), proj_vol, trend)
        st.divider()

        # Grade Principal de Visualização
        cl, cr = st.columns([1.6, 1])
        with cl:
            st.subheader("📊 Diagnóstico de Trajetória")
            fig_area = px.area(v_sem, x='semana', y='vol', color_discrete_sequence=[THEME_COLOR])
            fig_area.update_layout(
                height=300, margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                xaxis=dict(tickfont=dict(color='#0F172A')), yaxis=dict(tickfont=dict(color='#0F172A'), title=None)
            )
            st.plotly_chart(fig_area, use_container_width=True)
            
            st.markdown("<h3 style='margin-top:20px;'>📈 Estabilidade e Controle Estatístico</h3>", unsafe_allow_html=True)
            render_spc_chart(v_sem, v_fut, m_w, s_w)

        with cr:
            st.subheader("🔮 Antecipação Nominal (Nixtla)")
            df_p = PredictionService.get_client_predictions(df)
            if not df_p.is_empty():
                df_view = df_p.with_columns(pl.col("Valor_Est").map_elements(human_format, return_dtype=pl.String).alias("Potencial"))
                st.dataframe(
                    df_view.select(["Cliente", "Qtd_Prevista", "Potencial", "Probabilidade"]), 
                    use_container_width=True, hide_index=True, 
                    column_config={
                        "Probabilidade": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=1, format="%.2f", color="green"),
                        "Qtd_Prevista": st.column_config.NumberColumn("Volume")
                    }
                )
            else:
                st.warning("⚠️ Dados insuficientes para forecast nominal por cliente.")
            
            st.markdown("<h3 style='margin-top:25px;'>🏆 Mix de Liderança (Top 10)</h3>", unsafe_allow_html=True)
            fig_bar = px.bar(dist.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR])
            fig_bar.update_layout(
                height=380, margin=dict(t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='white',
                xaxis=dict(tickfont=dict(color='#0F172A'), title="Pedidos"), yaxis=dict(tickfont=dict(color='#0F172A'), title=None)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Rodapé de Insights Estratégicos
        st.divider()
        st.markdown(f"""
        <div style="background-color:white; padding:25px; border:1px solid #E2E8F0; border-radius:12px; color:#0F172A; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <h3 style="margin-top:0; color:#0F172A;">Strategic Insights v5.1.0</h3>
            <p style="margin-bottom:8px;"><strong>Perfil da Carteira:</strong> {ins.get('perfil').upper()} | <strong>HHI (Concentração):</strong> {ins.get('hhi',0):.2f} | <strong>CV (Variação):</strong> {ins.get('cv',0):.2f}</p>
            <p style="margin-bottom:8px;"><strong>Estabilidade Operacional:</strong> {ins.get('estabilidade').upper()} | <strong>Média Semanal:</strong> {m_w:.1f} unidades</p>
            <p style="margin-bottom:0;"><strong>Confiança do Forecast:</strong> {ins.get('confianca', 0):.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros para gerar análise operacional.")