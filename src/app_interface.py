import plotly.express as px
import streamlit as st
import polars as pl
import sys
import os
import plotly.graph_objects as go
from datetime import timedelta

from src.data_engine import load_processed_data, apply_business_filters
from src.prediction_service import PredictionService
from src.analytics_service import AnalyticsService
from src.config import APP_TITLE, THEME_COLOR

def human_format(num):
    """Formata números para escala K, M, B de fácil leitura."""
    if num is None or num == 0: return "0"
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{:.1f}{}'.format(num, ['', 'K', 'M', 'B', 'T'][magnitude])

def set_all_state(label, options, value):
    for opt in options:
        st.session_state[f"chk_{label}_{opt}"] = value

def run_dashboard():
    """
    Interface Black Crow Intel v3.0.0.
    Foco: Ciclo Mensal Atual + 7 Dias Preditivos (Semana Seguinte).
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # CSS Customizado (Rapha Edition)
    st.markdown(f"""
        <style>
        [data-testid="stMetricValue"] {{ font-size: 1.6rem !important; }}
        .stMetric {{ 
            border: 1px solid rgba(128, 128, 128, 0.2); 
            padding: 15px; 
            border-radius: 10px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .stButton button {{ width: 100%; height: 1.6rem; font-size: 0.7rem !important; font-weight: bold; }}
        </style>
    """, unsafe_allow_html=True)

    @st.cache_data(ttl=3600)
    def get_cached_data(): return load_processed_data()

    df_raw = get_cached_data()
    if df_raw.is_empty(): st.error("❌ Base de dados Fleet não encontrada."); st.stop()

    # --- MAPEAMENTO DE COLUNA DE DATA ---
    date_col = "purchase_date" if "purchase_date" in df_raw.columns else "data_faturamento"
    
    # --- SIDEBAR: FILTROS E SEGMENTAÇÃO ---
    with st.sidebar:
        st.image("https://static.vecteezy.com/system/resources/thumbnails/026/847/626/small/flying-black-crow-isolated-png.png", width=70)
        st.title("Market Control")
        st.divider()
        
        def create_smart_filter(label, options):
            with st.expander(label, expanded=False):
                c1, c2 = st.columns(2)
                c1.button("Todos", key=f"all_{label}", on_click=set_all_state, args=(label, options, True))
                c2.button("Nenhum", key=f"none_{label}", on_click=set_all_state, args=(label, options, False))
                return [opt for opt in options if st.checkbox(opt, key=f"chk_{label}_{opt}", value=st.session_state.get(f"chk_{label}_{opt}", True))]

        sel_marcas = create_smart_filter("Marcas", sorted(df_raw["marca"].unique().to_list()))
        sel_paises = create_smart_filter("Países", sorted(df_raw["uf"].unique().to_list()))
        sel_setores = create_smart_filter("Setores", sorted(df_raw["industry_sector"].unique().to_list()))
        st.divider()
        sel_days = st.slider("Janela Mensal (Dias do Mês):", 1, 31, (1, 31))

    # --- PROCESSAMENTO D-1 E FILTROS ---
    max_date = df_raw[date_col].max()
    df_filt = apply_business_filters(df_raw, sel_marcas, sel_paises, sel_days)
    if sel_setores:
        df_filt = df_filt.filter(pl.col("industry_sector").is_in(sel_setores))
    
    # Blindagem D-1
    df_filt = df_filt.filter(pl.col(date_col) < max_date)

    if not df_filt.is_empty():
        # Inteligência Analítica
        total_vol = len(df_filt)
        v_temporal = df_filt.group_by(date_col).len(name="vol").sort(date_col)
        
        # Identificamos o Mês Atual para o Gráfico de Controle
        latest_month = max_date.month
        latest_year = max_date.year
        
        # Filtramos a série temporal para mostrar apenas o mês atual
        v_current_month = v_temporal.filter(
            (pl.col(date_col).dt.month() == latest_month) & 
            (pl.col(date_col).dt.year() == latest_year)
        )
        
        # Previsão: Próxima semana (7 dias)
        proj_vol, trend = PredictionService.get_market_trend(df_filt)
        v_future = PredictionService.get_daily_forecast(df_filt, horizon=7) 
        
        # Estatísticas SPC (calculadas sobre a janela filtrada para manter contexto)
        v_dia_spc, m, s = AnalyticsService.calculate_spc_metrics(df_filt)
        dist_data = AnalyticsService.get_pareto_distribution(df_filt).sort("vendas", descending=False)
        
        st.title(f"{APP_TITLE}")
        
        # --- BLOCO 1: KPIs ---
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Volume (D-1)", f"{total_vol:,}")
        if not dist_data.is_empty():
            lider = dist_data.tail(1)
            share = (lider['vendas'][0]/total_vol)
            k2.metric("Líder de Canal", f"{lider['marca'][0][:12]}...", delta=f"{share:.1%} Share")
        k3.metric("Frequência", f"{total_vol/(sel_days[1]-sel_days[0]+1):.1f} un/dia")
        k4.metric("Mercados Ativos", f"{len(df_filt['uf'].unique())}")
        k5.metric("Forecast Próx. Ciclo", human_format(proj_vol * 450000), delta=trend)

        st.divider()

        # --- BLOCO 2: DIAGNÓSTICO E ANTECIPAÇÃO ---
        c_left, c_right = st.columns([1.6, 1])
        with c_left:
            st.subheader("📊 Diagnóstico de Trajetória (Série Completa)")
            fig_area = px.area(v_temporal, x=date_col, y='vol', color_discrete_sequence=[THEME_COLOR])
            fig_area.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_area, use_container_width=True)
            st.info(f"O mercado apresenta trajetória **{trend}**. Projeção estimada: **{proj_vol} unidades**.")
        
        with c_right:
            st.subheader("🔮 Antecipação Nominal (Próx. Semana)")
            df_forecast = PredictionService.get_client_predictions(df_filt)
            if not df_forecast.is_empty():
                df_view = df_forecast.with_columns(pl.col("Valor_Est").map_elements(human_format, return_dtype=pl.String).alias("Valor Formatado"))
                st.dataframe(df_view.select(["Cliente", "Qtd_Prevista", "Valor Formatado", "Probabilidade"]), use_container_width=True, hide_index=True,
                             column_config={"Probabilidade": st.column_config.ProgressColumn("Confiança", min_value=0, max_value=1)})

        st.divider()

        # --- BLOCO 3: ESTABILIDADE E PARETO ---
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🏆 Pareto de Líderes")
            fig_bar = px.bar(dist_data.tail(10), x='vendas', y='marca', orientation='h', color_discrete_sequence=[THEME_COLOR])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with c2:
            st.subheader("📈 Controle (Mês Atual + 7D Forecast)")
            ucl, lcl = m + 2*s, max(0, m - 2*s)
            
            # Alertas baseados no último ponto real do mês
            if not v_current_month.is_empty():
                last_vol = v_current_month.tail(1)["vol"][0]
                if last_vol > ucl:
                    st.success(f"🚀 **Expansão:** Volume ({last_vol}) acima do UCL.")
                elif last_vol < lcl:
                    st.error(f"⚠️ **Retração:** Volume ({last_vol}) abaixo do LCL.")
                else:
                    st.info("✅ **Estabilidade:** Operação dentro da normalidade estatística.")

            fig_spc = go.Figure()
            # 1. Histórico Real do Mês Atual (Linha Sólida)
            fig_spc.add_trace(go.Scatter(
                x=v_current_month[date_col], 
                y=v_current_month['vol'], 
                mode='lines+markers', 
                name='Real', 
                line=dict(color=THEME_COLOR, width=2)
            ))
            
            # 2. Projeção de 7 dias (Linha Pontilhada)
            if not v_future.is_empty():
                last_real_date = v_current_month.tail(1)[date_col][0] if not v_current_month.is_empty() else max_date
                
                v_fut_plot = v_future.with_columns([
                    pl.Series([last_real_date + timedelta(days=i+1) for i in range(len(v_future))]).alias(date_col)
                ]).select([date_col, "vol"])
                
                # Conexão entre as linhas
                hist_tail = v_current_month.select([date_col, "vol"]).tail(1)
                hist_tail = hist_tail.with_columns([pl.col(date_col).cast(pl.Date), pl.col("vol").cast(pl.Float64)])
                v_fut_plot = v_fut_plot.with_columns([pl.col(date_col).cast(pl.Date), pl.col("vol").cast(pl.Float64)])
                
                conn = pl.concat([hist_tail, v_fut_plot])
                fig_spc.add_trace(go.Scatter(
                    x=conn[date_col], 
                    y=conn['vol'], 
                    mode='lines', 
                    name='Forecast', 
                    line=dict(color=THEME_COLOR, dash='dot', width=2)
                ))
            
            fig_spc.add_hline(y=ucl, line_dash="dash", line_color="#238636", annotation_text="UCL")
            fig_spc.add_hline(y=lcl, line_dash="dash", line_color="#da3633", annotation_text="LCL")
            
            fig_spc.update_layout(
                height=300, 
                margin=dict(t=10, b=10), 
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                showlegend=False,
                xaxis=dict(tickformat="%d/%m") # Formatação amigável de dia/mês
            )
            st.plotly_chart(fig_spc, use_container_width=True)

        st.divider()
        
        # --- BLOCO 4: LOGIC ENGINE ---
        vol_cv = (s / m) if m > 0 else 0
        hhi = (dist_data['vendas'].tail(10) / total_vol).pow(2).sum()
        st.markdown(f"""
        ```python
        # Strategic Insights - Logic Engine v3.0.0
        - Saúde da Carteira: Perfil {'CONCENTRADO' if hhi > 0.25 else 'DIVERSIFICADO'} (HHI: {hhi:.2f}).
        - Previsibilidade: Nota de Confiança em {max(0, 100-(vol_cv*100)):.1f}%.
        - Estabilidade: Operando em regime de {'volatilidade' if vol_cv > 0.4 else 'normalidade'}.
        ```
        """)
    else:
        st.sidebar.warning("⚠️ Ajuste os filtros para gerar análise.")