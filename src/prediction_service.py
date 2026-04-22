import polars as pl
import pandas as pd
import numpy as np
import logging
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

# Configuração de Logs para auditoria de modelos
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

class PredictionService:
    """
    Motor de Inteligência Black Crow - Framework Fleet Market.
    Integra Nixtla (MLForecast) para predições de série temporal,
    Regressão Linear para fallbacks de baixa densidade e 
    Heurísticas de Mercado para o Logic Engine.
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """
        Analisa a trajetória macro do mercado.
        Retorna o volume projetado (Forecast) e o status da tendência (Alta/Baixa/Estável).
        """
        if df.is_empty() or "data_faturamento" not in df.columns:
            logger.warning("Dados insuficientes para análise de tendência.")
            return 0, "Sem Dados"
        
        try:
            # Agregação diária para série temporal
            v_dia = df.group_by("data_faturamento").len(name="y").sort("data_faturamento")
            
            # Nixtla exige massa crítica (lags=2 + seasonal). 
            # Se tivermos menos de 10 dias únicos, o fallback linear é mais confiável.
            if v_dia["data_faturamento"].n_unique() < 10:
                return PredictionService._linear_trend_fallback(v_dia)

            # Preparação Nixtla (ds, y, unique_id)
            df_macro = v_dia.to_pandas().rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_total_market"
            
            # Treinamento do Modelo Nixtla
            fcst = MLForecast(
                models=[LGBMRegressor(random_state=42, verbosity=-1)], 
                freq='D', 
                lags=[1, 2]
            )
            fcst.fit(df_macro)
            pred = fcst.predict(1)
            
            # Projeção de fechamento mensal baseada na tendência detectada
            val_pred = pred["LGBMRegressor"].iloc[0]
            proj_vol = int(val_pred * 30)
            
            # Cálculo de Delta para definir Status
            last_real = df_macro["y"].iloc[-1]
            diff = (val_pred - last_real) / (last_real if last_real > 0 else 1)
            
            trend_status = "Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável"
            return proj_vol, trend_status

        except Exception as e:
            logger.error(f"Falha no motor Nixtla Macro: {e}. Acionando fallback.")
            return PredictionService._linear_trend_fallback(v_dia)

    @staticmethod
    def get_daily_forecast(df: pl.DataFrame, horizon=7):
        """
        Gera a série temporal futura para a linha pontilhada da Carta de Controle (SPC).
        Mantém a continuidade visual entre o histórico real e a predição.
        """
        if df.is_empty() or "data_faturamento" not in df.columns: 
            return pl.DataFrame()
            
        try:
            v_dia = df.group_by("data_faturamento").len(name="y").sort("data_faturamento")
            
            if v_dia["data_faturamento"].n_unique() < 10:
                return PredictionService._linear_forecast_fallback(v_dia, horizon)

            df_macro = v_dia.to_pandas().rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "fleet_spc_forecast"
            
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 2])
            fcst.fit(df_macro)
            preds = fcst.predict(horizon)
            
            # Converte para o dia do mês para alinhar com o eixo X da interface
            preds['dia_do_mes'] = pd.to_datetime(preds['ds']).dt.day
            
            return pl.from_pandas(preds).select([
                pl.col("dia_do_mes").cast(pl.Int64), 
                pl.col("LGBMRegressor").alias("vol").cast(pl.Float64)
            ])
        except Exception as e:
            logger.error(f"Erro no forecast diário: {e}")
            return PredictionService._linear_forecast_fallback(v_dia, horizon)

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """
        Algoritmo de Antecipação Nominal.
        Cruza a predição Nixtla de volume por marca com o ticket médio histórico.
        """
        if df.is_empty() or "data_faturamento" not in df.columns: 
            return pl.DataFrame()
        
        try:
            # Preparação dos dados por Marca (unique_id)
            data_prep = df.select([
                pl.col("marca").alias("unique_id"),
                pl.col("data_faturamento").alias("ds"),
                pl.lit(1).alias("y"),
                pl.col("faturamento")
            ]).to_pandas()
            
            data_prep['ds'] = pd.to_datetime(data_prep['ds'])
            ts_data = data_prep.groupby(['unique_id', 'ds']).agg({
                'y': 'sum', 
                'faturamento': 'mean'
            }).reset_index()
            
            # Filtro de massa crítica por cliente para evitar erro no fit()
            valid_ids = ts_data.groupby('unique_id').size()
            ts_filtered = ts_data[ts_data['unique_id'].isin(valid_ids[valid_ids > 2].index)]

            if ts_filtered.empty:
                return PredictionService._heuristic_client_fallback(df)

            # Modelo Nixtla Nominal
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1])
            fcst.fit(ts_filtered)
            preds = fcst.predict(7)

            # Consolidação de volume e valor
            res = pl.from_pandas(preds).group_by("unique_id").agg(
                pl.col("LGBMRegressor").sum().round(0).alias("Qtd_Prevista")
            )
            
            ticket_medio = df.group_by("marca").agg(pl.col("faturamento").mean().alias("avg_price"))
            final = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            
            return final.with_columns([
                pl.col("unique_id").alias("Cliente"),
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
                pl.lit(0.85).alias("Probabilidade")
            ]).select([
                "Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"
            ]).sort("Valor_Est", descending=True).head(10)

        except Exception as e:
            logger.error(f"Erro em predições nominais: {e}")
            return PredictionService._heuristic_client_fallback(df)

    @staticmethod
    def identify_anomalies(df: pl.DataFrame):
        """
        Diagnóstico de Saúde Estatística.
        Identifica dias de faturamento atípico (Outliers) via Z-Score.
        """
        if df.is_empty(): return pl.DataFrame()
        
        # Agrupamento temporal para análise de volatilidade
        v_dia = df.group_by(pl.col("data_faturamento").dt.day().alias("dia_do_mes")).len(name="vol")
        
        m = v_dia["vol"].mean()
        s = v_dia["vol"].std()
        
        if s == 0 or s is None: return pl.DataFrame()
        
        # Filtro de 2 Sigma (95% de confiança)
        return v_dia.filter(
            (pl.col("vol") > m + 2*s) | (pl.col("vol") < m - 2*s)
        ).sort("dia_do_mes")

    @staticmethod
    def get_strategic_insights(df: pl.DataFrame):
        """
        Logic Engine Centralizado.
        Calcula drivers de decisão: Previsibilidade (CV) e Saúde da Carteira (HHI).
        """
        if df.is_empty(): return {}
        
        # 1. Previsibilidade (Coeficiente de Variação)
        v_dia = df.group_by("data_faturamento").len(name="vol")
        m, s = v_dia["vol"].mean(), v_dia["vol"].std()
        cv = (s / m) if m > 0 else 1
        confianca = max(0, 100 - (cv * 100))
        
        # 2. Concentração (HHI)
        v_marca = df.group_by("marca").len(name="vendas")
        total = v_marca["vendas"].sum()
        hhi = (v_marca["vendas"] / total).pow(2).sum() if total > 0 else 0
        
        return {
            "confianca": confianca,
            "hhi": hhi,
            "estabilidade": "Normal" if cv < 0.4 else "Atípica",
            "perfil": "Diversificado" if hhi < 0.25 else "Concentrado"
        }

    # --- MÉTODOS DE FALLBACK (REDES DE SEGURANÇA) ---

    @staticmethod
    def _linear_trend_fallback(v_dia: pl.DataFrame):
        """Fallback via Regressão Linear para tendências macro."""
        y = v_dia["y"].to_numpy()
        if len(y) < 2: return int(y.sum() if len(y) > 0 else 0), "Estável"
        X = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        slope = model.coef_[0]
        return int(y.mean() * 30), ("Alta" if slope > 0.05 else "Baixa" if slope < -0.05 else "Estável")

    @staticmethod
    def _linear_forecast_fallback(v_dia: pl.DataFrame, horizon: int):
        """Projeção linear simples para manter a linha pontilhada no gráfico."""
        try:
            y = v_dia["y"].to_numpy()
            if len(y) < 2: return pl.DataFrame()
            X = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            
            future_indices = np.arange(len(y), len(y) + horizon).reshape(-1, 1)
            preds = model.predict(future_indices)
            
            last_date = v_dia["data_faturamento"].max()
            future_days = [(last_date + pd.Timedelta(days=i+1)).day for i in range(horizon)]
            
            return pl.DataFrame({
                "dia_do_mes": pl.Series(future_days, dtype=pl.Int64), 
                "vol": pl.Series(preds, dtype=pl.Float64)
            })
        except: return pl.DataFrame()

    @staticmethod
    def _heuristic_client_fallback(df: pl.DataFrame):
        """Fallback baseado em média histórica quando ML não tem dados suficientes."""
        return df.group_by("marca").agg([
            pl.len().alias("Qtd_Prevista"),
            pl.col("faturamento").mean().alias("avg_price")
        ]).with_columns([
            pl.col("marca").alias("Cliente"),
            (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
            pl.lit(0.60).alias("Probabilidade")
        ]).sort("Valor_Est", descending=True).select([
            "Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"
        ]).head(10)