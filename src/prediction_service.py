import polars as pl
import pandas as pd
import numpy as np
import logging
from mlforecast import MLForecast
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

# Configuração de monitoramento
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

class PredictionService:
    """
    Motor Black Crow.
    Forecasting (Nixtla + Fallback), Projeção Pontilhada, Anomalias e Antecipação Nominal.
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """
        Previsão macro de fechamento mensal.
        Tenta Nixtla, mas usa Regressão Linear se houver poucos dados.
        """
        if df.is_empty():
            return 0, "Estável"
            
        # Define a coluna de data (suporta aliases comuns)
        date_col = "data_faturamento" if "data_faturamento" in df.columns else "data_transacao"
        if date_col not in df.columns:
            return 0, "Sem Coluna Data"

        try:
            # Agregação para série temporal
            v_dia = df.group_by(date_col).len(name="y").sort(date_col)
            
            # Se houver menos de 10 pontos, Nixtla (lags=7) não terá fôlego. Então usa Fallback.
            if len(v_dia) < 10:
                return PredictionService._calculate_linear_fallback(v_dia)

            # Preparação Nixtla
            df_macro = v_dia.to_pandas().rename(columns={date_col: "ds"})
            df_macro["unique_id"] = "fleet_total"
            
            fcst = MLForecast(
                models=[LGBMRegressor(random_state=42, verbosity=-1)], 
                freq='D', 
                lags=[1, 7]
            )
            fcst.fit(df_macro)
            pred = fcst.predict(1)
            
            # Projeção para o ciclo (estimativa baseada na predição do próximo dia)
            proj_vol = int(pred["LGBMRegressor"].sum() * 30)
            
            last_real = df_macro["y"].iloc[-1]
            next_val = pred["LGBMRegressor"].iloc[0]
            diff = (next_val - last_real) / (last_real if last_real > 0 else 1)
            
            trend_status = "Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável"
            return proj_vol, trend_status
            
        except Exception as e:
            logger.warning(f"Nixtla falhou, acionando fallback linear: {e}")
            return PredictionService._calculate_linear_fallback(v_dia)

    @staticmethod
    def _calculate_linear_fallback(v_dia: pl.DataFrame):
        """Cálculo de tendência via Regressão Linear para poucos dados."""
        try:
            # Índice numérico para o tempo
            X = np.arange(len(v_dia)).reshape(-1, 1)
            y = v_dia["y"].to_numpy()
            
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            
            # Projeção simplificada
            proj_vol = int(y.mean() * 30)
            trend_status = "Alta" if slope > 0.1 else "Baixa" if slope < -0.1 else "Estável"
            
            return proj_vol, trend_status
        except:
            return 0, "Erro"

    @staticmethod
    def get_daily_forecast(df: pl.DataFrame, horizon=7):
        # Gera os dados para a linha pontilhada (Forecast Nixtla).
        if df.is_empty():
            return pl.DataFrame()
            
        date_col = "data_faturamento" if "data_faturamento" in df.columns else "data_transacao"
        if date_col not in df.columns:
            return pl.DataFrame()

        try:
            v_dia = df.group_by(date_col).len(name="y").sort(date_col)
            
            if len(v_dia) < 10:
                return pl.DataFrame()

            df_macro = v_dia.to_pandas().rename(columns={date_col: "ds"})
            df_macro["unique_id"] = "fleet_total"
            
            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1, 7])
            fcst.fit(df_macro)
            preds = fcst.predict(horizon)
            
            # Mantém o dia_do_mes para compatibilidade com o gráfico da interface
            preds['dia_do_mes'] = pd.to_datetime(preds['ds']).dt.day
            
            return pl.from_pandas(preds).select([
                pl.col("dia_do_mes"),
                pl.col("LGBMRegressor").alias("vol")
            ])
        except Exception as e:
            logger.error(f"Erro no forecast diário: {e}")
            return pl.DataFrame()

    @staticmethod
    def identify_anomalies(df: pl.DataFrame):
        # Detecção original de outliers via Z-Score.
        if df.is_empty(): 
            return pl.DataFrame()
            
        v_dia = df.group_by("dia_do_mes").len(name="vol")
        if v_dia.is_empty():
            return pl.DataFrame()
            
        mean, std = v_dia["vol"].mean(), v_dia["vol"].std()
        if std == 0:
            return pl.DataFrame()
            
        return v_dia.filter(
            (pl.col("vol") > mean + 2*std) | (pl.col("vol") < mean - 2*std)
        ).sort("dia_do_mes")

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        # Tabela nominal por cliente (Nixtla Micro) com suporte a séries curtas.
        if df.is_empty():
            return pl.DataFrame()
            
        date_col = "data_faturamento" if "data_faturamento" in df.columns else "data_transacao"
        if date_col not in df.columns:
            return pl.DataFrame()

        try:
            # Seleção de colunas necessárias
            data_prep = df.select([
                pl.col("marca").alias("unique_id"),
                pl.col(date_col).alias("ds"),
                pl.lit(1).alias("y"),
                pl.col("faturamento")
            ]).to_pandas()
            
            data_prep['ds'] = pd.to_datetime(data_prep['ds'])
            ts_data = data_prep.groupby(['unique_id', 'ds']).agg({'y': 'sum', 'faturamento': 'mean'}).reset_index()

            # Filtro de marcas com histórico mínimo para o Nixtla (lags=1)
            marcas_validas = ts_data.groupby('unique_id').size()
            marcas_validas = marcas_validas[marcas_validas > 2].index
            
            ts_filtered = ts_data[ts_data['unique_id'].is_in(marcas_validas)]
            
            if ts_filtered.empty:
                # Fallback heurístico se nenhuma marca tiver histórico suficiente
                return df.group_by("marca").agg([
                    pl.len().alias("Qtd_Prevista"),
                    pl.col("faturamento").mean().alias("Valor_Est")
                ]).with_columns([
                    pl.col("marca").alias("Cliente"),
                    pl.lit(0.6).alias("Probabilidade")
                ]).sort("Valor_Est", descending=True).head(10)

            fcst = MLForecast(models=[LGBMRegressor(random_state=42, verbosity=-1)], freq='D', lags=[1])
            fcst.fit(ts_filtered)
            preds = fcst.predict(7)

            res = pl.from_pandas(preds).group_by("unique_id").agg(pl.col("LGBMRegressor").sum().round(0).alias("Qtd_Prevista"))
            ticket_medio = df.group_by("marca").agg(pl.col("faturamento").mean().alias("avg_price"))
            
            final = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            
            return final.with_columns([
                pl.col("unique_id").alias("Cliente"),
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
                pl.lit(0.85).alias("Probabilidade")
            ]).select(["Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"]).sort("Valor_Est", descending=True).head(10)
        except Exception as e:
            logger.error(f"Erro nas predições nominais: {e}")
            return pl.DataFrame()