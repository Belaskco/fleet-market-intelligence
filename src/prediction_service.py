import polars as pl
import pandas as pd
import numpy as np
import logging
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from lightgbm import LGBMRegressor

# Configuração de monitoramento
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PredictionService")

class PredictionService:
    """
    Motor de Inteligência Black Crow Intel.
    Utiliza MLForecast (Nixtla) para predições macro e micro.
    """

    @staticmethod
    def get_market_trend(df: pl.DataFrame):
        """
        Calcula a tendência macro do mercado usando Nixtla.
        Retorna: (Volume Projetado, Status da Tendência)
        """
        if df.is_empty() or len(df.group_by("dia_do_mes")) < 5:
            return 0, "Estável"

        try:
            # Preparação Nixtla Macro
            df_macro = df.group_by("data_faturamento").len(name="y").to_pandas()
            df_macro = df_macro.rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "total_market"
            
            # Motor Nixtla Simplificado para Tendência
            fcst = MLForecast(
                models=[LGBMRegressor(random_state=42, verbosity=-1)],
                freq='D',
                lags=[1, 7]
            )
            
            fcst.fit(df_macro)
            pred = fcst.predict(1) # Projeção do próximo dia
            
            proj_vol = int(pred["LGBMRegressor"].sum() * 30) # Estimativa ciclo mensal
            
            # Cálculo de inclinação (Slope) para status
            last_val = df_macro["y"].iloc[-1]
            next_val = pred["LGBMRegressor"].iloc[0]
            diff = (next_val - last_val) / (last_val if last_val > 0 else 1)
            
            trend = "Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável"
            return proj_vol, trend

        except Exception as e:
            logger.error(f"Erro no Nixtla Macro: {e}")
            return 0, "Indisponível"

    @staticmethod
    def get_client_predictions(df: pl.DataFrame):
        """
        Predição Nominal por Cliente (Nixtla MLForecast).
        Retorna: Cliente, Qtd_Prevista, Valor_Est, Probabilidade
        """
        if df.is_empty() or df["dia_do_mes"].n_unique() < 5:
            return 0, "Estável"

        try:
            # Preparação dos dados para MLForecast
            # unique_id: Marca | ds: Data | y: Contagem de pedidos
            data_prep = df.select([
                pl.col("marca").alias("unique_id"),
                pl.col("data_faturamento").alias("ds"),
                pl.lit(1).alias("y"),
                pl.col("faturamento")
            ]).to_pandas()

            data_prep['ds'] = pd.to_datetime(data_prep['ds'])
            ts_data = data_prep.groupby(['unique_id', 'ds']).agg({'y': 'sum', 'faturamento': 'mean'}).reset_index()

            # Configuração do Motor Nixtla
            fcst = MLForecast(
                models=[LGBMRegressor(random_state=42, verbosity=-1)],
                freq='D',
                lags=[1, 7],
                target_transforms=[Differences([1])]
            )

            # Fit e Forecast (Janela de 7 dias)
            fcst.fit(ts_data)
            predictions_raw = fcst.predict(7)

            # Formatação Final para Interface
            res = pl.from_pandas(predictions_raw).group_by("unique_id").agg([
                pl.col("LGBMRegressor").sum().round(0).alias("Qtd_Prevista")
            ])

            # Cálculo de faturamento estimado (Ticket Médio histórico * Qtd Prevista)
            ticket_medio = df.group_by("marca").agg(pl.col("faturamento").mean().alias("avg_price"))
            
            final_df = res.join(ticket_medio, left_on="unique_id", right_on="marca")
            final_df = final_df.with_columns([
                (pl.col("Qtd_Prevista") * pl.col("avg_price")).alias("Valor_Est"),
                pl.col("unique_id").alias("Cliente"),
                pl.lit(0.88).alias("Probabilidade")
            ]).filter(pl.col("Qtd_Prevista") > 0)

            return final_df.select(["Cliente", "Qtd_Prevista", "Valor_Est", "Probabilidade"]).sort("Valor_Est", descending=True).head(10)

        except Exception as e:
            logger.error(f"Erro no Nixtla Micro: {e}")
            return pl.DataFrame()