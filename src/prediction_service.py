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
    def get_market_trend(df: pl.DataFrame):
        """
        Calcula a tendência macro do mercado usando Nixtla.
        """
        # CORREÇÃO: n_unique() resolve o erro de 'len(df.group_by)'
        if df.is_empty() or df["dia_do_mes"].n_unique() < 5:
            return 0, "Estável"

        try:
            # Preparação Nixtla Macro (MLForecast espera ds e y)
            df_macro = df.group_by("data_faturamento").len(name="y").to_pandas()
            df_macro = df_macro.rename(columns={"data_faturamento": "ds"})
            df_macro["unique_id"] = "total_market"
            
            fcst = MLForecast(
                models=[LGBMRegressor(random_state=42, verbosity=-1)],
                freq='D',
                lags=[1, 7]
            )
            
            fcst.fit(df_macro)
            pred = fcst.predict(1)
            
            proj_vol = int(pred["LGBMRegressor"].sum() * 30) 
            
            last_val = df_macro["y"].iloc[-1]
            next_val = pred["LGBMRegressor"].iloc[0]
            diff = (next_val - last_val) / (last_val if last_val > 0 else 1)
            
            trend = "Alta" if diff > 0.05 else "Baixa" if diff < -0.05 else "Estável"
            return proj_vol, trend

        except Exception as e:
            logger.error(f"Erro no Nixtla Macro: {e}")
            return 0, "Erro"

        except Exception as e:
            logger.error(f"Erro no Nixtla Micro: {e}")
            return pl.DataFrame()