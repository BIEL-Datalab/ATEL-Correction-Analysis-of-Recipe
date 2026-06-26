from src.models.xgb_regressor import XGBRegressor
from src.models.lgbm_regressor import LGBMRegressor
from src.models.catboost_regressor import CatBoostRegressorModel
from src.models.gandalf_regressor import GandalfRegressor

# 模型映射
MODEL_REGISTRY = {
    "xgb": XGBRegressor,
    "lgbm": LGBMRegressor,
    "catboost": CatBoostRegressorModel,
    "gandalf": GandalfRegressor,
}
