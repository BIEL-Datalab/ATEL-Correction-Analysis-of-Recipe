from src.core.models.tabular.xgb_regressor import XGBRegressor
from src.core.models.tabular.lgbm_regressor import LGBMRegressor
from src.core.models.tabular.catboost_regressor import CatBoostRegressorModel
from src.core.models.tabular.gandalf_regressor import GandalfRegressor

# 模型映射
MODEL_REGISTRY = {
    "xgb": XGBRegressor,
    "lgbm": LGBMRegressor,
    "catboost": CatBoostRegressorModel,
    "gandalf": GandalfRegressor,
}
