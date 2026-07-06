from src.core.models.tabular.xgb_regressor import XGBRegressor
from src.core.models.tabular.lgbm_regressor import LGBMRegressor
from src.core.models.tabular.catboost_regressor import CatBoostRegressorModel
from src.core.models.tabular.gandalf_regressor import GandalfRegressor
from src.core.models.tabular.tabm_regressor import TabMRegressor

# 模型映射
# - 单目标树模型：xgb / lgbm / catboost（每目标一模型，按统计量后缀选损失）
# - 多目标深度模型：gandalf（pytorch_tabular）/ tabm（BatchEnsemble，含 masked loss + OOV）
MODEL_REGISTRY = {
    "xgb": XGBRegressor,
    "lgbm": LGBMRegressor,
    "catboost": CatBoostRegressorModel,
    "gandalf": GandalfRegressor,
    "tabm": TabMRegressor,
}
