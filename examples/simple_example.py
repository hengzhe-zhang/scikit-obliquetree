from scikit_obliquetree.HHCART import HouseHolderCART
from scikit_obliquetree.segmentor import MSE, MeanSegmentor
from sklearn.datasets import load_boston
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import cross_val_score

X, y = load_boston(return_X_y=True)
reg = BaggingRegressor(
    HouseHolderCART(MSE(), MeanSegmentor(), max_depth=3),
    n_estimators=100,
    n_jobs=-1,
)
print("CV Score", cross_val_score(reg, X, y))
