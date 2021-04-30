import pandas as pd
from scikit_obliquetree.BUTIF import BUTIF
from scikit_obliquetree.CO2 import ContinuouslyOptimizedObliqueRegressionTree
from scikit_obliquetree.HHCART import HouseHolderCART
from scikit_obliquetree.segmentor import MSE, MeanSegmentor
from sklearn import model_selection
from sklearn.datasets import load_boston
from sklearn.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_exps(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Lightweight script to test many models and find winners
    :param X_train: training split
    :param y_train: training target vector
    :param X_test: test split
    :param y_test: test target vector
    :return: DataFrame of predictions
    """

    dfs = []

    models = [
        ("RF", RandomForestRegressor(max_depth=3)),
        ("GBDT", GradientBoostingRegressor(max_depth=3)),
        (
            "BUTIF",
            BaggingRegressor(
                BUTIF(
                    linear_model=LogisticRegression(max_iter=10000),
                    task="regression",
                    max_leaf=8,
                ),
                100,
                n_jobs=-1,
            ),
        ),
        (
            "CO2",
            BaggingRegressor(
                ContinuouslyOptimizedObliqueRegressionTree(
                    MSE(), MeanSegmentor(), thau=500, max_iter=100, max_depth=3
                ),
                100,
                n_jobs=-1,
            ),
        ),
        (
            "HHCART",
            BaggingRegressor(
                HouseHolderCART(MSE(), MeanSegmentor(), max_depth=3),
                100,
                n_jobs=-1,
            ),
        ),
    ]
    results = []
    names = []
    scoring = ["r2", "neg_mean_absolute_error"]
    for name, model in models:
        model = Pipeline(
            [
                ("StandardScaler", StandardScaler()),
                ("model", model),
            ]
        )
        kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=0)
        cv_results = model_selection.cross_validate(
            model, X_train, y_train, cv=kfold, scoring=scoring
        )
        clf = model.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(name)

        results.append(cv_results)
        names.append(name)
        this_df = pd.DataFrame(cv_results)
        this_df["model"] = name
        dfs.append(this_df)
    final = pd.concat(dfs, ignore_index=True)
    return final


def comparison_task():
    from sklearn.model_selection import train_test_split

    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    final = run_exps(X_train, y_train, X_test, y_test)

    bootstraps = []
    for model in list(set(final.model.values)):
        model_df = final.loc[final.model == model]
        bootstrap = model_df.sample(n=30, replace=True)
        bootstraps.append(bootstrap)

    bootstrap_df = pd.concat(bootstraps, ignore_index=True)
    results_long = pd.melt(
        bootstrap_df, id_vars=["model"], var_name="metrics", value_name="values"
    )
    time_metrics = ["fit_time", "score_time"]  # fit time metrics
    ## PERFORMANCE METRICS
    results_long_nofit = results_long.loc[
        ~results_long["metrics"].isin(time_metrics)
    ]  # get df without fit data
    results_long_nofit = results_long_nofit.sort_values(by="values")
    ## TIME METRICS
    results_long_fit = results_long.loc[
        results_long["metrics"].isin(time_metrics)
    ]  # df with fit data
    results_long_fit = results_long_fit.sort_values(by="values")

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(20, 12))
    sns.set(font_scale=2.5)
    g = sns.boxplot(
        x="model",
        y="values",
        hue="metrics",
        data=results_long_nofit,
        palette="Set3",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title("Comparison of Model by Classification Metric")
    plt.savefig("./benchmark_models_performance.png", dpi=300)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(20, 12))
    sns.set(font_scale=2.5)
    g = sns.boxplot(
        x="model",
        y="values",
        hue="metrics",
        data=results_long_fit,
        palette="Set3",
    )
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.title("Comparison of Model by Fit and Score Time")
    plt.savefig("./benchmark_models_time.png", dpi=300)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    comparison_task()
