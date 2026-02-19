import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


def model_split(train, validate, test):

    X_train= train.drop(columns = ['sex','smoker','region','charges'])
    y_train = train.charges
    
    
    X_validate = validate.drop(columns = ['sex','smoker','region','charges'])
    y_validate = validate.charges
    
    X_test = test.drop(columns = ['sex','smoker','region','charges'])
    y_test = test.charges



    return X_train, y_train, X_validate, y_validate, X_test, y_test


def scaled_data(X_train, X_validate, X_test):
    
    X_train_scaled = X_train.copy()
    X_validate_scaled = X_validate.copy()
    X_test_scaled = X_test.copy()    
    
    scaler = StandardScaler()
    
    X_train_scaled[['age','bmi','children']] = scaler.fit_transform(X_train_scaled[['age','bmi','children']])
    X_validate_scaled[['age','bmi','children']] = scaler.transform(X_validate_scaled[['age','bmi','children']])
    X_test_scaled[['age','bmi','children']] = scaler.transform(X_test_scaled[['age','bmi','children']])
    
    return X_train_scaled, X_validate_scaled, X_test_scaled


def log_tran(y_train, y_validate):
    
    y_train_log = np.log1p(y_train)
    y_validate_log = np.log1p(y_validate)

    return y_train_log, y_validate_log    

def baseline_rmse_log_and_dollars(
    y_train,
    y_validate,
    y_train_log,
    y_validate_log,
    log_transform: str = "log1p",
    return_preds: bool = False
):
    """
    Compute baseline RMSE in log space and in dollars for mean and median baselines.

    Parameters
    ----------
    y_train, y_validate : array-like
        Original target values in dollars.
    y_train_log, y_validate_log : array-like
        Log-transformed targets (must correspond to y_train/y_validate).
    log_transform : {"log1p", "log"}
        Which inverse transform to use when converting predictions back to dollars.
        - "log1p" assumes y_log = np.log1p(y) and uses np.expm1 to invert.
        - "log"   assumes y_log = np.log(y)   and uses np.exp to invert.
    return_preds : bool
        If True, returns the train/validate prediction DataFrames too.

    Returns
    -------
    baseline_df : pd.DataFrame
        RMSE summary table with columns:
        Train_Log, Validate_Log, Train_Dollars, Validate_Dollars
        and rows:
        Mean Baseline, Median Baseline
    (optional) base_y_train, base_y_validate : pd.DataFrame
        Only returned if return_preds=True
    """

    # Ensure series for consistent behavior
    y_train = pd.Series(y_train).astype(float)
    y_validate = pd.Series(y_validate).astype(float)
    y_train_log = pd.Series(y_train_log).astype(float)
    y_validate_log = pd.Series(y_validate_log).astype(float)

    # Choose inverse transform
    if log_transform == "log1p":
        inv = np.expm1
    elif log_transform == "log":
        inv = np.exp
    else:
        raise ValueError("log_transform must be 'log1p' or 'log'.")

    # Put logged y into DataFrames for convenience
    base_y_train = pd.DataFrame({'charges_log': y_train_log})
    base_y_validate = pd.DataFrame({'charges_log': y_validate_log})

    # --- Baseline predictions in LOG space ---
    pred_mean_log = y_train_log.mean()
    pred_median_log = y_train_log.median()

    base_y_train['pred_mean_log'] = pred_mean_log
    base_y_validate['pred_mean_log'] = pred_mean_log

    base_y_train['pred_median_log'] = pred_median_log
    base_y_validate['pred_median_log'] = pred_median_log

    # --- RMSE in LOG space ---
    rmse_train_mean_log = mean_squared_error(
        base_y_train['charges_log'], base_y_train['pred_mean_log'])**(1/2)
    rmse_validate_mean_log = mean_squared_error(
        base_y_validate['charges_log'], base_y_validate['pred_mean_log'])**(1/2)

    rmse_train_median_log = mean_squared_error(
        base_y_train['charges_log'], base_y_train['pred_median_log'])**(1/2)
    rmse_validate_median_log = mean_squared_error(
        base_y_validate['charges_log'], base_y_validate['pred_median_log'])**(1/2)

    # --- Convert baseline predictions back to DOLLARS ---
    base_y_train['base_mean_dollars'] = inv(base_y_train['pred_mean_log'])
    base_y_validate['base_mean_dollars'] = inv(base_y_validate['pred_mean_log'])

    base_y_train['base_median_dollars'] = inv(base_y_train['pred_median_log'])
    base_y_validate['base_median_dollars'] = inv(base_y_validate['pred_median_log'])

    # --- RMSE in DOLLARS ---
    rmse_train_mean_dollars = mean_squared_error(
        y_train, base_y_train['base_mean_dollars'])**(1/2)
    rmse_validate_mean_dollars = mean_squared_error(
        y_validate, base_y_validate['base_mean_dollars'])**(1/2)

    rmse_train_median_dollars = mean_squared_error(
        y_train, base_y_train['base_median_dollars'])**(1/2)
    rmse_validate_median_dollars = mean_squared_error(
        y_validate, base_y_validate['base_median_dollars'])**(1/2)

    # Summary table
    baseline_df = pd.DataFrame({
        'Train_Log': [rmse_train_mean_log, rmse_train_median_log],
        'Validate_Log': [rmse_validate_mean_log, rmse_validate_median_log],
        'Train_Dollars': [rmse_train_mean_dollars, rmse_train_median_dollars],
        'Validate_Dollars': [rmse_validate_mean_dollars, rmse_validate_median_dollars]
    }, index=['Mean Baseline', 'Median Baseline'])

    if return_preds:
        return baseline_df, base_y_train, base_y_validate
    return baseline_df


def run_regression_models(
    X_train_scaled, X_validate_scaled, X_test_scaled,
    y_train, y_validate, y_test,
    y_train_log, y_validate_log, y_test_log=None,
    baseline_df=None,
    poly_degree=2,
    tweedie_power=1,
    tweedie_alpha=0,
    lars_alpha=1.0,
    log_transform="log1p",
    sort_by="Validate_Dollars",
    return_models=False,
    return_preds=False
):
    """
    Fit multiple regression models on log-transformed target and evaluate RMSE
    in both log space and dollars.

    Models:
      - LinearRegression
      - LassoLars
      - TweedieRegressor
      - PolynomialFeatures(degree=poly_degree) + LinearRegression

    Parameters
    ----------
    X_*_scaled : array-like
        Scaled feature matrices (train/validate/test).
    y_* : array-like
        Original target in dollars.
    y_*_log : array-like
        Log-transformed targets corresponding to y_*.
    baseline_df : pd.DataFrame or None
        If you already computed baseline_df, pass it in to include in results.
    log_transform : {"log1p", "log"}
        Determines inverse transform for converting predictions back to dollars.
    sort_by : str
        Column to sort final results by.
    return_models : bool
        If True, returns a dict of fitted model objects.
    return_preds : bool
        If True, returns a dict with prediction DataFrames per model.

    Returns
    -------
    results_df : pd.DataFrame
    (optional) models : dict
    (optional) preds : dict
    """

    # Ensure series
    y_train = pd.Series(y_train).astype(float)
    y_validate = pd.Series(y_validate).astype(float)
    y_test = pd.Series(y_test).astype(float)

    y_train_log = pd.Series(y_train_log).astype(float)
    y_validate_log = pd.Series(y_validate_log).astype(float)
    if y_test_log is not None:
        y_test_log = pd.Series(y_test_log).astype(float)

    # Inverse transform
    if log_transform == "log1p":
        inv = np.expm1
    elif log_transform == "log":
        inv = np.exp
    else:
        raise ValueError("log_transform must be 'log1p' or 'log'.")

    def _rmse(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)**(1/2)

    def _fit_predict_eval(model, name, Xtr, Xv, ytr_log, yv_log, ytr, yv):
        # fit
        model.fit(Xtr, ytr_log)

        # predict in log space
        pred_tr_log = model.predict(Xtr)
        pred_v_log = model.predict(Xv)

        # rmse in log space
        rmse_tr_log = _rmse(ytr_log, pred_tr_log)
        rmse_v_log  = _rmse(yv_log, pred_v_log)

        # convert to dollars
        pred_tr_dol = inv(pred_tr_log)
        pred_v_dol  = inv(pred_v_log)

        # rmse in dollars
        rmse_tr_dol = _rmse(ytr, pred_tr_dol)
        rmse_v_dol  = _rmse(yv, pred_v_dol)

        row = pd.DataFrame({
            "Train_Log": [rmse_tr_log],
            "Validate_Log": [rmse_v_log],
            "Train_Dollars": [rmse_tr_dol],
            "Validate_Dollars": [rmse_v_dol],
        }, index=[name])

        pred_df = {
            "train": pd.DataFrame({
                "y_log": ytr_log,
                "pred_log": pred_tr_log,
                "pred_dollars": pred_tr_dol
            }),
            "validate": pd.DataFrame({
                "y_log": yv_log,
                "pred_log": pred_v_log,
                "pred_dollars": pred_v_dol
            }),
        }

        return row, model, pred_df

    results = []
    models = {}
    preds = {}

    # Include baseline if provided
    if baseline_df is not None:
        results.append(baseline_df)

    # 1) Linear Regression
    lm = LinearRegression()
    row, fitted, pred_df = _fit_predict_eval(
        lm, "linear_regression",
        X_train_scaled, X_validate_scaled,
        y_train_log, y_validate_log,
        y_train, y_validate
    )
    results.append(row)
    models["linear_regression"] = fitted
    preds["linear_regression"] = pred_df

    # 2) LassoLars
    lars = LassoLars(alpha=lars_alpha)
    row, fitted, pred_df = _fit_predict_eval(
        lars, "lasso_lars",
        X_train_scaled, X_validate_scaled,
        y_train_log, y_validate_log,
        y_train, y_validate
    )
    results.append(row)
    models["lasso_lars"] = fitted
    preds["lasso_lars"] = pred_df

    # 3) Tweedie
    glm = TweedieRegressor(power=tweedie_power, alpha=tweedie_alpha)
    row, fitted, pred_df = _fit_predict_eval(
        glm, f"tweedie(p={tweedie_power},a={tweedie_alpha})",
        X_train_scaled, X_validate_scaled,
        y_train_log, y_validate_log,
        y_train, y_validate
    )
    results.append(row)
    models["tweedie"] = fitted
    preds["tweedie"] = pred_df

    # 4) Polynomial degree N + Linear Regression
    pf = PolynomialFeatures(degree=poly_degree, include_bias=False)
    Xtr_poly = pf.fit_transform(X_train_scaled)
    Xv_poly  = pf.transform(X_validate_scaled)
    Xt_poly  = pf.transform(X_test_scaled)  # ready if you want later

    lm_poly = LinearRegression()
    row, fitted, pred_df = _fit_predict_eval(
        lm_poly, f"polynomial_deg_{poly_degree}",
        Xtr_poly, Xv_poly,
        y_train_log, y_validate_log,
        y_train, y_validate
    )
    results.append(row)
    models[f"polynomial_deg_{poly_degree}"] = fitted
    models["poly_transform"] = pf
    preds[f"polynomial_deg_{poly_degree}"] = pred_df

    # Combine results
    results_df = pd.concat(results, axis=0)

    # Sort
    if sort_by in results_df.columns:
        results_df = results_df.sort_values(sort_by, ascending=True)
    else:
        results_df = results_df.sort_values("Validate_Dollars", ascending=False)

    if return_models and return_preds:
        return results_df, models, preds
    if return_models:
        return results_df, models
    if return_preds:
        return results_df, preds
    return results_df
    