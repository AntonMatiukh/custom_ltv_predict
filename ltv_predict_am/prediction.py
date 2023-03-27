import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm as log_progress
import os
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import catboost as cb
import shap
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class Prediction:
    """
    Framework to transform data and predict LTV finally based on CatBoostRegressor.

    :param df: Pandas DataFrame
    :type df: pandas.core.frame.DataFrame
    """

    @staticmethod
    def calc_slope_intercept(x, y):
        """
        Return polyfit coefficients for chosen metric (y=a*x^b)

        :param x: Lifetime
        :param y: Cumulative metric
        :return: Coefficients
        :rtype: float, float
        """

        slope_fact, intercept_fact = np.polyfit(np.log(x), np.log(y), 1)
        a_fact, b_fact = round(np.exp(intercept_fact), 4), round(slope_fact, 4)
        return a_fact, b_fact

    @staticmethod
    def regression_basic_metrics(y, y_predict):
        """
        Calculate metrics for regression model

        :param y: target
        :param y_predict: target prediction
        :return: dictionary of metrics
        :rtype: dict
        """

        diff = np.round(abs(y_predict - y) * 100 / y)
        diff_na = (y_predict - y) * 100 / y
        metrics_dict = {'r2': r2_score(y, y_predict),
                        'mae': mean_absolute_error(y, y_predict),
                        'rmse': np.sqrt(mean_squared_error(y, y_predict)),
                        'diff_perc_abs': [np.quantile(diff, 0.025), np.quantile(diff, 0.25), np.quantile(diff, 0.5),
                                          np.quantile(diff, 0.75), np.quantile(diff, 0.975)],
                        'diff_perc': [np.quantile(diff_na, 0.025), np.quantile(diff_na, 0.25),
                                      np.quantile(diff_na, 0.5), np.quantile(diff_na, 0.75),
                                      np.quantile(diff_na, 0.975)]}
        return metrics_dict

    @staticmethod
    def check_df(df):
        """
        Convert columns to dates. Create lifetime column. Check columns types and data in df.

        :return: Dataframe for future work
        :rtype: pandas.core.frame.DataFrame
        """

        # Check df for N/A
        if df.isna().sum().sum() == 0:
            pass
        else:
            df = df.dropna()
            print('N/A values were deleted from df')

        # Create lifetime column
        df['date_'] = pd.to_datetime(df['date_'])
        df['date_reg'] = pd.to_datetime(df['date_reg'])
        df['lifetime'] = (df['date_'] - df['date_reg']).dt.days

        # Check and delete rows with negative lifetimes
        if df.query('lifetime < 0').empty:
            pass
        else:
            df = df.query('lifetime >= 0')
            print('Negative lifetimes were deleted from df')

        return df

    def df_calc(self, df, target_column, k, n, df_list):
        """
        Function for calculations

        :param df:
        :param append_list:
        :return:
        """

        df_tmp_g = df.groupby('lifetime', as_index=False).sum()
        dict_coeff = {}
        dict_coeff['date'] = str(df['date_'].min()).split(' ')[0] + ' - ' + \
                             str(df['date_'].max()).split(' ')[0]
        for c in df_tmp_g.drop(columns=['lifetime', target_column]):
            df_tmp_g[c] = df_tmp_g[c].cumsum()
            coeffs = self.calc_slope_intercept(df_tmp_g['lifetime'] + 1, df_tmp_g[c])
            dict_coeff[f'{c}_a'], dict_coeff[f'{c}_b'] = coeffs[0], coeffs[1]
        dict_coeff[target_column] = df.groupby('date_reg', as_index=False)[target_column].max()[target_column].sum()
        dict_coeff['days_in_cohort'] = k
        if n is not None:
            dict_coeff['lifetime'] = n
        df_list.append(dict_coeff)

    def get_final_df(self, df, is_closed_lifetime, min_cohort_days, max_cohort_days, min_lifetime, max_lifetime, target_column):
        """
        Create/replace new features

        :return: Dataframe for future work
        :rtype: pandas.core.frame.DataFrame
        """

        df = self.check_df(df=df)

        df_list = []

        if is_closed_lifetime == 0:
            for k in log_progress(range(min_cohort_days, max_cohort_days)):
                for d in df['date_reg'].unique():
                    df_tmp = df[(df['date_reg'] >= d)
                                     & (df['date_'] <= d + pd.DateOffset(k))]
                    self.df_calc(df=df_tmp, target_column=target_column, k=k, n=None, df_list=df_list)
        else:
            for k in log_progress(range(min_cohort_days, max_cohort_days)):
                for n in range(min_lifetime, max_lifetime):
                    for d in df['date_reg'].unique():
                        df_tmp = df[(df['date_reg'] >= d)
                                         & (df['date_reg'] <= d + pd.DateOffset(k))
                                         & (df['date_'] <= d + pd.DateOffset(k+n))]
                        self.df_calc(df=df_tmp, target_column=target_column, k=k, n=None, df_list=df_list)

        return pd.DataFrame(df_list).drop_duplicates()

    def make_basic_predict(self,df,test_size,random_state,target_column):
        """

        :param df: Final dataframe
        :param test_size: Test size for train/test
        :param random_state: Random state for train/test
        :return: predict, dictionary with regression metrics
        :rtype:
        """

        X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['date', target_column]),
                                                            df[target_column], test_size=test_size,
                                                            random_state=random_state)

        train_dataset = cb.Pool(X_train, y_train)
        test_dataset = cb.Pool(X_test, y_test)

        model = cb.CatBoostRegressor(loss_function='RMSE')

        model.fit(train_dataset)

        pred = model.predict(X_test)

        sorted_feature_importance = model.feature_importances_.argsort()
        plt.barh(df.drop(columns=['date', target_column]).columns,
                 model.feature_importances_[sorted_feature_importance],
                 color='turquoise')
        plt.xlabel("CatBoost Feature Importance")
        plt.show()

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test,
                          feature_names=df.drop(columns=['date', target_column]).columns)
        plt.show()

        return pred, self.regression_basic_metrics(y_test, pred)











