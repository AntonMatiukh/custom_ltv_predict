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

    def __init__(self, df):
        self.df = df

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

    def check_df(self):
        """
        Convert columns to dates. Create lifetime column. Check columns types and data in df.

        :return: Dataframe for future work
        :rtype: pandas.core.frame.DataFrame
        """

        # Check df for N/A
        if self.df.isna().sum().sum() == 0:
            pass
        else:
            df = self.df.dropna()
            print('N/A values were deleted from df')

        # Create lifetime column
        self.df['date_'] = pd.to_datetime(self.df['date_'])
        self.df['date_reg'] = pd.to_datetime(self.df['date_reg'])
        self.df['lifetime'] = (self.df['date_'] - self.df['date_reg']).dt.days

        # Check and delete rows with negative lifetimes
        if self.df.query('lifetime < 0').empty:
            pass
        else:
            self.df = self.df.query('lifetime >= 0')
            print('Negative lifetimes were deleted from df')


    def df_calc(self, df, target_column, k, n, df_list, cat_features, custom_features):
        """
        Function for calculations

        :param df:
        :param append_list:
        :return:
        """

        if cat_features is None:

            df_tmp_g = df.groupby('lifetime').agg(['sum','mean',np.median]).reset_index()
            df_tmp_g.columns = [k[0]+'_'+k[1] if k[1] != '' else k[0] for k in df_tmp_g.columns]

            columns_to_drop = [c for c in df_tmp_g.columns if target_column in c]+['lifetime']

            for c in df_tmp_g.drop(columns=columns_to_drop):
                df_tmp_g[c] = df_tmp_g[c].cumsum()
                coeffs = self.calc_slope_intercept(df_tmp_g.index + 1, df_tmp_g[c])
                df_tmp_g[f'{c}_a'], df_tmp_g[f'{c}_b'] = coeffs[0], coeffs[1]
                df_tmp_g[f'{c}_max'] = df_tmp_g[c].max()
                df_tmp_g = df_tmp_g.drop(columns=c)

            for j in custom_features.keys():
                df_tmp_g[j] = df_tmp_g[custom_features[j].split(';')[0]] / df_tmp_g[custom_features[j].split(';')[1]]

            df_tmp_g['date'] = str(df['date_'].min()).split(' ')[0] + ' - ' + \
                               str(df['date_'].max()).split(' ')[0]
            df_tmp_g[target_column] = df.groupby('date_reg', as_index=False)[target_column].max()[target_column].sum()
            df_tmp_g['days_in_cohort'] = k
            if n is not None:
                df_tmp_g['cohort_lifetime'] = n

            df_tmp_g = df_tmp_g.drop(columns=columns_to_drop).drop_duplicates()

            df_list.append(df_tmp_g)

        else:

            df_tmp_g = df.groupby([cat_features]+['lifetime'], as_index=False).sum()

            cat_list = []
            for f in df_tmp_g[cat_features].unique():
                df_cat = df_tmp_g[df_tmp_g[cat_features] == f]
                for c in df_cat.drop(columns=[cat_features]+['lifetime', target_column]):
                    df_cat[c] = df_cat[c].cumsum()
                    coeffs = self.calc_slope_intercept(df_cat['lifetime'] + 1, df_cat[c])
                    df_cat[f'{c}_a'], df_cat[f'{c}_b'] = coeffs[0], coeffs[1]
                    df_cat = df_cat.drop(columns=c)
                cat_list.append(df_cat)

            df_tmp_g = pd.concat(cat_list)

            df_tmp_g['date'] = str(df['date_'].min()).split(' ')[0] + ' - ' + \
                               str(df['date_'].max()).split(' ')[0]
            df_target = df.groupby([cat_features]+['date_reg'], as_index=False)[target_column].max()\
                        .groupby(cat_features, as_index=False)[target_column].sum()
            df_tmp_g = df_tmp_g.drop(columns=target_column).merge(df_target, how='inner', on=cat_features)
            df_tmp_g['days_in_cohort'] = k
            if n is not None:
                df_tmp_g['cohort_lifetime'] = n

            df_tmp_g = df_tmp_g.drop(columns=['lifetime']).drop_duplicates()

            df_list.append(df_tmp_g)

    def get_final_df(self, is_closed_lifetime, min_cohort_days, max_cohort_days, min_lifetime, max_lifetime,
                     target_column, cat_features, custom_features):
        """
        Create/replace new features

        :return: Dataframe for future work
        :rtype: pandas.core.frame.DataFrame
        """

        self.check_df()

        df_list = []

        if is_closed_lifetime == 0:
            for k in log_progress(range(min_cohort_days, max_cohort_days)):
                for d in self.df['date_reg'].unique():
                    df_tmp = self.df[(self.df['date_reg'] >= d)
                                     & (self.df['date_'] <= d + pd.DateOffset(k))]
                    self.df_calc(df=df_tmp, target_column=target_column, k=k, n=None,
                                 df_list=df_list, cat_features=cat_features, custom_features=custom_features)
        else:
            for k in log_progress(range(min_cohort_days, max_cohort_days)):
                for n in range(min_lifetime, max_lifetime):
                    for d in self.df['date_reg'].unique():
                        df_tmp = self.df[(self.df['date_reg'] >= d)
                                         & (self.df['date_reg'] <= d + pd.DateOffset(k))
                                         & (self.df['date_'] <= d + pd.DateOffset(k+n))]
                        self.df_calc(df=df_tmp, target_column=target_column, k=k, n=n,
                                     df_list=df_list, cat_features=cat_features, custom_features=custom_features)

        self.df = pd.concat(df_list).drop_duplicates()

        return self.df

    def make_basic_predict(self,test_size,random_state,target_column,plot_importance,plot_detailed_predict,cat_features):
        """
        Function to make basic predict based on CatBoostRegressor

        :param df: Final dataframe
        :param test_size: Test size for train/test
        :param random_state: Random state for train/test
        :return: predict, dictionary with regression metrics
        :rtype: list, dict
        """

        X_train, X_test, y_train, y_test = train_test_split(self.df.drop(columns=['date', target_column]),
                                                            self.df[target_column],
                                                            test_size=test_size,
                                                            random_state=random_state)

        if cat_features is None:
            train_dataset = cb.Pool(X_train, y_train)
        else:
            train_dataset = cb.Pool(X_train, y_train, cat_features=[X_train.columns.get_loc(cat_features)])
        # test_dataset = cb.Pool(X_test, y_test, cat_features=[X_test.columns.get_loc(cat_features)])

        model = cb.CatBoostRegressor(loss_function='RMSE', silent=True)

        model.fit(train_dataset)

        pred = model.predict(X_test)

        if plot_importance == 1:

            sorted_feature_importance = model.feature_importances_.argsort()
            plt.barh(self.df.drop(columns=['date', target_column]).columns,
                     model.feature_importances_[sorted_feature_importance],
                     color='turquoise')
            plt.xlabel("CatBoost Feature Importance")
            plt.show()

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test,
                              feature_names=self.df.drop(columns=['date', target_column]).columns)
            plt.show()

        if plot_detailed_predict == 1:
            sample_size_list = []

            df_test_res = pd.concat([pd.concat([X_test, y_test], axis=1).reset_index().drop(columns='index'),
                              pd.DataFrame(pred).reset_index().drop(columns='index')], axis=1)
            df_test_res = df_test_res.rename(columns={0: 'predict'})

            if 'cohort_lifetime' not in df_test_res.columns:
                for d in sorted(df_test_res['days_in_cohort'].unique()):
                    df_test_res_f = df_test_res.query(f'days_in_cohort == {d}')
                    sample_size_list.append(self.regression_basic_metrics(df_test_res_f['fee_attr_180'],
                                                                          df_test_res_f['predict']))
                plt.figure(figsize=(20, 8))

                x = sorted(df_test_res['days_in_cohort'].unique())
                y_min = [min(k['diff_perc']) for k in sample_size_list]
                y_max = [max(k['diff_perc']) for k in sample_size_list]

                plt.plot(x, y_min, label='Error 0.025 quantile')
                for index in range(len(x)):
                    plt.text(x[index], y_min[index],
                             s=[round(min(k['diff_perc'])) for k in sample_size_list][index])
                plt.plot(x, y_max, label='Error 0.975 quantile')
                for index in range(len(x)):
                    plt.text(x[index], y_max[index],
                             s=[round(max(k['diff_perc'])) for k in sample_size_list][index])
                plt.legend()
                plt.show()
            else:
                for l in sorted(df_test_res['cohort_lifetime'].unique()):
                    sample_size_list = []
                    df_test_res_tmp = df_test_res.query(f'cohort_lifetime == {l}')
                    for d in sorted(df_test_res_tmp['days_in_cohort'].unique()):
                        df_test_res_f = df_test_res_tmp.query(f'days_in_cohort == {d}')
                        sample_size_list.append(self.regression_basic_metrics(df_test_res_f['fee_attr_180'],
                                                                              df_test_res_f['predict']))
                    plt.figure(figsize=(20, 8))

                    x = sorted(df_test_res_tmp['days_in_cohort'].unique())
                    y_min = [min(k['diff_perc']) for k in sample_size_list]
                    y_max = [max(k['diff_perc']) for k in sample_size_list]

                    plt.title(f'Lifetime == {l}')
                    plt.plot(x, y_min, label='Error 0.025 quantile')
                    for index in range(len(x)):
                        plt.text(x[index], y_min[index],
                                 s=[round(min(k['diff_perc'])) for k in sample_size_list][index])
                    plt.plot(x, y_max, label='Error 0.975 quantile')
                    for index in range(len(x)):
                        plt.text(x[index], y_max[index],
                                 s=[round(max(k['diff_perc'])) for k in sample_size_list][index])
                    plt.legend()
                    plt.show()

        return pred, self.regression_basic_metrics(y_test, pred)











