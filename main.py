from datetime import datetime

import pandas as pd
import numpy as np
from pip._internal.operations.check import MissingDict
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import precision_recall_fscore_support as score
import os
from wwo_hist import retrieve_hist_data


class MissingDict(dict):
    __missing__ = lambda self, key: key


map_values = {
    "Athletic de Bilbao": "Ath Bilbao",
    "Atletico de Madrid": "Ath Madrid",
    "Barcelona": "Barcelona",
    "Cadiz": "Cadiz",
    "Elche": "Elche",
    "Getafe": "Getafe",
    "Granada": "Granada",
    "Levante": "Levante",
    "Osasuna": "Osasuna",
    "Sevilla": "Sevilla",
    "Real Madrid": "Real Madrid",
    "Valencia": "Valencia",
    "Huesca": "Huesca",
    "Celta de Vigo": "Celta",
    "Deportivo Alaves": "Alaves",
    "SD Eibar": "Eibar",
    "Real Sociedad": "Sociedad",
    "Real Valladolid": "Valladolid",
    "Real Betis": "Betis"
}

mapping = MissingDict(**map_values)


def clean_dataframe(df_to_clean):
    for col_i in df_to_clean:
        df_to_clean[col_i] = df_to_clean[col_i].astype(str)
        df_to_clean[col_i] = df_to_clean[col_i].str.replace('\xa0', '', regex=True)
        df_to_clean[col_i] = df_to_clean[col_i].str.replace('\u200B', '', regex=True)
        # df_to_clean[col_i] = df_to_clean[col_i].str.lower()
    return df_to_clean


liga_20_21_df = pd.read_csv('sources/la_liga_fixtures_20-21.csv')
liga_20_21_df = liga_20_21_df.iloc[:, 0:23]
teams_info_20_21 = pd.read_csv('sources/teams_info_20_21.csv')
teams_info_20_21 = clean_dataframe(teams_info_20_21)
teams_info_20_21['Ciudad'] = teams_info_20_21['Ciudad'].str.lower()
teams_info_20_21['NewTeam'] = teams_info_20_21['Equipo'].map(mapping)
liga_20_21_df_merged = liga_20_21_df.merge(teams_info_20_21, how='left', left_on='HomeTeam', right_on='NewTeam')

liga_21_22_df = pd.read_csv('sources/la_liga_fixtures_21_22.csv')
liga_21_22_df = liga_21_22_df.iloc[:, 0:23]
teams_info_21_22 = pd.read_csv('sources/teams_info_21_22.csv')
teams_info_21_22 = clean_dataframe(teams_info_21_22)
teams_info_21_22['Ciudad'] = teams_info_21_22['Ciudad'].str.lower()
teams_info_21_22['NewTeam'] = teams_info_21_22['Equipo'].map(mapping)
liga_21_22_df_merged = liga_21_22_df.merge(teams_info_21_22, how='left', left_on='HomeTeam', right_on='NewTeam')

pd.set_option('display.max_columns', None)
liga_all_df = pd.concat([liga_20_21_df_merged, liga_21_22_df_merged], ignore_index=True)

cities = liga_all_df['Ciudad'].unique().ravel().tolist()
city_list = [str(c).replace(" ", "_") for c in cities]
cleanedList = [x for x in city_list if str(x) != 'nan']

os.chdir("sources/weather")
simplefilter("ignore", category=ConvergenceWarning)
frequency = 24
start_date = '01-JAN-2020'
end_date = '05-JAN-2020'
api_key = '7210716c378e4aa9ab1141954222906'
location_list = cleanedList
# hist_weather_data = retrieve_hist_data(api_key,
#                                       location_list,
#                                       start_date,
#                                       end_date,
#                                       frequency,
#                                       location_label=False,
#                                       export_csv=True,
#                                       store_df=True)


liga_all_df['HomeTeam_code'] = liga_all_df['HomeTeam'].astype("category").cat.codes
liga_all_df['AwayTeam_code'] = liga_all_df['AwayTeam'].astype('category').cat.codes
liga_all_df['FTR_code'] = liga_all_df['FTR'].astype('category').cat.codes
liga_all_df['HTR_code'] = liga_all_df['HTR'].astype('category').cat.codes
liga_all_df['Time_code'] = liga_all_df['Time'].str.replace(':.+', '', regex=True).astype('int')
liga_all_df['Date'] = pd.to_datetime(liga_all_df['Date'], format='%d/%m/%Y', infer_datetime_format=True)
liga_all_df['date_code'] = liga_all_df['Date'].dt.dayofweek
liga_all_df['Ciudad_code'] = liga_all_df['Ciudad'].astype('category').cat.codes
liga_all_df['Entrenador_code'] = liga_all_df['Entrenador'].astype('category').cat.codes
liga_all_df['Capitan_code'] = liga_all_df['Capitan'].astype('category').cat.codes
liga_all_df['Estadio_code'] = liga_all_df['Estadio'].astype('category').cat.codes
liga_all_df['Aforo_code'] = liga_all_df['Aforo'].astype('category').cat.codes

gm = liga_all_df.groupby('HomeTeam')
group = gm.get_group('Real Madrid')


def rolling_averages(group, cols, new_cols):
    group = group.sort_values('Date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


cols = ['FTR_code', 'HTR_code', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS',
        'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
new_cols = [f"{c}_rolling" for c in cols]

matches_rolling = liga_all_df.groupby("HomeTeam").apply(lambda x: rolling_averages(x, cols, new_cols))

matches_rolling = matches_rolling.droplevel("HomeTeam")
matches_rolling.index = range(matches_rolling.shape[0])

# ['HomeTeam_code', 'AwayTeam_code', 'Time_code', 'date_code','AY_rolling', 'HR_rolling', 'AR_rolling',
#            'AST_rolling', 'HF_rolling', 'AF_rolling', 'HC_rolling', 'AC_rolling', 'HY_rolling', 'HTHG_rolling',
#            'HTAG_rolling', 'HS_rolling', 'AS_rolling', 'HST_rolling', 'FTR_code_rolling', 'HTR_code_rolling',
#            'FTHG_rolling', 'FTAG_rolling', 'Ciudad_code', 'Entrenador_code', 'Capitan_code', 'Estadio_code',
#            'Aforo_code']


features = ['HomeTeam_code', 'AwayTeam_code', 'Time_code', 'date_code', 'AST_rolling', 'HC_rolling', 'AC_rolling',
            'HTHG_rolling', 'HTAG_rolling', 'HS_rolling', 'AS_rolling', 'HST_rolling', 'FTR_code_rolling',
            'HTR_code_rolling', 'FTHG_rolling', 'FTAG_rolling', 'Ciudad_code', 'Entrenador_code', 'Capitan_code']

target = ['FTR_code']

# import plotly.express as px
# fig = px.imshow(matches_rolling[features].corr())
# fig.show()

print(matches_rolling.head())


def train_RF(data, n_est, depth, min_samples_split):
    train = data[data['Date'] < '2022-02-01']
    print(len(train))
    test = data[data['Date'] > '2022-02-01']
    print(len(test))
    X_train = train[features]
    y_train = train[target].to_numpy().ravel()
    X_test = test[features]
    y_test = test[target].to_numpy().ravel()
    rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth, min_samples_split=min_samples_split, n_jobs=-1,
                                random_state=1)
    rf_model = rf.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    precision, recall, fscore, support = score(y_test, y_pred, average='macro')
    print('Est: {}/ Depth: {} / Min Split: {} ---- Precision: {}/ Recall: {}/ Accuracy: {} '.format(n_est,
                                                                                                    depth,
                                                                                                    min_samples_split,
                                                                                                    round(precision, 3),
                                                                                                    round(recall, 3),
                                                                                                    round((
                                                                                                                  y_pred == y_test).sum() / len(
                                                                                                        y_pred), 3)))


# for n_est in [10, 50, 100, 200]:
#    for depht in [10, 20, 40, 60, 80, 100, None]:
#        for min_samples_split in [10, 50, 100, 200, 300]:
#            train_RF(matches_rolling, n_est, depht, min_samples_split)


def make_predictions(data, features):
    train = data[data['Date'] < '2022-02-01']
    print(len(train))
    test = data[data['Date'] > '2022-02-01']
    print(len(test))
    X_train = train[features]
    y_train = train[target].to_numpy().ravel()
    X_test = test[features]
    y_test = test[target].to_numpy().ravel()
    rf_model = RandomForestClassifier(n_estimators=50,
                                      min_samples_split=200,
                                      max_depth=None,
                                      random_state=1,
                                      n_jobs=-1)
    rf_model.fit(X_train, y_train)
    feature_importance = sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]
    rf_predict_test = rf_model.predict(X_test)
    rf_predict_test_accuracy = metrics.accuracy_score(y_test, rf_predict_test)
    rf_class_report = metrics.classification_report(y_test, rf_predict_test)
    return rf_predict_test_accuracy, rf_class_report, feature_importance


accuracy, class_report, feature_importance = make_predictions(matches_rolling, features)

print(accuracy, class_report, feature_importance)

exit()

X_train = train[features].to_numpy()
y_train = train[target].to_numpy()
X_test = test[features].to_numpy()
y_test = test[target].to_numpy()

C_start = 0.0001
C_end = 1
C_inc = 0.0001
C_values, recall_scores = [], []

C_val = C_start
best_recall_score = 0
while (C_val < C_end):
    C_values.append(C_val)
    lr_model_loop = LogisticRegression(C=C_val, random_state=1, n_jobs=-1, multi_class='ovr')
    lr_model_loop.fit(X_train, y_train.ravel())
    lr_predict_loop_test = lr_model_loop.predict(X_test)
    recall_score = metrics.recall_score(y_test, lr_predict_loop_test, average='weighted')
    recall_scores.append(recall_score)
    if (recall_score > best_recall_score):
        best_recall_score = recall_score
        best_lr_predict_test = lr_predict_loop_test
    C_val = C_val + C_inc

best_score_C_Val = C_values[recall_scores.index(best_recall_score)]
print("Best recall score: {0} with a C val or {1}".format(best_recall_score, best_score_C_Val))

lr_model = LogisticRegression(C=best_score_C_Val, random_state=1)
lr_model.fit(X_train, y_train.ravel())
lr_predict_test = lr_model.predict(X_test)

lr_predict_train = lr_model.predict(X_train)

# metrics
accuracy2 = metrics.accuracy_score(y_train.ravel(), lr_predict_train)
print("LR: Accuracy: {0:.4f}".format(accuracy2))
print(metrics.confusion_matrix(y_train.ravel(), lr_predict_train))
print("")
print("Classification Report")
print(metrics.classification_report(y_train.ravel(), lr_predict_train))
