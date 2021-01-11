import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,RandomizedSearchCV,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

def func(df):
    ss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

    traindata1 = df.drop(columns=['KAN'], axis=1)
    y1 = df['KAN']

    for train_index, test_index in ss.split(traindata1, y1):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = traindata1.iloc[train_index], traindata1.iloc[test_index]
        y_train, y_test = y1.iloc[train_index], y1.iloc[test_index]

    return X_train,y_train,X_test,y_test

def run():
    df = pd.read_csv('X_trainData_1.csv')
    df1 = pd.read_csv('Y_trainData_1.csv')

    df.drop('SNP_CN_2714366_C967A_V323L_eis', axis=1, inplace=True)
    df.drop('SNP_I_2713795_C329T_inter_Rv2415c_eis', axis=1, inplace=True)
    df.drop('SNP_I_2713872_C252A_inter_Rv2415c_eis', axis=1, inplace=True)
    count = 0
    for ind in df1.index:
        x = df1['KAN'][ind]
        if x == -1:
            count += 1
            df.drop(index=ind, inplace=True)
            df1.drop(index=ind, inplace=True)
    print(count)
    df['KAN'] = df1['KAN']

    X_train, y_train, X_test, y_test = func(df)
    rf = RandomForestClassifier(n_estimators=130, max_depth=4)
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))

    y_pred = rf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    X_testdata = pd.read_csv('X_testData_1.csv')
    Y_testdata_nolabel = pd.read_csv('Y_testData_1_nolabels_KAN.csv')
    output_id = set(Y_testdata_nolabel['ID'])
    print(len(output_id))

    X_testdata.drop('SNP_CN_2714366_C967A_V323L_eis', axis=1, inplace=True)
    X_testdata.drop('SNP_I_2713795_C329T_inter_Rv2415c_eis', axis=1, inplace=True)
    X_testdata.drop('SNP_I_2713872_C252A_inter_Rv2415c_eis', axis=1, inplace=True)

    generate_id = X_testdata['ID']
    X_testdata.drop('ID', axis=1, inplace=True)

    output_pred = rf.predict_proba(X_testdata)
    out = pd.DataFrame({'KAN': output_pred[:, 1]})
    out['ID'] = generate_id
    out = out[['ID', 'KAN']]
    # out
    for ind in out.index:
        if out['ID'][ind] not in output_id:
            out.drop(ind, inplace=True)
    #print(out['KAN'].value_counts())

    out.to_csv('Group_7_output_kan.csv', index=None)

if __name__ == '__main__':
    run()





