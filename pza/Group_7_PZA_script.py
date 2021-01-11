import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import tensorflow as tf
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.utils import class_weight

def run():
    model = tf.keras.models.load_model('./model_PZA')
    print(model.summary())

    X_testdata = pd.read_csv('X_testData_1.csv')
    Y_testdata_nolabel = pd.read_csv('Y_testData_1_nolabels_PZA.csv')
    output_id = set(Y_testdata_nolabel['ID'])
    print(len(output_id))
    X_testdata.drop('SNP_CN_2714366_C967A_V323L_eis', axis=1, inplace=True)
    X_testdata.drop('SNP_I_2713795_C329T_inter_Rv2415c_eis', axis=1, inplace=True)
    X_testdata.drop('SNP_I_2713872_C252A_inter_Rv2415c_eis', axis=1, inplace=True)

    generate_id = X_testdata['ID']
    X_testdata.drop('ID', axis=1, inplace=True)
    #generate_id
    # X_testdata = X_testdata.drop(columns=correlation.index,axis=1)
    output_pred = model.predict(X_testdata)
    # print(output_pred[:5])
    out = pd.DataFrame({'PZA': output_pred[:, 0]})
    out['ID'] = generate_id
    out = out[['ID', 'PZA']]
    # out
    for ind in out.index:
        if out['ID'][ind] not in output_id:
            out.drop(ind, inplace=True)
    #out['PZA'].value_counts()
    out.to_csv('Group_7_output_pza.csv', index=None)

if __name__ == '__main__':
    run()