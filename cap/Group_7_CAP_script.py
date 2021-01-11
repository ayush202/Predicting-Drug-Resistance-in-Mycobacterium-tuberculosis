import pandas as pd
import tensorflow as tf

def run():
    model = tf.keras.models.load_model('./model_CAP')
    print(model.summary())

    X_testdata = pd.read_csv('X_testData_1.csv')
    Y_testdata_nolabel = pd.read_csv('Y_testData_1_nolabels_CAP.csv')
    output_id = set(Y_testdata_nolabel['ID'])
    print(len(output_id))

    X_testdata.drop('SNP_CN_2714366_C967A_V323L_eis', axis=1, inplace=True)
    X_testdata.drop('SNP_I_2713795_C329T_inter_Rv2415c_eis', axis=1, inplace=True)
    X_testdata.drop('SNP_I_2713872_C252A_inter_Rv2415c_eis', axis=1, inplace=True)

    generate_id = X_testdata['ID']
    X_testdata.drop('ID',axis=1,inplace=True)
    # generate_id
    # X_testdata
    output_pred = model.predict(X_testdata)
    out = pd.DataFrame({'CAP': output_pred[:, 0]})
    out['ID'] = generate_id
    out = out[['ID', 'CAP']]
    # out
    for ind in out.index:
        if out['ID'][ind] not in output_id:
            out.drop(ind, inplace=True)
    out['CAP'].value_counts()
    out.to_csv('Group_7_output_cap.csv', index=None)

if __name__ == '__main__':
    run()