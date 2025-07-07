import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

csv_path = '/home/diego/teste_palemiras.csv'
# csv_path = '/home/diego/nova_auditoria_araguaina.csv'

dataframe_auditoria = pd.read_csv(csv_path)

print(dataframe_auditoria.columns.tolist())

dataframe_auditoria.loc[dataframe_auditoria['classificacao_ia'] == 91, 'classificacao_ia'] = 0
dataframe_auditoria.loc[dataframe_auditoria['classificacao_ia'] == 92, 'classificacao_ia'] = 0
dataframe_auditoria.loc[dataframe_auditoria['classificacao_ia'] == 93, 'classificacao_ia'] = 0
dataframe_auditoria.loc[dataframe_auditoria['classificacao_ia'] == 94, 'classificacao_ia'] = 0
dataframe_auditoria.loc[dataframe_auditoria['classificacao_ia'] == 95, 'classificacao_ia'] = 0
dataframe_auditoria.loc[dataframe_auditoria['classificacao_ia'] == 96, 'classificacao_ia'] = 0
dataframe_auditoria.loc[dataframe_auditoria['classificacao_ia'] == 97, 'classificacao_ia'] = 0

classificacao_user = dataframe_auditoria['classificacao_user'].to_numpy()
classificacao_ia = dataframe_auditoria['classificacao_ia'].to_numpy()

precision = precision_score(classificacao_user, classificacao_ia, average='weighted')
recall = recall_score(classificacao_user, classificacao_ia, average='weighted')
f1 = f1_score(classificacao_user, classificacao_ia, average='weighted')

print(precision, recall, f1)