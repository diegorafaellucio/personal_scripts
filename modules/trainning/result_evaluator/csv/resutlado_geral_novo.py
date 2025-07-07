import os
import cv2
import tqdm

from new_classifier.classifier import Classifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

nome_planta = 'Paranatinga'


def make_confusion_matrix(cf,
                          categories='auto',
                          planta='',
                          xyticks=True,
                          output_path='temp.png', days=None):
    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.rcParams["figure.figsize"] = [12, 6]
    plt.rcParams["figure.autolayout"] = True

    sns.heatmap(cf, annot=True, fmt="", cbar_kws=dict(use_gridspec=True, location="left"), cmap='Blues', cbar=True,
                xticklabels=days, yticklabels=categories)

    plt.title('Avaliação do Novo Modelo Preditivo\n\nPlanta: {}'.format(planta))

    # ax2.yaxis.tick_right()

    plt.savefig(output_path)
    # plt.show()


data = np.asarray([[0.92,0.92,0.91,0.96,0.96],
[0.95,0.91,0.94,0.94,0.9],
[0.93,0.96,0.96,0.95,0.9],
[0.89,0.82,0.81,0.89,0.74],
[0,0.67,0.67,1,0.9]]




)

# temp_data = np.asarray([[108, 126, 1, 0, 0, 0], [14, 635, 430, 20, 0, 0], [0, 25, 344, 304, 3, 0], [0, 0, 12, 343, 61, 0], [0, 0, 1, 143, 174, 0], [0, 0, 0, 0, 0, 0]])


evaluation_days = ["12/06/2023", "13/06/2023", "14/06/2023", "15/06/2023", "16/06/2023"]

make_confusion_matrix(data,
                      categories=["AUSENTE", "ESCASSA", "UNIFORME", "MEDIANA", "EXCESSIVA"],
                      planta='José Bonifácio', days=evaluation_days, output_path='/home/diego/Desktop/dados_auditoria_jbo.png'
                      )

