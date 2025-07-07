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

    plt.title('Avaliação do Novo Modelo Preditivo Para Erro 95\n\nPlanta: {}'.format(planta))

    # ax2.yaxis.tick_right()

    plt.savefig(output_path)
    # plt.show()


data = np.asarray([[1,88],
[266,616],
[472,582],
[313,168],
[2,12],
[12,12],
[421,2],
[0,0],
[35,42]]




)

# temp_data = np.asarray([[108, 126, 1, 0, 0, 0], [14, 635, 430, 20, 0, 0], [0, 25, 344, 304, 3, 0], [0, 0, 12, 343, 61, 0], [0, 0, 1, 143, 174, 0], [0, 0, 0, 0, 0, 0]])


evaluation_days = ["Modelo Atual", "Novo Modelo"]

make_confusion_matrix(data,
                      categories=["AUSENTE", "ESCASSA", "MEDIANA", "UNIFORME", "EXCESSIVA", "ERRO 91", "ERRO 95", "ERRO 96", "ERRO 97"],
                      planta='Paranatinga', days=evaluation_days, output_path='/home/diego/Desktop/erro_95_prn.png'
                      )

