import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

nome_planta = 'Paranatinga'


def make_confusion_matrix(cf,
                          categories='auto',
                          planta='',
                          xyticks=True,
                          output_path='temp.png', days=None):
    # cf = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.rcParams["figure.figsize"] = [14, 12]
    plt.rcParams["figure.autolayout"] = True

    sns.heatmap(cf, annot=True,  fmt="", cbar_kws=dict(use_gridspec=True, location="left"), cmap='Blues', cbar=True,
                xticklabels=days, yticklabels=categories)

    plt.title('Confusion Matrix Normalized{}'.format(planta))

    # ax2.yaxis.tick_right()

    plt.savefig(output_path)
    # plt.show()


data = np.asarray([[0.33, 0.0, 0.0],
                   [0.0, 0.38, 0.15],
                   [0.0, 0.12, 0.27,]]

                  )

# temp_data = np.asarray([[108, 126, 1, 0, 0, 0], [14, 635, 430, 20, 0, 0], [0, 25, 344, 304, 3, 0], [0, 0, 12, 343, 61, 0], [0, 0, 1, 143, 174, 0], [0, 0, 0, 0, 0, 0]])


evaluation_days = ["MUSCULO", "MEMBRANA",  'GORDURAA']

make_confusion_matrix(data,
                      categories=["MUSCULO", "MEMBRANA",  'GORDURAA'],
                      planta='', days=evaluation_days,
                      output_path='/home/diego/Desktop/FRAGMENTADO.png'
                      )
