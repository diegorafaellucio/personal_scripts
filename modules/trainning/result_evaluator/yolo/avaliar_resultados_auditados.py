import os
import cv2
import tqdm

from base.src.classifier.classifier import Classifier
from base.src.new_classifier.classifier_v8 import ClassifierV8
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

nome_planta = 'MSO'


def make_confusion_matrix(cf,
                          categories='auto',
                          planta='',
                          data='',
                          xyticks=True,
                          output_path='temp.png'):
    cfn = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]

    accuracy = np.trace(cf) / float(np.sum(cf))

    acuracia = "\n\nAcurácia={:0.3f}".format(accuracy)

    total_de_amostras = '\n\nTotal de amostras: {}'.format(np.sum(cf))

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    # MAKE THE HEATMAP VISUALIZATION
    plt.rcParams["figure.figsize"] = [18, 18]
    plt.rcParams["figure.autolayout"] = True

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    sns.heatmap(cfn, annot=True, fmt="", cbar_kws=dict(use_gridspec=True, location="left"), cmap='Purples', cbar=True,
                xticklabels=categories, yticklabels=categories, ax=ax1)
    # sns.heatmap(cfn, annot=True, fmt=".2f", cmap='Blues', cbar=True, xticklabels=categories, yticklabels='', ax=ax2)

    plt.suptitle('Avaliação do Modelo Preditivo\n\nPlanta: {}\n\nData: {}'.format(planta, data))

    ax1.set_xlabel(total_de_amostras)
    ax2.set_xlabel(acuracia)

    # ax2.yaxis.tick_right()

    fig.subplots_adjust(wspace=0.001)

    plt.savefig(output_path)
    # plt.show()


def make_confusion_matrix_1_side(cf,
                                 categories='auto',
                                 planta='',
                                 data='',
                                 xyticks=True,
                                 output_path='temp.png'):
    cfn = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]

    trace = np.trace(cf)
    all  = np.sum(cf)
    accuracy = np.trace(cf) / float(np.sum(cf))

    acuracia = "\nAcurácia={:0.3f}".format(accuracy)

    total_de_amostras = 'Total de amostras: {}'.format(np.sum(cf))

    # MAKE THE HEATMAP VISUALIZATION
    plt.rcParams["figure.figsize"] = [15, 10]
    plt.rcParams["figure.autolayout"] = True

    if xyticks == False:
        # Do not show categories if xyticks is False/
        categories = False

    sns.heatmap(cf, annot=True, fmt=".2f", cbar_kws=dict(use_gridspec=True, location="left"), cmap='Purples',
                cbar=True, xticklabels=categories, yticklabels=categories)


    # sns.heatmap(cf, annot=True, fmt=".2f", cbar_kws=dict(use_gridspec=True, location="left"), cmap='Purples',
    #             cbar=True, xticklabels=categories, yticklabels=categories)

    # sns.heatmap(cfn, annot=True, fmt=".2f", cmap='Purples', cbar=True, xticklabels=categories, yticklabels='')

    plt.title('Modelo Atual\n\n{} {}'.format(acuracia, ''))

    plt.savefig(output_path)
    plt.clf()


# temp_data = np.asarray([[108, 126, 1, 0, 0, 0], [14, 635, 430, 20, 0, 0], [0, 25, 344, 304, 3, 0], [0, 0, 12, 343, 61, 0], [0, 0, 1, 143, 174, 0], [0, 0, 0, 0, 0, 0]])

def get_best_result(results):
    best_result = None
    best_score = 0

    for result in results:
        score = result['confidence']
        if score > best_score:
            best_result = result

    return best_result


device = 'cuda:0'
classifier = ClassifierV8('/home/diego/2TB/yolo/Trains/v8/meat/MSO_NOVA_AUDITORIA/trains/nano/416/runs/detect/train_no_augmentatio_no_erase_and_no_crop_fraction_no_scale_no_translate_with_arn_v2/weights/best.pt')
# classifier = ClassifierV8('/home/diego/2TB/yolo/Trains/v8/meat/MSO_3.0/trains/nano/416/runs/detect/train_no_augmentatio_no_erase_and_no_crop_fraction_no_scale_no_translate/weights/best.pt')
day = 'mso_producao_nao_normalizado_20240404'
dataset_auditado_path = '/home/diego/2TB/datasets/eco/BOVINOS/DATASETS_AUDITADOS/mso/20240404'

# WHITE_LIST=["AUSENTE", "ESCASSA", "MEDIANA", "UNIFORME",
#                                          "INCLASSIFICAVEL"]
# WHITE_LIST=["AUSENTE", "MEDIANA", "EXCESSIVA"]
WHITE_LIST=["AUSENTE","ESCASSA",  "MEDIANA", "UNIFORME", "EXCESSIVA", "INCLASSIFICAVEL"]

confusion_matrix = np.zeros((len(WHITE_LIST), len(WHITE_LIST))).astype(int)

evaluation_subsets = os.listdir(dataset_auditado_path)

for evaluation_subset in evaluation_subsets:

    if evaluation_subset in WHITE_LIST:
        evaluation_subset_path = os.path.join(dataset_auditado_path, evaluation_subset)

        evaluation_subset_images = os.listdir(evaluation_subset_path)

        for evaluation_day_subset_image in tqdm.tqdm(evaluation_subset_images):
            try:
                evaluation_subset_image_path = os.path.join(evaluation_subset_path, evaluation_day_subset_image)

                image = cv2.imread(evaluation_subset_image_path)

                classification_results = classifier.detect(image)

                best_result = get_best_result(classification_results)

                if best_result is not None:
                    label = best_result['label']

                else:
                    label = 'INCLASSIFICAVEL'

                confusion_matrix[WHITE_LIST.index(evaluation_subset)][WHITE_LIST.index(label)] += 1
            except Exception as ex:
                pass
print('salvando resultados')
output_path = '/home/diego/Desktop/mso/{}_cf.png'.format(day)
make_confusion_matrix_1_side(confusion_matrix,
                             categories=WHITE_LIST, planta=nome_planta, data=day,
                             output_path=output_path
                             )
