import os
import cv2
import tqdm

from base.src.new_classifier.classifier import Classifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

nome_planta = 'Araguaina'

def make_confusion_matrix(cf,
                          categories='auto',
                          planta = '',
                          data = '',
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
    plt.rcParams["figure.figsize"] = [24, 16]
    plt.rcParams["figure.autolayout"] = True

    fig, (ax1, ax2) = plt.subplots(ncols=2)

    sns.heatmap(cf, annot=True, fmt="", cbar_kws = dict(use_gridspec=True,location="left"), cmap='Purples', cbar=True, xticklabels=categories, yticklabels=categories,  ax=ax1)
    sns.heatmap(cfn, annot=True, fmt=".2f", cmap='Blues', cbar=True, xticklabels=categories, yticklabels='',  ax=ax2)



    plt.suptitle('Avaliação do Modelo Preditivo\n\nPlanta: {}\n\nData: {}'.format(planta, data))

    ax1.set_xlabel(total_de_amostras)
    ax2.set_xlabel(acuracia)

    # ax2.yaxis.tick_right()

    fig.subplots_adjust(wspace=0.001)

    plt.savefig(output_path)
    # plt.show()

def make_confusion_matrix_1_side(cf,
                          categories='auto',
                          planta = '',
                          data = '',
                          xyticks=True,
                          output_path='temp.png'):


    cfn = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]


    accuracy = np.trace(cf) / float(np.sum(cf))

    acuracia = "\nAcurácia={:0.3f}".format(accuracy)

    total_de_amostras = 'Total de amostras: {}'.format(np.sum(cf))

    # MAKE THE HEATMAP VISUALIZATION
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.rcParams["figure.autolayout"] = True

    if xyticks == False:
        # Do not show categories if xyticks is False
        categories = False

    sns.heatmap(cfn, annot=True, fmt=".2f", cmap='Blues', cbar=True, xticklabels=categories, yticklabels='')



    plt.title('Novo Modelo\n\n{} {}'.format(acuracia, total_de_amostras))


    plt.savefig(output_path)
    # plt.show()
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
# device = 'cpu'
classifier = Classifier('/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_novo_carol_nano/weights/best.pt', device=device, max_det=10)
# classifier = Classifier('/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_novo_carol_only3/weights/best.pt', device=device, max_det=10)
# classifier = Classifier('/home/diego/2TB/yolo/new_v5/runs/train/eco_classifier_novo_carol_only2/weights/best.pt', device=device, max_det=10)

image_dataset_path = '/home/diego/2TB/arn'

WHITE_LIST = ['AUSENTE', 'ESCASSA', 'MEDIANA', 'UNIFORME', 'EXCESSIVA']
# WHITE_LIST = ['AUSENTE', 'ESCASSA', 'MEDIANA']
# WHITE_LIST = ['AUSENTE', 'ESCASSA']

evaluation_days = os.listdir(image_dataset_path)
# ['20220829', '20220830', '20220923', '20220924', '20221028', '20230106', '20230301', '20230307',
#                    '20230320', '20230322', '20230412', '20230523', '20230601']

classification_dict = {
    'AUSENTE': 0,
    'ESCASSA': 1,
    'EXCESSIVA': 4,
    'MEDIANA': 2,
    'UNIFORME': 3,
    'INCLASSIFICAVEL': 5
}



for evaluation_day in evaluation_days:
    confusion_matrix = np.zeros((6, 6)).astype(int)
    # confusion_matrix = np.zeros((4, 4)).astype(int)
    # confusion_matrix = np.zeros((3, 3)).astype(int)
    evaluation_day_path = os.path.join(image_dataset_path, evaluation_day)

    evaluation_day_subsets = os.listdir(evaluation_day_path)

    for evaluation_day_subset in evaluation_day_subsets:

        if evaluation_day_subset in WHITE_LIST:
            evaluation_day_subset_path = os.path.join(evaluation_day_path, evaluation_day_subset)

            evaluation_day_subset_images = os.listdir(evaluation_day_subset_path)

            for evaluation_day_subset_image in tqdm.tqdm(evaluation_day_subset_images):
                evaluation_day_subset_image_path = os.path.join(evaluation_day_subset_path, evaluation_day_subset_image)

                image = cv2.imread(evaluation_day_subset_image_path)

                classification_results = classifier.detect(image)

                best_result = get_best_result(classification_results)

                gt_index = classification_dict[evaluation_day_subset]

                if best_result is not None:
                    label = best_result['label']

                    predicted_index = classification_dict[label]
                else:
                    predicted_index = classification_dict['INCLASSIFICAVEL']

                confusion_matrix[gt_index][predicted_index] += 1
    print('salvando resultados')
    output_path = '/home/diego/Desktop/arn/ausente_escassa_avaliacao_{}.png'.format(evaluation_day)
    # make_confusion_matrix_1_side(confusion_matrix,
    #                       categories=["AUSENTE", "ESCASSA", "UNIFORME", "MEDIANA", "EXCESSIVA", "INCLASSIFICAVEL"], planta=nome_planta, data=evaluation_day, output_path=output_path
    #                       )

    # make_confusion_matrix_1_side(confusion_matrix,
    #                              categories=["AUSENTE", "ESCASSA", "MEDIANA", "INCLASSIFICAVEL"], planta=nome_planta, data=evaluation_day,
    #                              output_path=output_path
    #                              )

    make_confusion_matrix_1_side(confusion_matrix,
                                 categories=["AUSENTE", "ESCASSA"], planta=nome_planta,
                                 data=evaluation_day,
                                 output_path=output_path
                                 )
