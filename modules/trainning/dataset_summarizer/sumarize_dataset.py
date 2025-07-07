import csv
import os


if __name__ == '__main__':
    dataset_path = '/home/diego/2TB/datasets/GCP/eco/bovinos/3-MEAT'
    output_classes = ['AUSENTE', 'ESCASSA', 'MEDIANA', 'UNIFORME', 'EXCESSIVA']

    dataset_types = os.listdir(dataset_path)

    output_data = []

    for dataset_type in dataset_types:
        dataset_type_path = os.path.join(dataset_path, dataset_type)

        dataset_creators = os.listdir(dataset_type_path)

        for dataset_creator in dataset_creators:

            dataset_creator_path = os.path.join(dataset_type_path, dataset_creator)

            clients = os.listdir(dataset_creator_path)

            for client in clients:

                client_path = os.path.join(dataset_creator_path, client)

                plants = os.listdir(client_path)

                for plant in plants:

                    plant_path = os.path.join(client_path, plant)

                    versions = os.listdir(plant_path)

                    for version in versions:

                        years_data = {}
                        sif_counter = {}
                        amount_per_class = {}
                        amount_of_images = 0

                        version_path = os.path.join(plant_path, version)

                        classification_classes = os.listdir(version_path)

                        for classification_class in classification_classes:

                            classification_class_path = os.path.join(version_path, classification_class)

                            images_path = os.path.join(classification_class_path, 'IMAGES')

                            images = [image for image in os.listdir(images_path) if '.jpg' in image]

                            for image in images:
                                amount_of_images += 1

                                if classification_class not in amount_per_class:
                                    amount_per_class[classification_class] = 1
                                else:
                                    amount_per_class[classification_class] += 1

                                image_name_elements = image.split('-')

                                date_element = image_name_elements[0]
                                year = date_element[0:4]
                                month = date_element[4:6]
                                day = date_element[6:8]

                                sif = image_name_elements[-1].split('.')[0]

                                # if year not in years_data:
                                #     years_data[year] = {month: [day]}
                                # else:
                                #     if month not in years_data[year]:
                                #         years_data[year][month] = [day]
                                #     else:
                                #         if day not in years_data[year][month]:
                                #             years_data[year][month].append(day)

                                if year not in years_data:
                                    years_data[year] = [month]
                                else:
                                    if month not in years_data[year]:
                                        years_data[year].append(month)

                                if sif not in sif_counter:
                                    sif_counter[sif] = 1
                                else:
                                    sif_counter[sif] += 1

                        if len(output_data) == 0:
                            header = ['TIPO DO DATASET', 'CRIADOR DO DATASET', 'CLIENTE', 'PLANTA', 'SIF', 'VERSAO', 'TOTAL DE IMAGENS']

                            for output_class in output_classes:
                                header.append(output_class)
                            header.append('PERIODO')
                            header.append('CAMINHO DO DATASET')

                            output_data.append(header)

                        sif_counter = dict(sorted(sif_counter.items(), key=lambda item: item[1], reverse=True))

                        if len(sif_counter) == 0:
                            sif = '0000'
                        else:
                            sif, _ = next(iter(sif_counter.items()))
                            sif = sif.zfill(4)

                        data = [dataset_type, dataset_creator, client, plant, sif, str(version)+" ", amount_of_images]

                        dataset_bucket_path = os.path.join('gs://',dataset_path.split('/')[-1],dataset_type, dataset_creator, client, plant, version)


                        for output_class in output_classes:
                            if output_class in amount_per_class:
                                data.append(amount_per_class[output_class])
                            else:
                                data.append(0)

                        year_info = ''
                        for year in years_data:

                            months = ','.join(years_data[year])
                            year_info += '{}: {}; '.format(year, months)

                        data.append(year_info)
                        data.append(dataset_bucket_path)

                        output_data.append(data)




    with open('dataset_summary.csv', 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerows(output_data)
    print()

