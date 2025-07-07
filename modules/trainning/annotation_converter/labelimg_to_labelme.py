import json
import os
import shutil

import tqdm
from pascal import annotation_from_xml

if __name__ == "__main__":

    annotations_path = '/home/diego/Downloads/lesoes_frigol/ANNOTATIONS'
    json_annotations_path = '/home/diego/Downloads/lesoes_frigol/ANNOTATIONS_JSON'

    if os.path.exists(json_annotations_path):
        shutil.rmtree(json_annotations_path)
        os.mkdir(json_annotations_path)
    else:
        os.mkdir(json_annotations_path)

    xml_annotations_file_names = os.listdir(annotations_path)

    for xml_annotation_file_name in tqdm.tqdm(xml_annotations_file_names):

        try:
            xml_file_path = os.path.join(annotations_path,xml_annotation_file_name)
            json_file_path = os.path.join(json_annotations_path,xml_annotation_file_name).replace('.xml','.json')

            annotation = annotation_from_xml(xml_file_path)
            labelme_annotation = annotation.to_labelme()
            labelme_annotation['imagePath'] = os.path.join('../IMAGES', xml_annotation_file_name.replace('.xml','.jpg'))
            with open(json_file_path, "w") as f:
                json.dump(labelme_annotation, f, indent=4)
        except:
            continue