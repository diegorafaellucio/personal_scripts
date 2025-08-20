import os
import shutil
import math
import random


def fileGenerator(data, file_name):
    file = open(file_name, "a+")
    for i in range(len(data)):
        file.write("{}\n".format(data[i][0]))
    file.close()


def generateDatabase(dataset_path, data_output_path):
    eval = 0.1
    test = 0.2
    train = 0.7

    path_elements = dataset_path.split('/')

    if 'DATA' not in data_output_path:
        data_path = os.path.join(data_output_path, "{}_{}".format(path_elements[-2], path_elements[-1]), 'DATA')
    else:
        data_path = data_output_path

    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

    eval_file = os.path.join(data_path, "eval.txt")
    train_file = os.path.join(data_path, "train.txt")
    test_file = os.path.join(data_path, "test.txt")

    eval_data = []
    train_data = []
    test_data = []

    images = []

    images_path = os.path.join(dataset_path,'IMAGES')

    # images = images + [os.path.join(images_path,file) for file in os.listdir(images_path) if 'jpg' in file or 'png' in file]
    images = images + [os.path.join(images_path,file) for file in os.listdir(images_path) if 'jpg' in file or 'png' in file]

    random.shuffle(images)

    files_amount = len(images)

    eval_amount = int(math.floor(files_amount * eval))
    train_amount = int(math.floor(files_amount * train))
    test_amount = int(math.floor(files_amount * test))

    for counter, image in enumerate(images):
        print('{}/{}'.format(counter, len(images)))
        if len(eval_data) < eval_amount:
            eval_data.append([image])
        elif len(train_data) < train_amount:
            train_data.append([image])
        else:
            test_data.append([image])

    fileGenerator(eval_data, eval_file)
    fileGenerator(train_data, train_file)
    fileGenerator(test_data, test_file)


def generateDatabaseFromMultiplePaths(dataset_paths, data_output_path):
    """
    Generate database files (train, test, eval) from multiple dataset paths.
    
    Args:
        dataset_paths (list): List of paths to datasets
        data_output_path (str): Output directory for the generated files
    """
    eval = 0.1
    test = 0.2
    train = 0.7
    
    # Create a unique folder name based on the last dataset path
    # You might want to customize this naming logic
    last_path = dataset_paths[-1].split('/')
    
    if 'DATA' not in data_output_path:
        data_path = os.path.join(data_output_path, "multiple_datasets_{}".format(last_path[-1]), 'DATA')
    else:
        data_path = data_output_path

    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)

    eval_file = os.path.join(data_path, "eval.txt")
    train_file = os.path.join(data_path, "train.txt")
    test_file = os.path.join(data_path, "test.txt")

    eval_data = []
    train_data = []
    test_data = []

    # Collect all images from all dataset paths
    all_images = []
    
    for dataset_path in dataset_paths:
        print(f"Processing dataset: {dataset_path}")
        images_path = os.path.join(dataset_path, 'IMAGES')
        
        if os.path.exists(images_path):
            dataset_images = [os.path.join(images_path, file) for file in os.listdir(images_path) 
                             if 'jpg' in file or 'png' in file]
            all_images.extend(dataset_images)
            print(f"Found {len(dataset_images)} images in {dataset_path}")
        else:
            print(f"WARNING: Images path not found: {images_path}")
    
    # Shuffle all images to ensure random distribution
    random.shuffle(all_images)
    
    files_amount = len(all_images)
    print(f"Total images collected: {files_amount}")
    
    # Calculate amounts for each set
    eval_amount = int(math.floor(files_amount * eval))
    train_amount = int(math.floor(files_amount * train))
    test_amount = int(math.floor(files_amount * test))
    
    # Distribute images into sets
    for counter, image in enumerate(all_images):
        print('{}/{}'.format(counter, len(all_images)))
        if len(eval_data) < eval_amount:
            eval_data.append([image])
        elif len(train_data) < train_amount:
            train_data.append([image])
        else:
            test_data.append([image])
    
    # Generate output files
    fileGenerator(eval_data, eval_file)
    fileGenerator(train_data, train_file)
    fileGenerator(test_data, test_file)
    
    print(f"Generated files at: {data_path}")
    print(f"Eval: {len(eval_data)} images")
    print(f"Train: {len(train_data)} images")
    print(f"Test: {len(test_data)} images")


# Example of using the original function
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PGO/1.0', "/home/diego/2TB/TREINOS")

# Example of using the new function with an array of dataset paths

# generateDatabaseFromMultiplePaths(['/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/6.0'], "/home/diego/2TB/TREINOS/BETTER_BEEF_6.0/DATA")

# Other examples commented out
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/RLM/1.0', "/home/diego/2TB/TREINOS")
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BLN/1.0', "/home/diego/2TB/TREINOS")
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/MSO/1.0', "/home/diego/2TB/TREINOS")
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/BARRA_MANSA/BM/9.0', "/home/diego/2TB/TREINOS")
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/FRIGOL/LP/1.0', "/home/diego/2TB/TREINOS")
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/ARN/2.0', "/home/diego/2TB/TREINOS")
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/RIO_MARIA/RM/2.0', "/home/diego/2TB/TREINOS")
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/PRN/1.0', "/home/diego/2TB/TREINOS")
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/MINERVA/BTS/1.0', "/home/diego/2TB/TREINOS")
# generateDatabase('/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/5-BRUISE/TRAIN/ECOTRACE/SULBEEF/SB/1.0', "/home/diego/2TB/TREINOS")


# generateDatabase('/home/diego/2TB/datasets/COGTIVE/MAURICEA_MONITORAMENTO/2.0', "/home/diego/2TB/datasets/COGTIVE/MAURICEA_MONITORAMENTO/2.0/DATA")
# generateDatabase('/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/4.0', "/home/diego/2TB/TREINOS/BETTER_BEEF_4.0")

# generateDatabase('/home/diego/2TB/datasets/COGTIVE/Pancristal/1.0', "/home/diego/2TB/datasets/COGTIVE/Pancristal/1.0/DATA")
# generateDatabase('/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0', "/home/diego/2TB/datasets/COGTIVE/KOVI/PLATE_DETECTION/1.0/DATA")
generateDatabase('/home/diego/2TB/datasets/COGTIVE/BIG_CHARQUE/2.0', "/home/diego/2TB/datasets/COGTIVE/BIG_CHARQUE/2.0/DATA")

