import os
import shutil
import math
import random

subset_blacklist = ['INCLASSIFICAVEL', 'DATA', 'BANDA_A_ANGULADA', 'BANDA_B_ANGULADA', 'ANNOTATIONS',
                    'ANNOTATIONS_JSON']


# subset_blacklist = ['DATA', 'EXCESSIVA']

def fileGenerator(data, file_name):
    file = open(file_name, "a+")
    for i in range(len(data)):
        file.write("{}\n".format(data[i][0]))
    file.close()


def generateDatabase(dataset_path):
    eval = 0.1
    test = 0.2
    train = 0.7

    dataset_dir = os.path.join(dataset_path, 'DATA')

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    eval_file = os.path.join(dataset_dir, "eval.txt")
    train_file = os.path.join(dataset_dir, "train.txt")
    test_file = os.path.join(dataset_dir, "test.txt")

    eval_data = []
    train_data = []
    test_data = []

    images = []

    for root, subsets, _ in os.walk(dataset_path):

        for subset in subsets:
            if subset not in subset_blacklist:

                person_path = os.path.join(root, subset)

                person_images_path = os.path.join(person_path, 'IMAGES')

                if os.path.exists(person_images_path):
                    subset_images = [os.path.join(person_images_path, file) for file in os.listdir(person_images_path)
                                     if 'jpg' in file or 'png' in file]
                    images.extend(subset_images)
                    print(f"Found {len(subset_images)} images in {person_path}")
        break

    random.shuffle(images)

    files_amount = len(images)
    print(f"Total images: {files_amount}")

    eval_amount = int(math.floor(files_amount * eval))
    train_amount = int(math.floor(files_amount * train))
    test_amount = int(math.floor(files_amount * test))

    for image in images:
        if len(eval_data) < eval_amount:
            eval_data.append([image])
        elif len(train_data) < train_amount:
            train_data.append([image])
        else:
            test_data.append([image])

    fileGenerator(eval_data, eval_file)
    fileGenerator(train_data, train_file)
    fileGenerator(test_data, test_file)

    print(f"Generated files at: {dataset_dir}")
    print(f"Eval: {len(eval_data)} images")
    print(f"Train: {len(train_data)} images")
    print(f"Test: {len(test_data)} images")


def generateDatabaseFromMultiplePaths(dataset_paths, output_path=None):
    """
    Generate database files (train, test, eval) from multiple dataset paths.
    
    Args:
        dataset_paths (list): List of paths to datasets
        output_path (str, optional): Output directory for the generated files.
                                    If None, creates DATA directory in the last dataset path.
    """
    eval = 0.1
    test = 0.2
    train = 0.7

    # Determine output directory
    if output_path is None:
        dataset_dir = os.path.join(dataset_paths[-1], 'DATA')
    else:
        dataset_dir = output_path
        if 'DATA' not in dataset_dir:
            dataset_dir = os.path.join(output_path, 'DATA')

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    eval_file = os.path.join(dataset_dir, "eval.txt")
    train_file = os.path.join(dataset_dir, "train.txt")
    test_file = os.path.join(dataset_dir, "test.txt")

    eval_data = []
    train_data = []
    test_data = []

    # Collect all images from all dataset paths
    all_images = []

    for dataset_path in dataset_paths:
        print(f"Processing dataset: {dataset_path}")
        images_from_dataset = []

        for root, subsets, _ in os.walk(dataset_path):
            for subset in subsets:
                if subset not in subset_blacklist:
                    person_path = os.path.join(root, subset)
                    person_images_path = os.path.join(person_path, 'IMAGES')

                    if os.path.exists(person_images_path):
                        subset_images = [os.path.join(person_images_path, file) for file in
                                         os.listdir(person_images_path)
                                         if 'jpg' in file or 'png' in file]
                        images_from_dataset.extend(subset_images)
                        print(f"Found {len(subset_images)} images in {subset} from {dataset_path}")
            break  # Only process the top level directories

        all_images.extend(images_from_dataset)
        print(f"Total images from {dataset_path}: {len(images_from_dataset)}")

    # Shuffle all images to ensure random distribution
    random.shuffle(all_images)

    files_amount = len(all_images)
    print(f"Total combined images: {files_amount}")

    # Calculate amounts for each set
    eval_amount = int(math.floor(files_amount * eval))
    train_amount = int(math.floor(files_amount * train))
    test_amount = int(math.floor(files_amount * test))

    # Distribute images into sets
    for counter, image in enumerate(all_images):
        if counter % 1000 == 0:
            print(f'Processed {counter}/{len(all_images)} images')

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

    print(f"Generated files at: {dataset_dir}")
    print(f"Eval: {len(eval_data)} images")
    print(f"Train: {len(train_data)} images")
    print(f"Test: {len(test_data)} images")


# Example of using the original function
# generateDatabase('/home/diego/2TB/datasets/COGTIVE/BETTER_BEEF/3.0')

# Example of using the new function with an array of dataset paths

# Use this to save the output in a custom location
# generateDatabaseFromMultiplePaths(dataset_paths, '/home/diego/2TB/TREINOS/combined_dataset')

# Or use this to save in the default location (DATA folder in the last dataset path)
generateDatabaseFromMultiplePaths(
    [
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/ARN/5.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BLN/3.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/BTS/5.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/BARRA_MANSA/BM/4.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/JBO/1.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/JNB/1.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/FRIGOL/LP/1.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/MSO/7.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PGO/3.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/PRN/4.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/RIO_MARIA/RM/1.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/MINERVA/RLM/7.0',
     '/home/diego/2TB/datasets/ECOTRACE/GCP/eco/bovinos/3-MEAT/TRAIN/ECOTRACE/SUL_BEEF/SB/5.0',
     ],
    '/home/diego/2TB/TREINOS/MODELO_UNICO_V10/DATA')
