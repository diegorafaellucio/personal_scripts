import os
import shutil
import tqdm


def rename_files_in_subset(subset_path):
    subset = os.path.basename(subset_path)
    subset_files = os.listdir(subset_path)
    for subset_file in subset_files:

        new_subset_file_name = "{}_{}".format(subset, subset_file)

        subset_file_path = os.path.join(subset_path, subset_file)
        new_subset_file_path = os.path.join(subset_path, new_subset_file_name)

        shutil.move(subset_file_path, new_subset_file_path)


def process_dataset(dataset_path):
    subsets = os.listdir(dataset_path)

    for subset in tqdm.tqdm(subsets):

        subset_path = os.path.join(dataset_path, subset)

        rename_files_in_subset(subset_path)



if __name__ == '__main__':
    dataset_path = "/home/diego/2TB/tbFcZE-RodoSol-ALPR/images"
    process_dataset(dataset_path)