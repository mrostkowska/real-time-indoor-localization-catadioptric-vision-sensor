from pathlib import Path

home_path = Path.home()
current_path = Path('/mnt/data')
data_folder_path = current_path/'data'

split_dataset_train_path = data_folder_path/'train_omni_augmented'
split_dataset_validation_path = data_folder_path/'val_omni_augmented'
split_dataset_embedings_path =  data_folder_path/'embedings'
split_dataset_test_path =  data_folder_path/'test'

model_path = current_path/'models'

log_path = current_path/'logs'

results_path = current_path/'results'

def path_as_string(path: Path):
  return path.as_posix()