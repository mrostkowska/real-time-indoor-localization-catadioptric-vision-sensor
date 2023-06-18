# __all__ = ['paths_definitions']

from .paths_configs import (
    home_path, 
    current_path, 
    
    data_folder_path,
    split_dataset_train_path, 
    split_dataset_test_path,
    split_dataset_validation_path,
    split_dataset_embedings_path, 

    model_path,
    
    path_as_string,

    log_path,

    results_path,
) 

from .configs import (
    NUM_CLASSES,
    top_dropout_rate,
    pricture_resolution,
    IMG_SIZE,
    img_input_size,
    batch_size,
    current_model_name,
    model_name,
    image_interpolation
)

