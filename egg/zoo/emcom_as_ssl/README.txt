TRAIN A MODEL

To conduct the experiments in the paper we used nest, the job launcher provided by EGG. 
Nest can submit jobs on a slurm-based cluster and game-specific parameters can be passed through a .json file. 
The parameters used for the 4 games configuration and the simclr baseline can be found at EGG/egg/zoo/emcom_as_ssl/sweeps
To train a model and reproduce our results make sure to change the "dataset_dir": ["PATH TO IMAGENET TRAINING DATA"] option
in the .json file of interest, with the path to the folder containing the ImageNet train data.

To reproduce the expriments from the paper using 16 GPUs the following command should be launched from the EGG root directory specifying a checkpoint directory, a slurm partition and the specific json:

$ python egg/nest/nest.py --game=egg.zoo.emcom_as_ssl.train --nodes=2 --tasks=8 --partition=<SPECIFY_SLURM_PARTITION> --sweep=egg/zoo/emcom_as_ssl/paper_sweeps/<ADD_JSON_FILE> --checkpoint_dir="<PATH_TO_CHECKPOINTING_DIR>" --checkpoint_freq=5


At the end of training, in the specified checkpoint_dir directory, under a folder named after the job id, a file named "final.tar" will contain
the trained model. This file can be used to run an evaluation script for one of the two test sets or the gaussian sanity check.


GAUSSIAN EVALUATION

To perform the gaussian evaluation to a trained model the following command should be executed taking care of the following options:

* --loss_type: use "xent" for a communication game model and "ntxent" for a simclr model
* --shared_vision: set this options for configurations that used a shared CNN (inluding simclr)
* --simclr_sender: set this option when evaluating a simclr baseline
* --discrete_evaluation_simclr: set this option when evaluating a simclr baseline


$ python -m egg.zoo.emcom_as_ssl.scripts.gaussian_noise_analysis --loss_type="xent" --checkpoint_path="./checkpoints/game_augmentations_shared/final.tar" --shared_vision


Accuracy will be printed to stdout, please refer to "game_accuracy".
Please note that evalution scripts are supported for single-gpu inference only.


TEST SET EVALUATION:

To feed the ILSVRC validation set (i_test) or the OOD set (o_test) to a trained model the following command should be executed taking care of
the following options:
* --loss_type: use "xent" for a communication game model and "ntxent" for a simclr model
* --shared_vision: set this options for configurations that used a shared CNN (inluding simclr)
* --test_set: use "i_test" for ILSVRC-val or "o_test" for the OOD set. Under EGG/egg/zoo/emcom_as_ssl/scripts/utils.py make sure to set the right
path to the folder containing the ILSVRC images (line 17) and the OOD images (line 18)
* --simclr_sender: set this option when evaluating a simclr baseline
* --discrete_evaluation_simclr: set this option when evaluating a simclr baseline


$ python -m egg.zoo.emcom_as_ssl.scripts.imagenet_validation_analysis --loss_type="xent" --checkpoint_path="checkpoints/game/augmentations_no_shared_vision/final.tar" --test_set=i_test


Accuracy will be printed to stdout, please refer to "game_accuracy".
Please note that evalution scripts are supported for single-gpu inference only. 