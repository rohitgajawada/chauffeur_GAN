import argparse
import multiprocessing
# Import all the test libraries.
import sys
import os
import time


from coil_core import execute_train, execute_validation
#, execute_drive, folder_execute


# You could send the module to be executed and they could have the same interface.


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)


    argparser.add_argument(
        '--single_process',
        default='train',
        type=str
    )
    argparser.add_argument(
        '--gpus',
        nargs='+',
        default='0',
        dest='gpus',
        type=str
    )
    argparser.add_argument(
        '-f',
        '--folder',
        type=str
    )
    argparser.add_argument(
        '-vd',
        '--val_datasets',
        dest='validation_datasets',
        nargs='+',
        default=[]
    )
    argparser.add_argument(
        '--no-train',
        dest='is_training',
        action='store_false'
    )
    argparser.add_argument(
        '-de',
        '--drive_envs',
        dest='driving_environments',
        nargs='+',
        default=[]
    )

    args = argparser.parse_args()




    for gpu in args.gpus:
        try:
            int(gpu)
        except:
            raise ValueError(" Gpu is not a valid int number")



    # Obs this is like a fixed parameter, how much a validation and a train and drives ocupies

    # TODO: MAKE SURE ALL DATASETS ARE " WAYPOINTED "

    # execute_train("0", "eccv", "experiment_1")
    # execute_validation("0", "eccv", "experiment_1", "SeqVal")

    if args.single_process is not None:
        if args.single_process == 'train':
            # TODO make without position, increases the legibility.
            execute_train("1", "eccv", "experiment_unit_task")
            # execute_train("2", "eccv", "experiment_HighWeather12_nolane")

        if args.single_process == 'validation':
            execute_validation("0", "eccv", "experiment_1", "SeqVal")

        # if args.single_process == 'drive':
        #     execute_drive("0", "eccv", "experiment_1", 'Town02')


    # else:

    #     # TODO: of course this change from gpu to gpu , but for now we just assume at least a K40

    #     # Maybe the latest voltas will be underused
    #     # OBS: This usage is also based on my tensorflow experiences, maybe pytorch allows more.
    #     allocation_parameters = {'gpu_value': 3.5,
    #                              'train_cost': 2,
    #                              'validation_cost': 1.5,
    #                              'drive_cost': 1.5}

    #     params = {
    #         'folder': args.folder,
    #         'gpus': list(args.gpus),
    #         'is_training': args.is_training,
    #         'validation_datasets': list(args.validation_datasets),
    #         'driving_environments': list(args.driving_environments),
    #         'allocation_parameters': allocation_parameters
    #     }


    #     folder_execute(params)
