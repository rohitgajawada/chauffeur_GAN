import argparse
import multiprocessing
# Import all the test libraries.
import sys
import os
import time


from coil_core import execute_train, execute_validation, execute_drive, folder_execute, trainGAN, testGAN, trainnoGAN, trainnewGAN, traingoodGAN, traingoodACGAN


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
        dest='gpus',
        default='0',
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

    if args.single_process is not None:
        if args.single_process == 'train':
            # TODO make without position, increases the legibility.
            # trainnewGAN.execute("8", "eccv", "experiment_2_kya_noskip")
            traingoodACGAN.execute("0", "eccv", "experiment_2_kya_ACGAN_LSGAN_noskip_with_higher_l1")
            # trainnoGAN.execute("9", "eccv", "experiment_2")
            # trainGAN.execute("9", "eccv", "experiment_1")

        if args.single_process == 'validation':
            execute_validation("0", "eccv", "experiment_1", "SeqVal")

        if args.single_process == 'drive':
            execute_drive("0", "eccv", "experiment_1", 'Town02')


    else:

        # TODO: of course this change from gpu to gpu , but for now we just assume at least a K40

        # Maybe the latest voltas will be underused
        # OBS: This usage is also based on my tensorflow experiences, maybe pytorch allows more.
        allocation_parameters = {'gpu_value': 3.5,
                                 'train_cost': 2,
                                 'validation_cost': 1.5,
                                 'drive_cost': 1.5}

        params = {
            'folder': args.folder,
            'gpus': list(args.gpus),
            'is_training': args.is_training,
            'validation_datasets': list(args.validation_datasets),
            'driving_environments': list(args.driving_environments),
            'allocation_parameters': allocation_parameters
        }


        folder_execute(params)
