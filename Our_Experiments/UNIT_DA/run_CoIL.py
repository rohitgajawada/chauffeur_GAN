import argparse
import multiprocessing
# Import all the test libraries.
import sys
import os
import time


from coil_core import execute_train, execute_validation, execute_drive, folder_execute, testGAN, trainnoGAN, trainnewGAN, trainnewGAN_task, trainWGAN_task, train_da, train_wdgrl, train_wdgrl_withoutgen, train_da_no5, train_da_orig, train_da_shared


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
            # trainnewGAN_task.execute("6", "eccv", "exp_pretrained_F_IL")
            # train_da.execute("8", "eccv", "all_da_v2")
            # train_da_shared.execute("3", "eccv", "all_da_shared_E1E12_l1_25")
            # train_da_shared.execute("0", "eccv", "all_da_shared_E1E12_l1_25_aug")
            # train_da_shared.execute("3", "eccv", "all_da_shared_E1E12_l1_25")
            # train_wdgrl.execute("5", "eccv", "all_da_aug_orig_E1E12_l1_10")
            # train_da_no5.execute("8", "eccv", "all_da_aug_no5_E1E12")
            # testGAN.execute("7", "eccv", "all_da_orig")
            # trainnewGAN.execute("4", "eccv", "exp_reconstruct")
            # trainnewGAN.execute("4", "eccv", "experiment_1")

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
