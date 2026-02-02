DEBUG_MODE = True
USE_CUDA = True
# USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")
import logging
from new_version.utils.utils import create_logger, copy_all_src
from new_version.CVRP.CVRPTester import CVRPTester as Tester
import time


##########################################################################################
# parameters

b = os.path.abspath(".").replace('\\', '/')

mode = 'test'

test_data_path = b+"/../data/"

# baseline_path = b+"/data/" + file_name + '/LKH3_runs1.txt'

append_information = [True, True, True, False, True, True, False, False, False, False, False, False, False]
#                      0      1     2    3      4      5      6      7      8      9     10   11    12
# 0.distance_to_current,  1.average_distance_to_unvisited,  2.std_dev_distance_to_unvisited,  3.distance_to_destination, 4.sin_to_destination, 
# 5.cos_to_destination,   6.average_distance_to_all         7.std_dev_distance_to_all         8.first_node               9 knn_mean                    10. knn_std
# 11. knn_mean_demand     12. knn_std_demand

globle_params = {
    "test_mode": "aug_test"
}

env_params = {
    'data_path':test_data_path,
    # 'baseline_path': baseline_path,
    'mode': mode,
    'append_information': append_information,
    'pomo_size': 4,
    'aggregation_nums': 10,
    'test_mode' : globle_params["test_mode"]
}

model_params = {
    'mode': mode,
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'decoder_layer_num':3,
    'qkv_dim': 16,
    'head_num': 8,
    'ff_hidden_dim': 512,
    'append_information': append_information,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        # 'lr': 1e-5,
        'weight_decay': 1e-6
                 },
    'scheduler': {
        'milestones': [1 * i for i in range(1, 150)],
        'gamma': 0.97
        # 'milestones': [501,],
        # 'gamma': 0.1
                 }
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'test_mode' : globle_params["test_mode"],
    'epochs': 150,
    'test_episodes': 200,
    'test_batch_size': 10,
    'loop_in_one_epoch': 1,
    'beam_size': 100,
    'keep_threshold': 2,
    'logging': {
        'model_save_interval': 1,
        'img_save_interval': 3000,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
               },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
               },
               },
    'model_load': {
        'enable': True,  # enable loading pre-trained model
        # 'path': './result/20240723_220941_train',  # directory path of pre-trained model and log files saved.
        # 'file': "checkpoint-25.pt",  # epoch version of pre-trained model to laod.
        'path': './pretrain',
        'file': "checkpoint-100.pt"
                  },
    }

valid_params = {
    'sgbs_beta': 10,
    'sgbs_gamma_minus1': (10-1),
    'aug_factor':8,
    'valid_batch_size': 4
}

logger_params = {
    'log_file': {
        'desc': 'train',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)

    # sizes = [100,1000,5000,10000]
    # pomo_size = [64,32,16,8]
    # num_instance = [2000,200,20,20]
    # distributions = ['uniform', 'clustered1', 'clustered2', 'explosion', 'implosion']

    sizes = [50,500,5000]
    aug_size = [64,32,16]
    num_instance_all = [2000,200,20]
    test_num_instance = [500,50,5]
    test_batch_size = [50,25,1]
    test_mode = ["aug_test","aug_test", "aug_test"]
    distributions = ['uniform', 'clustered1', 'explosion', 'implosion']

    # sizes = [50]
    # pomo_size = [16]
    # num_instance_all = [2000]
    # test_num_instance = [2000]
    # test_batch_size = [100]
    # distributions = ['uniform', 'clustered1', 'clustered2', 'explosion', 'implosion']

    # sizes = [5000]
    # aug_size = [16]
    # num_instance_all = [20]
    # test_num_instance = [5]
    # test_batch_size = [1]
    # test_mode = ["aug_test"]
    # distributions = ['uniform', 'clustered1', 'explosion', 'implosion']

    for distribution in distributions:
        for i in range(len(sizes)):
            size = sizes[i]
            tester_params['problem_size'] = size
            env_params['aug_size'] = aug_size[i]
            tester_params['test_episodes_all'] = num_instance_all[i]
            tester_params['test_episodes'] = test_num_instance[i]
            tester_params['test_batch_size'] = test_batch_size[i]

            tester_params['test_mode'] = test_mode[i]
            env_params['test_mode'] = test_mode[i]


            env_params['distribution'] = distribution

            # env_params['data_path'] = test_data_path
            # env_params['baseline_path'] = baseline_path

            _print_config()

            trainer = Tester(env_params=env_params,
                            model_params=model_params,
                            optimizer_params=optimizer_params,
                            tester_params=tester_params,
                            valid_params=valid_params)

            copy_all_src(trainer.result_folder, os.path.dirname(os.path.abspath(__file__)))

            start_time = time.time()

            avg_gap = trainer.run()

            end_time = time.time()
            during_time = end_time - start_time

            logger = logging.getLogger('root')
            logger.info('distribution: {} size: {} avg_gap: {}% time: {}s, {}m'.format(distribution, size, avg_gap * 100, during_time,during_time/60))




def _set_debug_mode():
    global trainer_params

    tester_params['test_batch_size'] = 1
    tester_params['beam_size'] = 4
    tester_params['knn'] = 15
    tester_params['depot_knn'] = 15
    valid_params['valid_batch_size'] = 1
    

def _print_config():
    logger = logging.getLogger('root')
    
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":

    main()

