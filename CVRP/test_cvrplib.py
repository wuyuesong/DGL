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
from DGL.utils.utils import create_logger, copy_all_src, cvrplib_collections, parse_cvrplib_name, load_cvrplib_file, get_dist_matrix, normalize_nodes_to_unit_board, choose_bsz, calculate_tour_length_by_dist_matrix, avg_list
from DGL.CVRP.CVRPTester_cvrplib import CVRPTester as Tester
from pathlib import Path
import time
import torch
import math

import matplotlib.pyplot as plt



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
        'path': "./pretrain",
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

    # aug_size = [64,32,16]
    # num_instance_all = [2000,200,20]
    # test_num_instance = [500,50,5]
    # test_batch_size = [50,25,1]
    # test_mode = ["aug_test","aug_test", "aug_test"]
    # distributions = ['uniform', 'clustered1', 'explosion', 'implosion']

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
    
    st1 = []
    st2 = []
    st3 = []
    st4 = []
    
    cvrplib_names = list(cvrplib_collections.keys())
    cvrplib_names.sort(key=lambda x: parse_cvrplib_name(x)[1])
    
    for i in range(len(cvrplib_names)):
        name = cvrplib_names[i]
        opt_len = cvrplib_collections[name]
        _, load_size = parse_cvrplib_name(name)

        # prepare env
        depot, nodes, demands, capacity, name = load_cvrplib_file(Path("../data"), name)
        
        demands = demands.to("cuda:0")
        size = nodes.size(0)
        assert size == load_size
        depot_nodes = torch.cat((nodes, depot.unsqueeze(dim=0)), dim=0)
        dist_matrix = get_dist_matrix(depot_nodes).to("cuda:0")

        normalized_depot_nodes = normalize_nodes_to_unit_board(depot_nodes)
        size = nodes.size(0)
        aug_size = choose_bsz(size)
        normalized_instance = torch.tensor(normalized_depot_nodes).float().to("cuda:0")

        print("size: ", size)
        print("depot_nodes.shape", depot_nodes.shape)
        test_data_path = b + "/tmp_lib.txt"
        with open('tmp_lib.txt', 'w') as file:
            file.write(str(normalized_instance[-1][0].item()) + ',' + str(normalized_instance[-1][1].item()) + ' ')
            file.write(".|. ")
            
            for i in range(normalized_instance.shape[0] - 1):
                file.write(str(normalized_instance[i][0].item()) + ',' + str(normalized_instance[i][1].item()) + ' ')
                
            file.write(".|. ")
            for i in range(demands.shape[0]):
                file.write(str(demands[i].item()) + " ")
                
            file.write(".|. ")
            file.write(str(capacity.item()))

        tester_params['problem_size'] = size
        env_params['aug_size'] = aug_size
        print("aug_size", aug_size)
        tester_params['test_episodes_all'] = 1
        tester_params['test_episodes'] = 1
        tester_params['test_batch_size'] = 1

        tester_params['test_mode'] = "aug_test"
        env_params['test_mode'] = "aug_test"

        env_params['data_path'] = test_data_path
        env_params['baseline_path'] = ""

        _print_config()

        trainer = Tester(env_params=env_params,
                        model_params=model_params,
                        optimizer_params=optimizer_params,
                        tester_params=tester_params,
                        valid_params=valid_params)

        copy_all_src(trainer.result_folder, os.path.dirname(os.path.abspath(__file__)))

        start_time = time.time()

        avg_gap, shortest_tours, shortest_tours_flag = trainer.run()

        end_time = time.time()
        during_time = end_time - start_time
        
        shortest_tours = shortest_tours.squeeze(0)
        shortest_tours_flag = shortest_tours_flag.squeeze(0)
        tour = []
        for i in range(shortest_tours.shape[0]):
            if shortest_tours_flag[i] == 1:
                tour.append(torch.tensor(size))
            tour.append(shortest_tours[i] - 1)
            
        # print("tour:", tour)
        # print("size: ", size)
        # print("depot sum: ", sum(shortest_tours_flag))
        # print("len(tour): ", len(tour))
        
        best_tour = torch.tensor(tour)
        
        # print("====best_tour======")
        # print(best_tour)
        
        tour_len = calculate_tour_length_by_dist_matrix(dist_matrix, best_tour).item()
        tour_len = math.ceil(tour_len)
        gap = tour_len / opt_len - 1
        
        
        # draw(depot_nodes , best_tour)
        
        if size <= 200:
            st1.append(gap)
        elif size <= 500:
            st2.append(gap)
        else:
            st3.append(gap)
            
        print(f"\n\n")
        print(f"TSP 1~200     : {len(st1)} instances, "
            f"gap {avg_list(st1) * 100:.3f}%")
        print(f"TSP 201~500  : {len(st2)} instances, "
            f"gap {avg_list(st2) * 100:.3f}%")
        print(f"TSP >500    : {len(st3)} instances, "
            f"gap {avg_list(st3) * 100:.3f}%")
            

        logger = logging.getLogger('root')
        with open('output.txt', 'a') as file:
            file.write(name + " score " + str(tour_len)  + " opt_len " + str(opt_len) + " gap " + str(gap) + " time " + str(during_time) + " s" "\n")
        # logger.info('distribution: {} size: {} avg_gap: {}% time: {}s, {}m'.format(distribution, size, avg_gap * 100, during_time,during_time/60))


def draw(points_tensor, order):
    points_tensor = points_tensor.cpu().numpy()
    order = order.cpu().numpy()
    
    # 提取x和y坐标
    x = points_tensor[:, 0]
    y = points_tensor[:, 1]
    

    # 设置坐标轴的范围，确保所有点都显示在图表内
    plt.xlim(min(x) - 1, max(x) + 1)
    plt.ylim(min(y) - 1, max(y) + 1)

    # 绘制所有点，使用蓝色圆点标记
    plt.scatter(x, y, color='blue', label='Points')
    
    # 按照访问顺序连接点，使用灰色虚线，半透明效果，线条较细
    x_ordered = x[order]
    y_ordered = y[order]
    plt.plot(x_ordered, y_ordered, '--o', color='gray', alpha=0.5, linewidth=1, label='Path')

    # 在每个点上标注访问顺序，使用红色数字，字体稍大
    for i, txt in enumerate(order):
        plt.annotate(txt, (x_ordered[i], y_ordered[i]), color='red', fontsize=12, xytext=(5, -5), textcoords='offset points')

    # 标记起点和终点，分别使用绿色和黄色圆点，并添加标签，点的大小稍大
    plt.scatter(x_ordered[0], y_ordered[0], color='green', marker='s', label='Start', s=100)
    plt.scatter(x_ordered[-1], y_ordered[-1], color='yellow', marker='^', label='End', s=100)

    # 添加箭头表示路径的方向，使用蓝色箭头，箭头大小适中，线条宽度较细，透明度稍低
    for i in range(len(order) - 1):
        dx = x_ordered[i+1] - x_ordered[i]
        dy = y_ordered[i+1] - y_ordered[i]
        plt.arrow(x_ordered[i], y_ordered[i], dx, dy, color='blue', length_includes_head=True, head_width=0.4, head_length=0.4, alpha=0.6, linewidth=0.5)

    plt.savefig('out.pdf',bbox_inches='tight', pad_inches=0)    



def _set_debug_mode():
    global trainer_params

    tester_params['test_batch_size'] = 1
    tester_params['beam_size'] = 16
    tester_params['knn'] = 50
    tester_params['depot_knn'] = 50
    valid_params['valid_batch_size'] = 1
    

def _print_config():
    logger = logging.getLogger('root')
    
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":

    main()

