
"""
The MIT License

Copyright (c) 2021 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import json
import logging
import logging.config
import os
import shutil
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch

process_start_time = datetime.now(pytz.timezone("Asia/Seoul"))
b = os.path.abspath('.')
result_folder = b+'/result/' + process_start_time.strftime("%Y%m%d_%H%M%S") + '{desc}'


tsplib_collections = {
    'eil51': 426,
    'berlin52': 7542,
    'st70': 675,
    'pr76': 108159,
    'eil76': 538,
    'rat99': 1211,
    'kroA100': 21282,
    'kroE100': 22068,
    'kroB100': 22141,
    'rd100': 7910,
    'kroD100': 21294,
    'kroC100': 20749,
    'eil101': 629,
    'lin105': 14379,
    'pr107': 44303,
    'pr124': 59030,
    'bier127': 118282,
    'ch130': 6110,
    'pr136': 96772,
    'pr144': 58537,
    'kroA150': 26524,
    'kroB150': 26130,
    'ch150': 6528,
    'pr152': 73682,
    'u159': 42080,
    'rat195': 2323,
    'd198': 15780,
    'kroA200': 29368,
    'kroB200': 29437,
    'tsp225': 3916,
    'ts225': 126643,
    'pr226': 80369,
    'gil262': 2378,
    'pr264': 49135,
    'a280': 2579,
    'pr299': 48191,
    'lin318': 42029,
    'rd400': 15281,
    'fl417': 11861,
    'pr439': 107217,
    'pcb442': 50778,
    'd493': 35002,
    'u574': 36905,
    'rat575': 6773,
    'p654': 34643,
    'd657': 48912,
    'u724': 41910,
    'rat783': 8806,
    'pr1002': 259045,
    'u1060': 224094,
    'vm1084': 239297,
    'pcb1173': 56892,
    'd1291': 50801,
    'rl1304': 252948,
    'rl1323': 270199,
    'nrw1379': 56638,
    'fl1400': 20127,
    'u1432': 152970,
    'fl1577': 22249,
    'd1655': 62128,
    'vm1748': 336556,
    'u1817': 57201,
    'rl1889': 316536,
    'd2103': 80450,
    'u2152': 64253,
    'u2319': 234256,
    'pr2392': 378032,
    'pcb3038': 137694,
    'fl3795': 28772,
    'fnl4461': 182566,
    'rl5915': 565530,
    'rl5934': 556045,
    'rl11849': 923288,
    'usa13509': 19982859,
    'brd14051': 469385,
    'd15112': 1573084,
    'd18512': 645238
}


# national_collections = {
#   'WI29': 27603,
#   'DJ38': 6656,
#   'QA194': 9352,
#   'UY734': 79114,
#   'ZI929': 95345,
#   'LU980': 11340,
#   'RW1621': 26051,
#   'MU1979': 86891,
#   'NU3496': 96132,
#   'CA4663': 1290319,
#   'TZ6117': 394609,
#   'EG7146': 172386,
#   'YM7663': 238314,
#   'PM8079': 114831,
#   'EI8246': 206128,
#   'AR9152': 837377,
#   'JA9847': 491924,
#   'GR9882': 300899,
#   'KZ9976': 1061387,
#   'FI10639': 520383,
#   'MO14185': 427246,
#   'HO14473': 176940,
#   'IT16862': 557315,
#   'VM22775': 569115,
#   'SW24978': 855597,
#   'BM33708': 959011,
# # 'CH71009': 4565452,
# }


cvrplib_collections = {
    "X-n101-k25": 27591,
    "X-n106-k14": 26362,
    "X-n110-k13": 14971,
    "X-n115-k10": 12747,
    "X-n120-k6": 13332,
    "X-n125-k30": 55539,
    "X-n129-k18": 28940,
    "X-n134-k13": 10916,
    "X-n139-k10": 13590,
    "X-n143-k7": 15700,
    "X-n148-k46": 43448,
    "X-n153-k22": 21220,
    "X-n157-k13": 16876,
    "X-n162-k11": 14138,
    "X-n167-k10": 20557,
    "X-n172-k51": 45607,
    "X-n176-k26": 47812,
    "X-n181-k23": 25569,
    "X-n186-k15": 24145,
    "X-n190-k8": 16980,
    "X-n195-k51": 44225,
    "X-n200-k36": 58578,
    "X-n204-k19": 19565,
    "X-n209-k16": 30656,
    "X-n214-k11": 10856,
    "X-n219-k73": 117595,
    "X-n223-k34": 40437,
    "X-n228-k23": 25742,
    "X-n233-k16": 19230,
    "X-n237-k14": 27042,
    "X-n242-k48": 82751,
    "X-n247-k50": 37274,
    "X-n251-k28": 38684,
    "X-n256-k16": 18839,
    "X-n261-k13": 26558,
    "X-n266-k58": 75478,
    "X-n270-k35": 35291,
    "X-n275-k28": 21245,
    "X-n280-k17": 33503,
    "X-n284-k15": 20226,
    "X-n289-k60": 95151,
    "X-n294-k50": 47161,
    "X-n298-k31": 34231,
    "X-n303-k21": 21736,
    "X-n308-k13": 25859,
    "X-n313-k71": 94043,
    "X-n317-k53": 78355,
    "X-n322-k28": 29834,
    "X-n327-k20": 27532,
    "X-n331-k15": 31102,
    "X-n336-k84": 139111,
    "X-n344-k43": 42050,
    "X-n351-k40": 25896,
    "X-n359-k29": 51505,
    "X-n367-k17": 22814,
    "X-n376-k94": 147713,
    "X-n384-k52": 65940,
    "X-n393-k38": 38260,
    "X-n401-k29": 66154,
    "X-n411-k19": 19712,
    "X-n420-k130": 107798,
    "X-n429-k61": 65449,
    "X-n439-k37": 36391,
    "X-n449-k29": 55233,
    "X-n459-k26": 24139,
    "X-n469-k138": 221824,
    "X-n480-k70": 89449,
    "X-n491-k59": 66483,
    "X-n502-k39": 69226,
    "X-n513-k21": 24201,
    "X-n524-k153": 154593,
    "X-n536-k96": 94846,
    "X-n548-k50": 86700,
    "X-n561-k42": 42717,
    "X-n573-k30": 50673,
    "X-n586-k159": 190316,
    "X-n599-k92": 108451,
    "X-n613-k62": 59535,
    "X-n627-k43": 62164,
    "X-n641-k35": 63684,
    "X-n655-k131": 106780,
    "X-n670-k130": 146332,
    "X-n685-k75": 68205,
    "X-n701-k44": 81923,
    "X-n716-k35": 43373,
    "X-n733-k159": 136187,
    "X-n749-k98": 77269,
    "X-n766-k71": 114417,
    "X-n783-k48": 72386,
    "X-n801-k40": 73311,
    "X-n819-k171": 158121,
    "X-n837-k142": 193737,
    "X-n856-k95": 88965,
    "X-n876-k59": 99299,
    "X-n895-k37": 53860,
    "X-n916-k207": 329179,
    "X-n936-k151": 132715,
    "X-n957-k87": 85465,
    "X-n979-k58": 118976,
    "X-n1001-k43": 72355,
}

def avg_list(list_object):
    return sum(list_object) / len(list_object) if len(list_object) > 0 else 0


def parse_tsplib_name(tsplib_name):
    return "".join(filter(str.isalpha, tsplib_name)), int("".join(filter(str.isdigit, tsplib_name)))

def parse_cvrplib_name(cvrplib_name):
    problem_set, size, _ = cvrplib_name.split("-")
    size = int("".join(list(filter(str.isdigit, size)))) - 1
    return problem_set, size

def read_tsplib_file(file_path):
    """
    The read_tsplib_file function reads a TSPLIB file and returns the nodes and name of the problem.
    
    :param file_path: Specify the path to the file that is being read
    :return: A list of nodes and a name
    """
    properties = {}
    reading_properties_flag = True
    nodes = []


    import os

    current_working_directory = os.getcwd()
    print("当前工作路径是：", current_working_directory)

    with open(file_path, "r", encoding="utf8") as read_file:
        line = read_file.readline()
        while line.strip():
            # read properties
            if reading_properties_flag:
                if ':' in line:
                    key, val = [x.strip() for x in line.split(':')]
                    properties[key] = val
                else:
                    reading_properties_flag = False

            # read node coordinates
            else:
                if line.startswith("NODE_COORD_SECTION"):
                    pass
                elif line.startswith("EOF"):
                    pass
                else:
                    line_contents = [x.strip() for x in line.split(" ") if x.strip()]
                    _, x, y = line_contents
                    nodes.append([float(x), float(y)])
            line = read_file.readline()

    return nodes, properties["NAME"]


def read_cvrplib_file(file_path):
    """
    The read_cvrplib_file function reads a CVRP file and returns the depot, nodes, demands and properties.
    
    :param file_path: Specify the path of the file to be read
    :return: A tuple of four elements:
    """
    properties = {}
    reading_properties_flag = True
    reading_nodes_flag = True
    read_demands_flag = True
    read_depot_flag = True

    depot_nodes = []
    demands = []
    depot_check = []

    with open(file_path, "r", encoding="utf8") as read_file:
        line = read_file.readline()
        while line.strip():
            # read properties
            if reading_properties_flag:
                if ':' in line:
                    key, val = [x.strip() for x in line.split(':')]
                    properties[key] = val
                else:
                    reading_properties_flag = False

            # read node coordinates
            elif reading_nodes_flag:
                if line.startswith("NODE_COORD_SECTION"):
                    pass
                elif line.startswith("DEMAND_SECTION"):
                    reading_nodes_flag = False
                else:
                    line_contents = [x.strip() for x in line.replace(" ", "\t").split("\t") if x.strip()]
                    _, x, y = line_contents
                    depot_nodes.append([float(x), float(y)])

            # read demands coordinates
            elif read_demands_flag:
                if line.startswith("DEMAND_SECTION"):
                    pass
                elif line.startswith("DEPOT_SECTION"):
                    read_demands_flag = False
                else:
                    line_contents = [x.strip() for x in line.replace(" ", "\t").split("\t") if x.strip()]
                    demands.append(int(line_contents[1]))

            # read depot position
            elif read_depot_flag:
                if line.startswith("DEPOT_SECTION"):
                    pass
                elif line.startswith("EOF"):
                    read_depot_flag = False
                else:
                    line_contents = [x.strip() for x in line.replace(" ", "\t").split("\t") if x.strip()]
                    depot_check.append(int(line_contents[0]))
            line = read_file.readline()

    depot = depot_nodes[0]
    nodes = depot_nodes[1:]
    demands = demands[1:]

    return depot, nodes, demands, properties



def load_tsplib_file(root, tsplib_name):
    tsplib_dir = "tsplib"
    file_name = f"{tsplib_name}.tsp"
    file_path = root.joinpath(tsplib_dir).joinpath(file_name)
    instance, name = read_tsplib_file(file_path)

    instance = torch.tensor(instance)
    return instance, name

def load_cvrplib_file(root, cvrplib_name):
    cvrplib_dir = "vrplib"
    file_name = f"{cvrplib_name}.vrp"
    file_path = root.joinpath(cvrplib_dir).joinpath(file_name)
    depot, nodes, demands, properties = read_cvrplib_file(file_path)

    depot = torch.tensor(depot)
    nodes = torch.tensor(nodes)
    demands = torch.tensor(demands)
    capacity = torch.tensor(int(properties["CAPACITY"]))
    name = properties["NAME"]
    return depot, nodes, demands, capacity, name



def normalize_tsp_to_unit_board(tsp_instance):
    """
    normalize a tsp instance to a [0, 1]^2 unit board, prefer to have points on both x=0 and y=0
    :param tsp_instance: a (tsp_size, 2) tensor
    :return: a (tsp_size, 2) tensor, a normalized tsp instance
    """
    normalized_instance = tsp_instance.clone()
    normalization_factor = (normalized_instance.max(dim=0).values - normalized_instance.min(dim=0).values).max()
    normalized_instance = (normalized_instance - normalized_instance.min(dim=0).values) / normalization_factor
    return normalized_instance


def normalize_nodes_to_unit_board(nodes):
    return normalize_tsp_to_unit_board(nodes)


def get_dist_matrix(instance):
    size = instance.shape[0]
    x = instance.unsqueeze(0).repeat((size, 1, 1))
    y = instance.unsqueeze(1).repeat((1, size, 1))
    return torch.norm(x - y, p=2, dim=-1)


def calculate_tour_length_by_dist_matrix(dist_matrix, tours):
    # useful to evaluate one/multiple solutions on one (not-extremely-huge) instance
    if tours.dim() == 1:
        tours = tours.unsqueeze(0)
    tour_shifts = torch.roll(tours, shifts=-1, dims=1)
    tour_lens = dist_matrix[tours, tour_shifts].sum(dim=1)
    return tour_lens

def get_result_folder():
    return result_folder


def set_result_folder(folder):
    global result_folder
    result_folder = folder


def create_logger(log_file=None):
    if 'filepath' not in log_file:
        log_file['filepath'] = get_result_folder()

    if 'desc' in log_file:
        log_file['filepath'] = log_file['filepath'].format(desc='_' + log_file['desc'])
    else:
        log_file['filepath'] = log_file['filepath'].format(desc='')

    set_result_folder(log_file['filepath'])

    if 'filename' in log_file:
        filename = log_file['filepath'] + '/' + log_file['filename']
    else:
        filename = log_file['filepath'] + '/' + 'log.txt'

    if not os.path.exists(log_file['filepath']):
        os.makedirs(log_file['filepath'])

    file_mode = 'a' if os.path.isfile(filename)  else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += (val * n)
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0


class LogData:
    def __init__(self):
        self.keys = set()
        self.data = {}

    def get_raw_data(self):
        return self.keys, self.data

    def set_raw_data(self, r_data):
        self.keys, self.data = r_data

    def append_all(self, key, *args):
        if len(args) == 1:
            value = [list(range(len(args[0]))), args[0]]
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].extend(value)
        else:
            self.data[key] = np.stack(value, axis=1).tolist()
            self.keys.add(key)

    def append(self, key, *args):
        if len(args) == 1:
            args = args[0]

            if isinstance(args, int) or isinstance(args, float):
                if self.has_key(key):
                    value = [len(self.data[key]), args]
                else:
                    value = [0, args]
            elif type(args) == tuple:
                value = list(args)
            elif type(args) == list:
                value = args
            else:
                raise ValueError('Unsupported value type')
        elif len(args) == 2:
            value = [args[0], args[1]]
        else:
            raise ValueError('Unsupported value type')

        if key in self.keys:
            self.data[key].append(value)
        else:
            self.data[key] = [value]
            self.keys.add(key)

    def get_last(self, key):
        if not self.has_key(key):
            return None
        return self.data[key][-1]

    def has_key(self, key):
        return key in self.keys

    def get(self, key):
        split = np.hsplit(np.array(self.data[key]), 2)

        return split[1].squeeze().tolist()

    def getXY(self, key, start_idx=0):
        # print("===========key==========")
        # print(key)
        # print("======self.data[key]======")
        # print(self.data[key])
        split = np.hsplit(np.array(self.data[key]), 2)

        xs = split[0].squeeze().tolist()
        ys = split[1].squeeze().tolist()

        if type(xs) is not list:
            return xs, ys

        if start_idx == 0:
            return xs, ys
        elif start_idx in xs:
            idx = xs.index(start_idx)
            return xs[idx:], ys[idx:]
        else:
            raise KeyError('no start_idx value in X axis data.')

    def get_keys(self):
        return self.keys


class TimeEstimator:
    def __init__(self):
        self.logger = logging.getLogger('TimeEstimator')
        self.start_time = time.time()
        self.count_zero = 0

    def reset(self, count=1):
        self.start_time = time.time()
        self.count_zero = count-1

    def get_est(self, count, total):
        curr_time = time.time()
        elapsed_time = curr_time - self.start_time
        remain = total-count
        remain_time = elapsed_time * remain / (count - self.count_zero)

        elapsed_time /= 3600.0
        remain_time /= 3600.0

        return elapsed_time, remain_time

    def get_est_string(self, count, total):
        elapsed_time, remain_time = self.get_est(count, total)

        elapsed_time_str = "{:.2f}h".format(elapsed_time) if elapsed_time > 1.0 else "{:.2f}m".format(elapsed_time*60)
        remain_time_str = "{:.2f}h".format(remain_time) if remain_time > 1.0 else "{:.2f}m".format(remain_time*60)

        return elapsed_time_str, remain_time_str

    def print_est_time(self, count, total):
        elapsed_time_str, remain_time_str = self.get_est_string(count, total)

        self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
            count, total, elapsed_time_str, remain_time_str))


def util_print_log_array(logger, result_log: LogData):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    for key in result_log.get_keys():
        logger.info('{} = {}'.format(key+'_list', result_log.get(key)))


def util_save_log_image_with_label(result_file_prefix,
                                   img_params,
                                   result_log: LogData,
                                   labels=None):
    dirname = os.path.dirname(result_file_prefix)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    _build_log_image_plt(img_params, result_log, labels)

    if labels is None:
        labels = result_log.get_keys()
    file_name = '_'.join(labels)
    fig = plt.gcf()

    # print("========result_file_prefix======")
    # print(result_file_prefix)
    # print("========file_name======")
    # print(file_name)


    fig.savefig('{}-{}.jpg'.format(result_file_prefix, file_name))
    plt.close(fig)
# /public/home/luof/project/BQ-POMO/CVRP/utils/log_image_style/style_tsp_100.json

def _build_log_image_plt(img_params,
                         result_log: LogData,
                         labels=None):
    assert type(result_log) == LogData, 'use LogData Class for result_log.'

    # Read json
    folder_name = img_params['json_foldername']
    file_name = img_params['filename']
    log_image_config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_name, file_name)

    # print("==========log_image_config_file=========")
    # print(log_image_config_file)

    with open(log_image_config_file, 'r') as f:
        config = json.load(f)

    figsize = (config['figsize']['x'], config['figsize']['y'])
    plt.figure(figsize=figsize)

    if labels is None:
        labels = result_log.get_keys()
    for label in labels:
        # print("=====label====")
        # print(label)
        plt.plot(*result_log.getXY(label), label=label)

    plt.draw()

    ylim_min = config['ylim']['min']
    ylim_max = config['ylim']['max']

    # print("=====before_ylim_min=====")
    # print(ylim_min)   
    # print("=====before_ylim_max=====")
    # print(ylim_max)

    if ylim_min is None:
        ylim_min = plt.gca().dataLim.ymin
    if ylim_max is None:
        ylim_max = plt.gca().dataLim.ymax

    # print("=====ylim_min=====")
    # print(ylim_min)   
    # print("=====ylim_max=====")
    # print(ylim_max)

    plt.ylim(ylim_min, ylim_max)
    
    plt.ylim(ylim_min, ylim_max)

    xlim_min = config['xlim']['min']
    xlim_max = config['xlim']['max']
    if xlim_min is None:
        xlim_min = plt.gca().dataLim.xmin
    if xlim_max is None:
        xlim_max = plt.gca().dataLim.xmax
    plt.xlim(xlim_min, xlim_max)

    plt.rc('legend', **{'fontsize': 18})
    plt.legend()
    plt.grid(config["grid"])


def copy_all_src(dst_root, src_path):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(src_path)

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if os.path.commonprefix([execution_path, src_abspath]) == execution_path:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                if os.path.exists(src_abspath):
                    shutil.copy(src_abspath, dst_filepath)
                else:
                    print(f"Warning: Source file does not exist, skipping: {src_abspath}")



def tour_nodes_to_tour_len(nodes, W_values):
    """Helper function to calculate tour length from ordered list of tour nodes.
    """
    tour_len = 0
    for idx in range(len(nodes) - 1):
        i = nodes[idx]
        j = nodes[idx + 1]
        tour_len += W_values[i][j]
    # Add final connection of tour in edge target
    tour_len += W_values[j][nodes[0]]
    return tour_len

def is_valid_tour(nodes, num_nodes):
    """Sanity check: tour visits all nodes given.
    """
    return sorted(nodes) == [i for i in range(num_nodes)]


def Scale(X):
    """
    The Scale function takes in a batch of points and scales them to be between 0 and 1.
    It does this by translating the points so that the minimum x-value is at 0, 
    and then dividing all x-values by the maximum value. It does this for both dimensions.
    
    :param X: Store the data and the scale_method parameter is used to determine how to scale it
    :param scale_method: Decide whether to scale the data based on the boundary of all points or just
    :return: The scaled x and the ratio
    """
    B = X.size(0)
    SIZE = X.size(1)
    X = X - torch.reshape(torch.min(X,1).values,(B,1,2)).repeat(1,SIZE,1) # translate
    ratio_x = torch.reshape(torch.max(X[:,:,0], 1).values - torch.min(X[:,:,0], 1).values,(-1,1))
    ratio_y = torch.reshape(torch.max(X[:,:,1], 1).values - torch.min(X[:,:,1], 1).values,(-1,1))
    ratio = torch.max(torch.cat((ratio_x,ratio_y),1),1).values
    ratio[ratio==0] = 1
    X = X / (torch.reshape(ratio,(B,1,1)).repeat(1,SIZE,2))
    return X, ratio

def Scale_for_vrp(X,num):
    """
    The Scale function takes in a batch of points and scales them to be between 0 and 1.
    It does this by translating the points so that the minimum x-value is at 0, 
    and then dividing all x-values by the maximum value. It does this for both dimensions.
    
    :param X: Store the data and the scale_method parameter is used to determine how to scale it
    :param scale_method: Decide whether to scale the data based on the boundary of all points or just
    :return: The scaled x and the ratio
    """
    B = X.size(0)
    SIZE = X.size(1)
    graph = X[:,:num,:]
    min_values = torch.reshape(torch.min(graph,1).values,(B,1,2)).repeat(1,SIZE,1)
    X = X - min_values # translate
    ratio_x = torch.reshape(torch.max(graph[:,:,0], 1).values - torch.min(graph[:,:,0], 1).values,(-1,1))
    ratio_y = torch.reshape(torch.max(graph[:,:,1], 1).values - torch.min(graph[:,:,1], 1).values,(-1,1))
    ratio = torch.max(torch.cat((ratio_x,ratio_y),1),1).values
    ratio[ratio==0] = 1
    X = X / (torch.reshape(ratio,(B,1,1)).repeat(1,SIZE,2))
    X[ratio==0,:,:] = X[ratio==0,:,:]+min_values[ratio==0,:,:]
    return X, ratio

def Rotate_aug(X):
    """
    The Rotate_aug function takes in a batch of points and rotates them by a random angle.
    The function also scales the points to be between 0 and 1.
    
    :param X: Pass the input data to the function
    :return: The rotated point cloud and the ratio of the bounding box
    """
    device = X.device
    B = X.size(0)
    SIZE = X.size(1)
    Theta = torch.rand((B,1),device=device)* 2 * np.pi
    Theta = Theta.repeat(1,SIZE)
    tmp1 = torch.reshape(X[:,:,0]*torch.cos(Theta) - X[:,:,1]*torch.sin(Theta),(B,SIZE,1))
    tmp2 = torch.reshape(X[:,:,0]*torch.sin(Theta) + X[:,:,1]*torch.cos(Theta),(B,SIZE,1))
    X_out = torch.cat((tmp1, tmp2), dim=2)
    X_out += 10
    X_out, ratio = Scale(X_out)
    return X_out, ratio

def Reflect_aug(X):
    """
    The Reflect_aug function takes in a batch of points and performs the following operations:
        1. Rotate each point by a random angle between 0 and 2pi radians
        2. Reflect each point across the x-axis (i.e., multiply y coordinate by -2)
        3. Add 10 to all coordinates so that no points are negative anymore (this is for convenience)
        4. Scale all coordinates down to be between 0 and 1
    
    :param X: Pass the data points to the function
    :return: A reflected point cloud and a scale ratio
    """
    device = X.device
    B = X.size(0)
    SIZE = X.size(1)
    Theta = torch.rand((B,1),device=device)* 2 * np.pi
    Theta = Theta.repeat(1,SIZE)
    tmp1 = torch.reshape(X[:,:,0]*torch.cos(2*Theta) + X[:,:,1]*torch.sin(2*Theta),(B,SIZE,1))
    tmp2 = torch.reshape(X[:,:,0]*torch.sin(2*Theta) - X[:,:,1]*torch.cos(2*Theta),(B,SIZE,1))
    X_out = torch.cat((tmp1, tmp2), dim=2)
    X_out += 10
    X_out, ratio = Scale(X_out)
    return X_out, ratio

def mix_aug(X):
    """
    The mix_aug function takes in a batch of images and returns the same batch with half of them rotated and half reflected.
    The function also returns the ratio between the number of pixels that are black after augmentation to before augmentation.
    
    :param X: Pass in the data
    :return: The augmented images and the ratio of the number of augmented images to original ones
    """
    X_out = X.clone()
    X_out[0::2],ratio = Rotate_aug(X[0::2])
    X_out[1::2],ratio = Reflect_aug(X[1::2])
    return X_out,ratio

def run_aug(aug,x,aug_num=None,aug_all=False):
    """
    The run_aug function takes in an augmentation type, a batch of images, and two optional arguments.
    The first optional argument is the number of images to augment per batch. The second is whether or not to 
    augment all the images in the batch (defaults to False). It then returns a copy of x with some augmented 
    images inserted into it.
    
    :param aug: Select the augmentation to apply
    :param x: Pass in the data
    :param aug_num: Control the number of augmented images in each batch
    :param aug_all: Decide whether to apply the augmentation on all images or only a subset of them
    :return: A tensor with the same size as x, but with some of its values replaced by augmented data
    """
    x_clone = x.clone()
    if aug == 'rotate':
        x_out,_ = Rotate_aug(x)
    elif aug == 'reflect':
        x_out,_ = Reflect_aug(x)
    elif aug == 'mix':
        x_out,_ = mix_aug(x)
    elif aug == 'noise':
        x_out = x+torch.rand(x.size(), device=x.device)*1e-5
    else:
        x_out = x
    if not aug_all:
        if aug_num is not None:
            x_out[0::aug_num]=x_clone[0::aug_num]
        else:
            x_out[0]=x_clone[0]
    return x_out


def choose_bsz(size):
    if size<=200:
        return 64
    elif size<=1000:
        return 32
    elif size<=5000:
        return 16
    else:
        return 4


