# -*- coding: utf-8 -*-
from paths import ROOT_PATH  # isort:skip
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in range(10)])
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import argparse
import os.path as osp

from loguru import logger
import numpy as np
import random
import torch
torch.set_num_threads(4)

from videoanalyst.config.config import cfg as root_cfg
from videoanalyst.config.config import specify_task
from videoanalyst.engine.builder import build as tester_builder
from videoanalyst.model import builder as model_builder
from videoanalyst.pipeline import builder as pipeline_builder
from videoanalyst.utils import complete_path_wt_root_in_cfg
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_parser():
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('-cfg',
                        '--config',
                        default='experiments/tatrack/test/tiny/got.yaml',
                        type=str,
                        help='experiment configuration')

    return parser


def build_tatrack_tester(task_cfg):
    # build model
    model = model_builder.build("track", task_cfg.model)
    # build pipeline
    pipeline = pipeline_builder.build("track", task_cfg.pipeline, model)
    # build tester
    testers = tester_builder("track", task_cfg.tester, "tester", pipeline)
    return testers


if __name__ == '__main__':
    set_seed(1000000007)
    # parsing
    parser = make_parser()
    parsed_args = parser.parse_args()

    # experiment config
    exp_cfg_path = osp.realpath(parsed_args.config)
    root_cfg.merge_from_file(exp_cfg_path)
    logger.info("Load experiment configuration at: %s" % exp_cfg_path)

    # resolve config
    root_cfg = complete_path_wt_root_in_cfg(root_cfg, ROOT_PATH)
    root_cfg = root_cfg.test
    task, task_cfg = specify_task(root_cfg)
    task_cfg.freeze()

    torch.multiprocessing.set_start_method('spawn', force=True)

    if task == 'track':
        testers = build_tatrack_tester(task_cfg)
    else:
        raise ValueError('Undefined task: {}'.format(task))

    # output = []
    # for i in range(45,60):
    #     for j in range(len(testers)):
        #     testers[j]._hyper_params["exp_save"] = '/hkj/code/mywork4-1-4/logs/9-{}-test'.format(i)
        #     testers[j]._pipeline._model._hyper_params['pretrain_model_path']= \
        #     '/hkj/code/mywork4-1-4/snapshots/tiny-9-train/epoch-{}.pkl'.format(i)
        #     testers[j]._pipeline._model.update_params()    
        # print("this epoch is:",i)
        # result = -1
    for tester in testers:
        result = tester.test()
        print(result)
        # output.append(result)

    # for i in range(len(output)):
    #     print(i+0,"epoch:",output[i])