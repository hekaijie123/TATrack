# -*- coding: utf-8 -*

# from copy import deepcopy
# from re import template

import numpy as np
# import math
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms
import torch
from data.operator.bbox.spatial.vectorized.torch.scale_and_translate import bbox_scale_and_translate_vectorized
from data.tracking.methods.SiamFC.common.siamfc_curation import prepare_SiamFC_curation, do_SiamFC_curation
from videoanalyst.pipeline.pipeline_base import TRACK_PIPELINES, PipelineBase
from videoanalyst.pipeline.utils import (xywh2xyxy,xyxy2xywh)
from data.operator.bbox.spatial.vectorized.torch.cxcywh_to_xyxy import box_cxcywh_to_xyxy
import videoanalyst.utils.visualize_score_map as vsm
from data.operator.image.pytorch.mean import get_image_mean
from data.tracking.methods.sequential.curation_parameter_provider import _adjust_bbox_size
from data.operator.bbox.spatial.vectorized.torch.utility.half_pixel_offset.image import bbox_restrict_in_image_boundary_
from videoanalyst.data.target.target_impl.utils.make_densebox_target import gaussian_label_function,generate_ltbr_regression_targets
# ============================== Tracker definition ============================== #
@TRACK_PIPELINES.register
class TATrackTracker(PipelineBase):
    r"""
    default_hyper_params setting rules:
    0/0.0: to be set in config file manually.
    -1: to be calculated in code automatically.
    >0: default value.
    """

    default_hyper_params = dict(
    
        window_influence=0.0,
        m_size=0,
        q_size=0,
        min_w=10,
        min_h=10,
        phase_memorize="memorize",
        phase_track="track",
        num_segments=4,
        confidence_threshold=0.6,
        gpu_memory_threshold=-1,
        template_area_factor=4.0,
        search_area_factor=4.0,
        visualization=False,
        interpolate_mode='bilinear',
        method = 'norm',
        score_size = 0,
    )

    def __init__(self, *args, **kwargs):
        super(TATrackTracker, self).__init__(*args, **kwargs)
        self.update_params()

        # set underlying model to device
        self.device = torch.device("cpu")
        self.debug = False
        self.set_model(self._model)
        self.transform = self.get_transform()

    def set_model(self, model):
        """model to be set to pipeline. change device & turn it into eval mode
        Parameters
        ----------
        model : ModuleBase
            model to be set to pipeline
        """
        self._model = model
        # self._model = model.to(self.device)
        self._model.eval()

    def set_device(self, device):
        self.device = device
        self._model = self._model.to(device)
        if self.device != torch.device('cuda:0'):
            self._hyper_params['gpu_memory_threshold'] = 3000

    def update_params(self):
        hps = self._hyper_params
        # assert hps['q_size'] == hps['m_size']

        if hps['gpu_memory_threshold'] == -1:
            hps['gpu_memory_threshold'] = 1 << 30  # infinity
        self._hyper_params = hps

        # self._hp_score_size = self._hyper_params['score_size']
        self._hp_m_size = self._hyper_params['m_size']
        self._hp_q_size = self._hyper_params['q_size']
        self._hp_num_segments = self._hyper_params['num_segments']
        self._hp_gpu_memory_threshold = self._hyper_params['gpu_memory_threshold']
        self._hp_confidence_threshold = self._hyper_params['confidence_threshold']
        self._hp_visualization = self._hyper_params['visualization']
        self._hp_method = self._hyper_params['method']
        self._hp_score_size = self._hyper_params['score_size']


    

    def select_representatives(self, cur_frame_idx):       
        if self._hp_method=='mean':
            scores = self._state['pscores']
        elif self._hp_method=='p_mean':
            scores = self._state['penalty_scores']
        indexes = None
        for i in range(cur_frame_idx-1,-1,-1):
            if(indexes == None):
                if(self._state['pscores'][i]>=scores.mean() ):#
                    indexes = i
            else:break



        return self._state['all_memory_frame_feats'][indexes],self._state['all_gauss_label'][indexes],self._state['all_ltbr_label'][indexes]

    def init(self, im, bbox):
        r"""Initialize tracker
            Internal target state representation: self._state['state'] = (target_pos, target_sz)
        
        Arguments
        ---------
        im : np.array
            initial frame image
        state
            target state on initial frame (bbox in case of SOT), format: xywh
        """
        self._state['im_h'] = im.shape[1]
        self._state['im_w'] = im.shape[2]       #
        self._state['last_img'] = im
        self._state['all_memory_frame_feats'] = []
        self._state['all_gauss_label'] = []
        self._state['all_ltbr_label'] = []
        self._state['pscores'] = [1.0 ]
        self._state['pscores'] = torch.tensor(self._state['pscores'])
        self._state['penalty_scores'] = [1.0]
        self._state['penalty_scores'] = torch.tensor(self._state['penalty_scores'])
        self._state['cur_frame_idx'] = 1
        self._state['last_bbox'] = bbox     #
        self._state["rng"] = np.random.RandomState(123456)
        self._state['avg_chans'] = get_image_mean(im)
        self.window = torch.flatten(torch.outer(torch.hann_window(self._hp_score_size, periodic=False),
                                                    torch.hann_window(self._hp_score_size, periodic=False)))
        self.window = self.window.to(self.device)
        last_bbox = xywh2xyxy(self._state['last_bbox']) 
        curation_parameter, _ = prepare_SiamFC_curation(last_bbox, self._hyper_params['template_area_factor'], [self._hp_m_size,self._hp_m_size])
        curated_first_frame_image, _ = do_SiamFC_curation(im, [self._hp_m_size,self._hp_m_size], curation_parameter, self._hyper_params['interpolate_mode'])
        curated_first_frame_image = self.transform(curated_first_frame_image)   

        self._state['template_feature'] = curated_first_frame_image.unsqueeze(0).to(self.device)
        if self._hp_visualization:
            vsm.rename_dir()



    def get_avg_chans(self):
        return self._state['avg_chans']

    def track(self,
            im_q,
            last_bbox,
            features,gauss_label,ltbr_label,
            update_state=False,
            **kwargs):
        if 'avg_chans' in kwargs:
            avg_chans = kwargs['avg_chans']
        else:
            avg_chans = self._state['avg_chans']

        q_size = self._hp_q_size
        phase_track = self._hyper_params['phase_track']
        last_bbox = _adjust_bbox_size(torch.tensor(last_bbox),[self._hyper_params['min_w'],self._hyper_params['min_h']])
        curation_parameter,_ = prepare_SiamFC_curation(last_bbox, self._hyper_params['search_area_factor'], [q_size,q_size])
        im_q_crop,_ = do_SiamFC_curation(im_q,[q_size,q_size],curation_parameter,self._hyper_params['interpolate_mode'],avg_chans)
        self.last_curation_parameter = curation_parameter
        im_q_crop = self.transform(im_q_crop)
        with torch.no_grad():
            output = self._model(
                im_q_crop.unsqueeze(0).to(self.device),
                self._state['template_feature'],
                features,gauss_label,ltbr_label,
                phase=phase_track)

        curation_bbox,confidence_score = self.post_tracking(output)
        if update_state:
            self._state['pscores'] = torch.cat((self._state['pscores'],confidence_score),dim=-1)
            self._state['penalty_scores'] = torch.cat((self._state['penalty_scores'],self._state['pscores'].mean().unsqueeze(0)),dim=-1)
        return curation_bbox



    def update(self, im, state=None):
        """ Perform tracking on current frame
            Accept provided target state prior on current frame
            e.g. search the target in another video sequence simutanously

        Arguments
        ---------
        im : np.array
            current frame image
        state
            provided target state prior (bbox in case of SOT), format: xywh
        """
        # use prediction on the last frame as target state prior
        if state is None:
            last_bbox = xywh2xyxy(self._state['last_bbox'])     ##

        fidx = self._state['cur_frame_idx']

        prev_frame_feat,prev_gauss_label,prev_ltbr_label = self.memorize(self._state['last_img'],
                                last_bbox,self._state['avg_chans'])     


        if fidx > self._hp_gpu_memory_threshold:
            prev_frame_feat = prev_frame_feat.detach().cpu()
            prev_gauss_label = prev_gauss_label.detach().cpu()
            prev_ltbr_label = prev_ltbr_label.detach().cpu()
        self._state['all_memory_frame_feats'].append(prev_frame_feat)
        self._state['all_gauss_label'].append(prev_gauss_label)
        self._state['all_ltbr_label'].append(prev_ltbr_label)


        features,gauss_label,ltbr_label = self.select_representatives(fidx)

            
        # forward inference to estimate new state
        output = self.track(im,
                            last_bbox,
                            features,gauss_label,ltbr_label,
                            update_state=True).squeeze(0)

        self._state['last_img'] = im
        self._state['last_bbox'] = output

        self._state['cur_frame_idx'] += 1

        return output

    def memorize(self, im: np.array, last_bbox, avg_chans):

        q_size = self._hp_q_size
        phase = self._hyper_params['phase_memorize']
        curation_parameter, output_bbox = prepare_SiamFC_curation(last_bbox, 4.0 , [q_size,q_size])##
        curated_first_frame_image, _ = do_SiamFC_curation(im, [q_size,q_size], curation_parameter, self._hyper_params['interpolate_mode'])##
        curated_first_frame_image = self.transform(curated_first_frame_image)   ##
        
        gauss_label = gaussian_label_function(output_bbox.view(1,-1),0.1,1,self._hp_score_size, q_size, end_pad_if_even=True).view(1,-1,1)
        ltbr_label = generate_ltbr_regression_targets(output_bbox.view(1,-1),16,q_size).unsqueeze(0)

        return curated_first_frame_image.unsqueeze(0).to(self.device),gauss_label.to(self.device),ltbr_label.to(self.device)



    def get_transform(self):
        return transforms.Normalize(mean=torch.tensor(IMAGENET_DEFAULT_MEAN), std=torch.tensor(IMAGENET_DEFAULT_STD))

    def post_tracking(self,outputs):
        if outputs is None:
            return None
        class_score_map, predicted_bbox = outputs['class_score'], outputs['bbox']  # shape: (N, 1, H, W), (N, H, W, 4)
        
        if self._hp_method=='p_mean':
            class_score_map = -torch.log(1/(class_score_map+1e-10) -1)

        N, C, H, W = class_score_map.shape
        assert C == 1
        class_score_map = class_score_map.view(N, H * W)

        # window penalty
        class_score_map = class_score_map * (1 - self._hyper_params['window_influence']) + \
                    self.window.view(1, H * W) * self._hyper_params['window_influence']


        confidence_score, best_idx = torch.max(class_score_map, 1)

        predicted_bbox = predicted_bbox.view(N, H * W, 4)
        bounding_box = predicted_bbox[torch.arange(len(predicted_bbox)), best_idx, :]

        confidence_score = confidence_score.cpu()
        bounding_box = bounding_box.cpu()
        bounding_box = bounding_box.to(torch.float64)

        assert self.last_curation_parameter.ndim in (2, 3)
        curation_scaling, curation_source_center_point, curation_target_center_point = self.last_curation_parameter.unbind(dim=-2)
        bbox = bounding_box*self._hp_q_size
        if self._hp_method=='mean':
            bbox[:,2] = self._state['last_bbox'][2] * (1 - confidence_score)*curation_scaling[0] + bbox[:,2] * confidence_score
            bbox[:,3] = self._state['last_bbox'][3] * (1 - confidence_score)*curation_scaling[1] + bbox[:,3] * confidence_score
        bbox = box_cxcywh_to_xyxy(bbox)
        bbox = bbox_scale_and_translate_vectorized(bbox, 1.0 / curation_scaling, curation_target_center_point, curation_source_center_point)        
        bbox = bbox_restrict_in_image_boundary_(bbox, [self._state['im_w'],self._state['im_h']])
        
        bbox = xyxy2xywh(bbox)

        return bbox,confidence_score

def clip_box(box: list, H, W, margin=0):
    x1, y1, w, h = box[0]
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W-margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H-margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2-x1)
    h = max(margin, y2-y1)
    box[0] = x1, y1, w, h
    return box