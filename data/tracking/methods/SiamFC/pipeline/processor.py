import torch
from data.tracking.methods.SiamFC.common.siamfc_curation import do_SiamFC_curation
import numpy as np
from torchvision.transforms import Grayscale
from .pipeline import SiamTracker_training_prepare_SiamFC_curation, build_SiamTracker_image_augmentation_transformer
from data.operator.bbox.spatial.cxcywh2xyxy import bbox_cxcywh2xyxy
from data.operator.bbox.spatial.vectorized.torch.utility.half_pixel_offset.image import bbox_restrict_in_image_boundary_
class SiamTrackerProcessor:             ##有用
    def __init__(self,
                 template_size, search_size,
                 template_area_factor, search_area_factor,
                 template_scale_jitter_factor, search_scale_jitter_factor,
                 template_translation_jitter_factor, search_translation_jitter_factor,
                 gray_scale_probability,
                 color_jitter, interpolation_mode,rng):
        self.template_size = template_size
        self.search_size = search_size
        self.template_area_factor = template_area_factor
        self.search_area_factor = search_area_factor
        self.template_scale_jitter_factor = template_scale_jitter_factor
        self.search_scale_jitter_factor = search_scale_jitter_factor
        self.template_translation_jitter_factor = template_translation_jitter_factor
        self.search_translation_jitter_factor = search_translation_jitter_factor
        self.gray_scale_probability = gray_scale_probability
        self.interpolation_mode = interpolation_mode
        self.transform = build_SiamTracker_image_augmentation_transformer(color_jitter, True)
        self.gray_scale_transformer = Grayscale(3)          
        self.rng = rng

    def __call__(self, image, bbox, is_z,is_special):

        image = image.to(torch.float)
        if (is_z):
            curated_bbox, curation_parameter = SiamTracker_training_prepare_SiamFC_curation(
                bbox, self.template_area_factor,
                self.template_size,
                self.template_scale_jitter_factor,
                self.template_translation_jitter_factor,self.rng)
            curated_image, _ = do_SiamFC_curation(image, self.template_size, curation_parameter, self.interpolation_mode)
            curated_image /= 255.
        else :
            if(is_special):
                if torch.rand(1)>=0.0:
                    curated_bbox, curation_parameter = SiamTracker_training_prepare_SiamFC_curation(
                        bbox, self.search_area_factor,
                        self.search_size,
                        0.15,
                        1.0,self.rng)
                    curated_image, _ = do_SiamFC_curation(image, self.search_size, curation_parameter, self.interpolation_mode)            
                    curated_bbox = torch.tensor(curated_bbox)
                    curated_bbox[2:] -= curated_bbox[:2]
                    curated_bbox[2:] = curated_bbox[2:] / torch.exp(torch.randn(2) * 0.15)
                    curated_bbox[:2] = torch.tensor(self.search_size)/2.0
                    curated_bbox = bbox_cxcywh2xyxy(curated_bbox)
                    curated_bbox = torch.tensor(curated_bbox)
                    bbox_restrict_in_image_boundary_(curated_bbox, self.search_size)
                    curated_image /= 255.
                # else:
                #     curated_bbox, curation_parameter = SiamTracker_training_prepare_SiamFC_curation(
                #         bbox, self.search_area_factor,
                #         self.search_size,
                #         0.25,
                #         3.0,self.rng)

                #     curated_image, curated_mean = do_SiamFC_curation(image, self.search_size, curation_parameter, self.interpolation_mode)            
                #     curated_bbox = torch.tensor(curated_bbox)
                #     # curated_bbox = curated_bbox.to(torch.int)                    
                #     # curated_image[:,curated_bbox[1]:curated_bbox[3]+1,curated_bbox[0]:curated_bbox[2]+1]=curated_mean.unsqueeze(1).unsqueeze(1)
                #     curated_bbox[2:] -= curated_bbox[:2]
                #     curated_bbox[2:] = curated_bbox[2:] / torch.exp(torch.randn(2) * 0.25)
                #     curated_bbox[:2] = torch.tensor(self.search_size)/2.0
                #     curated_bbox = bbox_cxcywh2xyxy(curated_bbox)
                #     curated_bbox = torch.tensor(curated_bbox)
                #     bbox_restrict_in_image_boundary_(curated_bbox, self.search_size)
                #     curated_image /= 255.                    

            else:
                curated_bbox, curation_parameter = SiamTracker_training_prepare_SiamFC_curation(
                    bbox, self.search_area_factor,
                    self.search_size,
                    self.search_scale_jitter_factor,
                    self.search_translation_jitter_factor,self.rng)
                curated_image, _ = do_SiamFC_curation(image, self.search_size, curation_parameter, self.interpolation_mode)            
                curated_image /= 255.

        if self.rng.random() < self.gray_scale_probability:
            curated_image = self.gray_scale_transformer(curated_image)

        curated_image = self.transform(curated_image)       ##为什么会有不少超过1的像素值？？？？

        return curated_image,curated_bbox,curation_parameter


