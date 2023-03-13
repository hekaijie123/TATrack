from typing import Dict
from ..transformer_base import TRACK_TRANSFORMERS, TransformerBase
from data.tracking.methods.SiamFC.pipeline.processor import SiamTrackerProcessor


@TRACK_TRANSFORMERS.register
class RandomCropTransformer(TransformerBase):

    default_hyper_params = dict(
        m_size=224,
        q_size=224,
        num_memory_frames=0,
        template_area_factor=0.0,
        search_area_factor=0.0,
        phase_mode="train",
        color_jitter=0.4,
        template_scale_jitter_factor=0.0,
        search_scale_jitter_factor=0.25,
        template_translation_jitter_factor=0.0,
        search_translation_jitter_factor=3.0,
        gray_scale_probability=0.05,
        interpolation_mode = 'bilinear'
    )

    def __init__(self, seed: int = 0) -> None:
        super(RandomCropTransformer, self).__init__(seed=seed)


    def __call__(self, sampled_data: Dict) -> Dict:
        data1 = sampled_data["data1"]       ##c,h,w
        # print("sampled_len",len(sampled_data["data1"]))

        data2 = sampled_data["data2"]
        sampled_data["data1"] = {}
        nmf = self._hyper_params['num_memory_frames']
        
        # print("data1_len",len(data1))
        for i in range(nmf): 
            im_memory, bbox_memory = data1["image_{}".format(i)], data1["anno_{}".format(i)]
            if(i==0):
                im_m, bbox_m, _ = self.crop( im_memory, bbox_memory,True,True)
            else:
                im_m, bbox_m, _ = self.crop( im_memory, bbox_memory,False,True)
            sampled_data["data1"]['image_{}'.format(i)] = im_m
            sampled_data["data1"]['anno_{}'.format(i)] = bbox_m
        im_query, bbox_query = data2["image"], data2["anno"]
        # if (bbox_query==(-1.,-1.,-2.,-2.)).all():
        #     print("how")        
        im_q, bbox_q, _ = self.crop( im_query, bbox_query,False,False )
        sampled_data["data2"] = dict(image=im_q, anno=bbox_q)
        return sampled_data


    def update_params(self, ) -> None:

        self.m_size = (self._hyper_params['m_size'],self._hyper_params['m_size'])
        self.q_size = (self._hyper_params['q_size'],self._hyper_params['q_size'])
        self.template_area_factor = self._hyper_params['template_area_factor']
        self.search_area_factor = self._hyper_params['search_area_factor']
        self.template_scale_jitter_factor = self._hyper_params['template_scale_jitter_factor']
        self.search_scale_jitter_factor = self._hyper_params['search_scale_jitter_factor']
        self.tempalte_translation_jitter_factor = self._hyper_params['template_translation_jitter_factor']
        self.search_translation_jitter_factor = self._hyper_params['search_translation_jitter_factor']
        self.gray_scale_probability = self._hyper_params['gray_scale_probability']
        self.colar_jitter =  self._hyper_params['color_jitter']
        self.interpolation_mode = self._hyper_params['interpolation_mode']     

        self.crop = SiamTrackerProcessor(self.m_size,self.q_size,self.template_area_factor,self.search_area_factor,
        self.template_scale_jitter_factor,self.search_scale_jitter_factor,
        self.tempalte_translation_jitter_factor,self.search_translation_jitter_factor,
        self.gray_scale_probability,self.colar_jitter,self.interpolation_mode,rng=self._state["rng"])




