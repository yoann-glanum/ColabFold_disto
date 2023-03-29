# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 12:09:35 2023

@author: Yoann
"""


import numpy as np

import json
import logging

from pathlib import Path

from typing import Any, Callable, Dict, List, Optional, Tuple, Union



# 
class Module1_Predictor():
    
    # ---- Attributes ----
    
    # -- sequence information
    sequence: str #TODO add
    nb_peptide: int
    
    # -- rank info storage
    multimer_list: list
    max_pep_plddt_list: list
    max_sum_to_20_prob_list: list
    
    # -- inference weights
    sub_vars_mapping: dict
    sub_vars_raw_vals: dict
    sub_vars_norm_vals: dict
    ens_method: str
    ens_threshold: float
    raw_prediction: float
    predicted_class: int
    weights_filepath: str
    
    # ---- Functions ----
    
    # -- class functions
    
    # 
    def __init__(self):
        
        self.multimer_list = []
        self.max_pep_plddt_list = []
        self.max_sum_to_20_prob_list = []
        
        self.sub_vars_mapping = {"med_all_ranks_multimer_score": self.multimer_list, 
                                 "avg_all_ranks_max_pep_plddt": self.max_pep_plddt_list, 
                                 "avg_all_ranks_20_max_prob": self.max_sum_to_20_prob_list}
        self.sub_vars_raw_vals = {}
        self.sub_vars_norm_vals = {}
        
        return
    
    # 
    def __repr__(self):
        
        #TODO
        
        return
    
    # -- init run
    
    # get value and set it
    def set_nb_peptide(self, input_sequence: Union[List[str], str], logger = None) -> int:
        
        if type(input_sequence) == str:
            if logger:
                logger.info(input_sequence)
            else:
                print(input_sequence)
            temp_nb_peptide = len(input_sequence.split(':')[0])
            self.nb_peptide = temp_nb_peptide
        elif type(input_sequence) == list:
            if logger:
                logger.info(input_sequence)
            else:
                print(input_sequence)
            temp_nb_peptide = len(input_sequence[0])
            self.nb_peptide = temp_nb_peptide
        
        #dev only?
        if logger:
            logger.info(f"Found {self.nb_peptide} peptide AAs in sequence")
        else:
            print(f"Found {self.nb_peptide} peptide AAs in sequence")
        
        return temp_nb_peptide
    
    # -- for each sub-prediction
    
    # 
    def add_multimer(self, new_iptm: Union[np.ndarray, float], 
                     new_ptm: Union[np.ndarray, float]):
        if type(new_iptm) != float:
            new_iptm = new_iptm[0]
        if type(new_ptm) != float:
            new_ptm = new_ptm[0]
            
        self.multimer_list.append(0.8*float(new_iptm) + 0.2*float(new_ptm))
        
        return
    
    # 
    def add_max_pep_plddt(self, new_plddt: Union[List, np.ndarray]):
        
        pep_plddt = new_plddt[0:self.nb_peptide]
        self.max_pep_plddt_list.append(np.max(pep_plddt))
        
        return
    
    # 
    # made for a slice 16 - 22 mode
    def add_20_max_prob(self, contact_map: Union[List, np.ndarray], 
                        slices: Union[List, np.ndarray]):
        
        sum_to_15 = contact_map - np.sum(slices[:, :, :3+1], axis = -1)
        sum_to_20 = sum_to_15 + np.sum(slices[:, :, 0:4+1], axis = -1)
        sum_to_20 = sum_to_20[self.nb_peptide:, :self.nb_peptide]
        
        max_prob = np.max(sum_to_20)
        self.max_sum_to_20_prob_list.append(max_prob)
        
        return
    
    # -- prediction
    
    # 
    #TODO precaution not needed, remove?
    def refresh_mapping_dict(self):
        
        self.sub_vars_mapping = {"med_all_ranks_multimer_score": self.multimer_list, 
                                 "avg_all_ranks_max_pep_plddt": self.max_pep_plddt_list, 
                                 "avg_all_ranks_20_max_prob": self.max_sum_to_20_prob_list}
        
        return
    
    # 
    def predict_module(self, weights_json_fp: str, logger = None):
        
        with open(weights_json_fp, 'r') as j:
             weights_json = json.loads(j.read())
        self.weights_filepath = weights_json_fp
        
        if len(weights_json.keys()) != 1:
            print("Can not take multiple ensembling functions yet")
            #TODO
            return "error"
        ensemble_name = list(weights_json.keys())[0]
        self.ens_method = ensemble_name
        
        ens_dict = weights_json[ensemble_name]
        self.ens_threshold = float(ens_dict['post_scale_ens_threshold'])
        
        self.refresh_mapping_dict()
        #TODO add check for vars min == vars max
        for subvar in ens_dict['vars_min_pre_scale']:
            # get average over all ranks : seeds * models
            if subvar[:3] == 'avg':
                subvar_avg = np.mean(self.sub_vars_mapping[subvar])
            elif subvar[:3] == 'med':
                subvar_avg = np.median(self.sub_vars_mapping[subvar])
            subvar_avg = float(subvar_avg) # for json float64
            self.sub_vars_raw_vals[subvar] = subvar_avg
            
            # normalise with min max from the weights
            ratio_num = subvar_avg - ens_dict['vars_min_pre_scale'][subvar]
            ratio_denum = ens_dict['vars_max_pre_scale'][subvar] - ens_dict['vars_min_pre_scale'][subvar]
            normalised_value = ratio_num/ratio_denum
            normalised_value = float(normalised_value) # for json float64
            self.sub_vars_norm_vals[subvar] = normalised_value
        
        # raw prediction
        raw_pred = np.mean([value for key, value in self.sub_vars_norm_vals.items()])
        raw_pred = float(raw_pred)
        predicted_class = raw_pred >= self.ens_threshold
        self.raw_prediction = raw_pred
        self.predicted_class = bool(predicted_class)
        
        #TODO add distance to threshold?
        
        #dev only?
        if logger:
            logger.info(f"module 1 prediction done : Binary Pep-Prot Interaction predicted as \
                        {self.predicted_class} (raw value of {self.raw_prediction})")
        else:
            print(f"module 1 prediction done : Binary Pep-Prot Interaction predicted as \
                  {self.predicted_class} (raw value of {self.raw_prediction})")
        
        return
    
    # -- save
    
    # 
    def save_ens_prediction(self, filepath: str, logger = None):
        
        save_dict = {'raw_sub_var_values':self.sub_vars_raw_vals, 
                     'norm_sub_var_values':self.sub_vars_norm_vals, 
                     'ensembling_method':self.ens_method, 
                     'ensembling_threshold':self.ens_threshold, 
                     'raw_prediction':self.raw_prediction, 
                     'predicted_class':self.predicted_class
                     }
        with open(filepath, 'w') as f:
            json.dump(save_dict, f)
        
        #dev only?
        if logger:
            logger.info(f"file saved at {filepath}")
        else:
            print(f"file saved at {filepath}")
        
        return
# 
