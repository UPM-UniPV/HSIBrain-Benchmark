import re
import os
import json
import yaml
import numpy as np

from scipy.stats import ttest_rel, wilcoxon, friedmanchisquare



base_res_path = "/path/to/results/folders/"
model_list = ['LiteDwNet', 'ViT', 'HiT', 'RSSAN', 'GhostNet', 'DBDA', 'Hyb3D_2D', 'SpectralFormer', 'EfficientNet', 'SSAN']


""" METRICS RANKING FOR p-VALUE"""

met = "tumor"

mod_dict = {}

for model in model_list:
	
	mod_dict[model] = []

	folds = []
	seed_ix = []
	for fold_ in os.listdir(f"{base_res_path}{model}/"):
		if os.path.isdir(f"{base_res_path}{model}/{fold_}/"):
			folds.append(fold_)
			seed_ix.append(int(fold_[-1]))
	
	sorted_folds = [folds[s_ix] for s_ix in np.argsort(seed_ix)]

	for sfold in sorted_folds:
				
		res_dict = json.load(open(f"{base_res_path}{model}/{sfold}/test/metrics.json"))
		pat_list = list(res_dict["individual"].keys())
		pat_list.sort()

		for pat_id in pat_list:
		
			if met == "tumor":
				pat_met = res_dict["individual"][pat_id]["F1"]["per_class"][1]
			elif met == "OA":
				pat_met = res_dict["individual"][pat_id]["F1"]["global"]
			else:
				raise ValueError("It is necessary to specify a metric for model ranking.")
	
			mod_dict[model].append(float(pat_met))




rank_mod_list = ['DBDA', 'EfficientNet', 'Hyb3D_2D', 'LiteDwNet', 'HiT', 'SSAN', 'SpectralFormer', 'GhostNet', 'ViT', 'RSSAN']


print("\n\np-values:\n")

adj = 1

for i in range(len(rank_mod_list)-adj):
	
	
	print(rank_mod_list[i], rank_mod_list[i+adj])

	cand_mets, baseline_mets = mod_dict[rank_mod_list[i]], mod_dict[rank_mod_list[i+adj]]
	
	p_value_wil = wilcoxon(cand_mets, baseline_mets, alternative='greater').pvalue
	
	print(p_value_wil, "\n")
