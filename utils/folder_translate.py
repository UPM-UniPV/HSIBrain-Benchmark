import os
import yaml
import shutil


def check_dirs(*args):
	for dir_ in args:
		if not os.path.exists(dir_):
			os.makedirs(dir_)
			


base_res_path = "/path/to/default/mlruns_folder/"

base_trans_path = "/path/to/ml_runs_readable/"


targ_models = ["LiteDwNet", "ViT", "HiT", "RSSAN", "GhostNet",
			   "DBDA", "Hyb3D_2D", "SpectralFormer", "EfficientNet", "SSAN"]

ignore_folders = [".trash", "0", "models", "tags", "meta.yaml"]

targ_met = "fscore_class"

name_convert = {"LP": "LP", "madrid":"M", "Madrid": "M", "M":"M"}


check_dirs(base_trans_path)

for path in os.listdir(base_res_path):

	if path in ignore_folders:
		continue

	local_run = f"{base_res_path}{path}/"

	yaml_dict = yaml.safe_load(open(f"{local_run}meta.yaml", "r"))

	current_model = yaml_dict["name"]

	if current_model not in targ_models:
		continue
	
	for sub_path in os.listdir(local_run):

		if sub_path in ignore_folders:
			continue
		
		seed = open(f"{base_res_path}{path}/{sub_path}/params/seed").read()
		db_name = name_convert[open(f"{base_res_path}{path}/{sub_path}/params/db_name").read()]
		patch_size = open(f"{base_res_path}{path}/{sub_path}/params/patch_size").read()

		if int(patch_size) != 7:
			continue
		
		mets_folder = f"{base_res_path}{path}/{sub_path}/artifacts/"

		check_dirs(f"{base_trans_path}{db_name}/{current_model}/{seed}/")
		check_dirs(f"{base_trans_path}{db_name}/{current_model}/{seed}/model/")

		for mets_subfold in os.listdir(mets_folder):

			
			if "best_model" in mets_subfold:
				# shutil.copy(f"{mets_folder}/{mets_subfold}/data/model.pth",
				# 			f"{base_trans_path}{db_name}/{current_model}/{seed}/model/")
				
				shutil.copytree(f"{mets_folder}/{mets_subfold}",
							f"{base_trans_path}{db_name}/{current_model}/{seed}/model/",
							dirs_exist_ok=True)
				continue
				
			for mets_files in os.listdir(f"{mets_folder}/{mets_subfold}/"):
				
				# print(f"{mets_folder}/{mets_subfold}/{mets_files}", f"{base_trans_path}{db_name}/{current_model}/{seed}/")
				shutil.copy(f"{mets_folder}/{mets_subfold}/{mets_files}",
							f"{base_trans_path}{db_name}/{current_model}/{seed}/")
			