import os
import sys
from Datasets.build_rotation_celeba import *
from Datasets.build_rotation_celeba import build_rotation_90_dataset
import subprocess
CFID_dir = os.path.dirname(os.path.abspath(__file__)).strip()
sys.path.append(CFID_dir)

################ Datsets:###################

################# CelebA: ##################
############# Download CelebA ##############
url_link = "https://storage.googleapis.com/kaggle-data-sets/29561/37705/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20210927%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20210927T082441Z&X-Goog-Expires=259199&X-Goog-SignedHeaders=host&X-Goog-Signature=5001039c34e0d7779b73c5809b6ddd6e79e57ab9fdd14a296ec54637920f015fe9f66dcbf59659e4dacbf84761b5f2715cdb516931fc912bdd80bd4c224d80589daa8c83b4db92f39d69fbd754b447b8c443eae540323186e21831113478f7e736c8726a747cd23b540c1e0fdcc02605f4493edb699b0d0a4c937e9a0eb82539424111b052374bfb85c4d233269e5e4e03a31e849d9adfe3998a63eafed213e618a31e2c2e6085bd20ccb2749ce742d9395d645b70340de99af7372989a1b00f91f564c780c2df0605af0ea43b41f5e2610fb202079326c46b14478eedd37fdbcf0596c570f33a050dc1cd1daf97117b08bcb8de0535b6cd0d405189cb47a760"
subprocess.run(["cd", CFID_dir],shell=True)
subprocess.run(["wget","-nv","--show-progress", url_link, "-O", "./Datasets/CelebA_archive", "--progress=bar:force:noscrol"])

######## Create CelebA rotation ~90 ########
build_rotation_90_dataset()

######### Create CelebA inpainting #########
build_rotation_inpainting()