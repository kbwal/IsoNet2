import os 
import sys
import logging
import sys
import mrcfile
from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes
from IsoNet.preprocessing.img_processing import normalize
from IsoNet.utils.missing_wedge import apply_wedge
from multiprocessing import Pool
import numpy as np
from functools import partial
from IsoNet.utils.rotations import rotation_list
# from difflib import get_close_matches
#Make a new folder. If exist, nenew it
# Do not set basic config for logging here
# logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)


def combine_volumes(vin, vout, wedge, normalize_percentile=False):

    #iw_data = normalize(vin, percentile = normalize_percentile)
    #ow_data = normalize(vout, percentile = normalize_percentile)

    orig_data = apply_wedge(vout, wedge, ld1=0, ld2=1) + apply_wedge(vin, wedge, ld1 = 1, ld2=0)
    #orig_data = normalize(orig_data, percentile = normalize_percentile)
    return orig_data



def rotate_cubes(data):
    rotated_data = np.zeros((len(rotation_list), *data.shape))
    old_rotation = True
    if old_rotation:
        for i,r in enumerate(rotation_list):
            data_copy = np.rot90(data, k=r[0][1], axes=r[0][0])
            data_copy = np.rot90(data_copy, k=r[1][1], axes=r[1][0])
            rotated_data[i] = data_copy
    else:
        from scipy.ndimage import affine_transform
        from scipy.stats import special_ortho_group 
        for i in range(len(rotation_list)):
            rot = special_ortho_group.rvs(3)
            center = (np.array(data.shape) -1 )/2.
            offset = center-np.dot(rot,center)
            rotated_data[i] = affine_transform(data,rot,offset=offset)
    return rotated_data

def get_cubes(inp):
    '''
    current iteration mrc(in the 'results') + infomation from orignal subtomo
    normalized predicted + normalized orig -> normalize
    rotate by rotation_list and feed to get_cubes_one
    '''
    mrc1, mrc2, wedge_file, start, data_dir = inp
    from IsoNet.utils.utils import read_mrc
    vin, voxel_size = read_mrc(mrc1)
    vout, voxel_size = read_mrc(mrc2)
    wedge, voxel_size = read_mrc(wedge_file)
    data1 = combine_volumes(vin, vout, wedge, normalize_percentile=False)
    # with mrcfile.open(mrc1) as mrcData:
    #     data1 = mrcData.data.astype(np.float32) * -1
    #data1 = normalize(data1, percentile=False)
    # if mrc2 is None:
    #     with mrcfile.open(mrc2) as mrcData:
    #         data2 = mrcData.data.astype(np.float32) * -1
    #         rot_cube2 = rotate_cubes(data2)
    #data2 = normalize(data2, percentile=False)
    # with mrcfile.open(wedge_file) as mrcData:
    #     wedge = mrcData.data.astype(np.float32)

    rot_cube1 = rotate_cubes(data1)

    for i in range(rot_cube1.shape[0]):
        with mrcfile.new('{}/train_x/x_{}.mrc'.format(data_dir, start+i), overwrite=True) as output_mrc:
            output_mrc.set_data(apply_wedge(rot_cube1[i], wedge).astype(np.float32))
        with mrcfile.new('{}/train_y/y_{}.mrc'.format(data_dir, start+i), overwrite=True) as output_mrc:
            output_mrc.set_data(rot_cube1[i].astype(np.float32))
        # if mrc2 is None:
        #     with mrcfile.new('{}/train_x2/x_{}.mrc'.format(data_dir, start+i), overwrite=True) as output_mrc:
        #         output_mrc.set_data(apply_wedge(rot_cube2[i], wedge).astype(np.float32))
        #     with mrcfile.new('{}/train_y2/y_{}.mrc'.format(data_dir, start+i), overwrite=True) as output_mrc:
        #         output_mrc.set_data(rot_cube2[i].astype(np.float32))

def get_cubes_list(star, data_dir, ncpus, start_over=False):
    '''
    generate new training dataset:
    map function 'get_cubes' to mrc_list from subtomo_dir
    seperate 10% generated cubes into test set.
    '''
    import os
    dirs_tomake = ['train_x', 'train_y']
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for d in dirs_tomake:
        folder = '{}/{}'.format(data_dir, d)
        if not os.path.exists(folder):
            os.makedirs(folder)

    inp=[]
    mrc_list = star['rlnParticleName'].tolist()
    wedge_list = star['rlnWedgeName'].tolist()
    print(star.columns)
    if 'rlnCorrectedParticle' in star.columns and not start_over:
        print("addcorrected")
        mrc2_list = star['rlnCorrectedParticle'].tolist()    
        print(mrc2_list)
        for i,mrc in enumerate(mrc_list):
            inp.append((mrc,mrc2_list[i], wedge_list[i], i*len(rotation_list),data_dir))
    else:
        for i,mrc in enumerate(mrc_list):
            inp.append((mrc,mrc, wedge_list[i], i*len(rotation_list),data_dir))

    if ncpus > 1:
        with Pool(ncpus) as p:
            p.map(get_cubes,inp)
    else:
        for i in inp:
            print(i)
            get_cubes(i)


# def get_noise_level(noise_level_tuple,noise_start_iter_tuple,iterations):
#     assert len(noise_level_tuple) == len(noise_start_iter_tuple) and type(noise_level_tuple) in [tuple,list]
#     noise_level = np.zeros(iterations+1)
#     for i in range(len(noise_start_iter_tuple)-1):
#         #remove this assert because it may not be necessary, and cause problem when iterations <3
#         #assert i < iterations and noise_start_iter_tuple[i]< noise_start_iter_tuple[i+1]
#         noise_level[noise_start_iter_tuple[i]:noise_start_iter_tuple[i+1]] = noise_level_tuple[i]
#     assert noise_level_tuple[-1] < iterations 
#     noise_level[noise_start_iter_tuple[-1]:] = noise_level_tuple[-1]
#     return noise_level

# def generate_first_iter_mrc(mrc,settings):
#     '''
#     Apply mw to the mrc and save as xx_iter00.xx
#     '''
#     # with mrcfile.open("fouriermask.mrc",'r') as mrcmask:
#     root_name = mrc.split('/')[-1].split('.')[0]
#     extension = mrc.split('/')[-1].split('.')[1]
#     with mrcfile.open(mrc) as mrcData:
#         orig_data = normalize(mrcData.data.astype(np.float32)*-1, percentile = settings.normalize_percentile)

#     orig_data = apply_wedge(orig_data, ld1=1, ld2=0)
    
#     #prefill = True
#     if settings.prefill==True:
#         rot_data = np.rot90(orig_data, axes=(0,2))
#         rot_data = apply_wedge(rot_data, ld1=0, ld2=1)
#         orig_data = rot_data + orig_data

#     orig_data = normalize(orig_data, percentile = settings.normalize_percentile)
#     with mrcfile.new('{}/{}_iter00.{}'.format(settings.result_dir,root_name, extension), overwrite=True) as output_mrc:
#         output_mrc.set_data(-orig_data)

# def prepare_first_iter(settings):
#     if settings.ncpus >1:
#         with Pool(settings.ncpus) as p:
#             func = partial(generate_first_iter_mrc, settings=settings)
#             p.map(func, settings.mrc_list)
#     else:
#         for i in settings.mrc_list:
#             generate_first_iter_mrc(i,settings)
#     return settings

