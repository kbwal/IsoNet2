import logging
from IsoNet.preprocessing.cubes import create_cube_seeds,crop_cubes
from IsoNet.utils.utils import mkfolder
from IsoNet.utils.plot_metrics import plot_metrics


def run_training(data_list, epochs = 10, mixed_precision = False,
               output_dir = "results", outmodel_path="half", pretrained_model=None,
               ncpus=16, batch_size = 8, acc_batches=2, learning_rate= 3e-4):

    logging.info("Start training!")
    from IsoNet.models.network import Net
    network = Net(filter_base = 64,unet_depth=3, add_last=True)
    if pretrained_model is not None:
       print(f"loading previous model {pretrained_model}")
       network.load(pretrained_model)
    if epochs > 0:
       network.train(data_list, output_dir, outmodel_path=outmodel_path, batch_size=batch_size, epochs = epochs, steps_per_epoch = 1000, 
                           mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate) #train based on init model and save new one as model_iter{num_iter}.h5
    plot_metrics(network.metrics, f"{output_dir}/loss.png")
    return network

def run_old(star_file, epochs = 10, mixed_precision = False,
               output_dir = "results", output_base="half", n_subvolume = 50, pretrained_model=None,
               cube_size = 64, predict_crop_size=96, batch_size = 8, acc_batches=2, learning_rate= 3e-4, limit_res=None):


    logging.info("Start training!")
    from IsoNet.models.network import Net
    network = Net(filter_base = 64,unet_depth=3, add_last=True)
    if pretrained_model is not None:
        print(f"loading previous model {pretrained_model}")
        network.load(pretrained_model)
    if epochs > 0:
        network.train(star_file, output_dir, alpha=1, output_base=output_base, batch_size=batch_size, epochs = epochs, steps_per_epoch = 1000, 
                            mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate) #train based on init model and save new one as model_iter{num_iter}.h5
    plot_metrics(network.metrics, f"{output_dir}/loss_{output_base}.png")

    logging.info("Start predicting!")           
    #out_map = network.predict_map(halfmap, output_dir=output_dir, cube_size = cube_size, crop_size=predict_crop_size, output_base=output_base)

    if limit_res is None:
        out_name = f"{output_dir}/corrected_{output_base}.mrc"
    else:
        out_name = f"{output_dir}/corrected_{output_base}_filtered.mrc"


import starfile
from IsoNet.preprocessing.prepare import get_cubes_list
def run_refine(star_file, epochs = 10, mixed_precision = False,
               output_dir = "results", output_base="half", n_subvolume = 50, pretrained_model=None,
               cube_size = 64,ncpus=16, predict_crop_size=96, batch_size = 8, acc_batches=2, learning_rate= 3e-4, limit_res=None):
    star = starfile.read(star_file)

    mkfolder(output_dir)
    data_dir = output_dir+"/tmpdata"
    mkfolder(data_dir)
    get_cubes_list(star, data_dir,ncpus=ncpus)
    logging.info("Start training!")
    from IsoNet.models.network import Net
    network = Net(filter_base = 64,unet_depth=3, add_last=False)
    if pretrained_model is not None:
       print(f"loading previous model {pretrained_model}")
       network.load(pretrained_model)
    if epochs > 0:
       network.train(data_dir, output_dir, output_base=output_base, batch_size=batch_size, epochs = epochs, steps_per_epoch = 1000, 
                           mixed_precision=mixed_precision, acc_batches=acc_batches, learning_rate = learning_rate) #train based on init model and save new one as model_iter{num_iter}.h5
    plot_metrics(network.metrics, f"{output_dir}/loss_{output_base}.png")

