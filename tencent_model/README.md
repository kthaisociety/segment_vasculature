On my computer I have the data sitting on a different disk than the repo. Hence, I need to use the `mounts` section in the `devcontainer.json`. If you are using the devcontainer then you need to specify the paths to your files as environment variables on your local machine. I also store these enviroment variables in a file `.devcontainer/.devcontainer.env`. This file is loaded into your dev container on runtime.

Example: In your `~/.bashrc` you have 
```bash
export SOURCE_TRAIN=/mnt/e/blood-vessel-segmentation/train
export SOURCE_TEST=/mnt/e/blood-vessel-segmentation/train
export SOURCE_MODEL=/mnt/c/tencent-model
```

Thereafter run the model using:
```bash
python test.py --no_cuda --resume_path trails/models/resnet_50_epoch_110_batch_0.pth.tar --img_list data/val.txt --data_root ./data
```

If you have an Nvidia GPU which support CUDA >= 12, then you can also run:
```bash
python test.py --gpu_id 0 --resume_path trails/models/resnet_50_epoch_110_batch_0.pth.tar --img_list data/val.txt --data_root ./data
```

Otherwise, you need to adjust the PyTorch dependency to support your GPU.

Note that the model is built more than 4 years ago and therefore uses Python 3.7 and other ancient dependencies. I have therefore tried to upgrade the dependencies to make it easy to expand the code base. 

If you do not use the devcontainer, then you might need to do some changes if you don't upgrade `nibabel`. `nibabel` is based on a old numpy version so you might need to swap out `np.float` to `np.float64` inside `quaternions.py`. You might also need to delete the last two lines of code in `/usr/local/lib/python3.11/site-packages/nibabel/pydicom_compat.py` which skips a test. They look like this:
```py
# test decorator that skips test if dicom not available.
dicom_test = np.testing.dec.skipif(not have_dicom,
                                   'could not import dicom or pydicom')
```