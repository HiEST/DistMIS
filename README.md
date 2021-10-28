<br/><br/><br/><br/>
<h1 align="center">
  Distributing Deep Learning Hyperparameter Tuning for 3D Medical Image Segmentation
  <br/><br/>
  <img src="./images/3d_unet.svg" alt="Distributing Deep Learning for 3D Medical Image Segmentation" width="400">
  <img src="./images/pipeline.png" alt="Distributing Deep Learning for 3D Medical Image Segmentation" width="450">
  <br/><br/>
  
</h1>

<p align="center">
  <a href="#quick-introduction">Quick Introduction </a> •
  <a href="#setup">Setup</a> •
  <a href="#how-to-use">How To Use</a> •
</p>

## Quick Introduction
Official repository. And quick introduction to the paper?

### Citation
If you use our code for your research please cite our [preprint](./):
> "TODO: Citar paper"

<br/><br/>

## Setup
### Installation:
Just download the code, and execute the scripts inside the directory. You can clone the repository using:
```console
foo@bar:~$ git clone https://github.com/oriolaranda/dist-dl-3d-mis.git
foo@bar:~$ cd dist-dl-3d-mis/
foo@bar:~$ python -m pip install -r requirements.txt
```

### Requirements:

The scrips work with python >= 3.7.4 and uses the following packages:
```
tensorflow==2.3.0
tensorflow-addons==0.13.0
ray==1.4.1
numpy==1.18.4
imageio==2.6.1
matplotlib==3.4.3
nibabel==3.2.1
tqdm==4.46.1
```
### Dataset:

The `tfrecord` script assumes that the _Task01_BrainTumor.tar_ has been downloaded from its official webpage [MSD Challenge](http://medicaldecathlon.com/) and extracted to your local machine. The directory of the original dataset is slightly used in the next sections and it refers to the extracted folder. This folder must contain:
* dataset.json: Information about the dataset.
* imagesTr: Brain images for training.
* labelsTr: Label images for training.

> Note that the folder imagesTs is not used because their respective labels are not provided. The training, validation and test sets are split using the data available (imagesTr and labelsTr).

The data is composed by 3D samples of shape (240,240,155,4) for the brain images and (240,240,155,1) for the ground truth segmentation masks. The data format is NIfTI, commonly used in medical imaging. A transposition of the channels is applied and the shape of the samples is reduced to (4,240,240,152) and (1, 240,240,152). Additionally, standardization is applied to the brain images and the 3 clases for the labels are joined to form a binary clasification problem: pixel is tumor (1) or is not (0).

In the following figure the 4 channels from the brain images and the ground truth are shown.
<p align="center">
<img src="./images/dataset_msd.png" alt="Brain Tumor MSD dataset">
</p>
<br/><br/><br/><br/>

## How To Use
The framework is composed by 4 main scripts: `tfrecord`, `visualize`, `data_parallel` and `exp_parallel`.

### Tfrecord
The `tfrecord` script reads the `dataset.json` file located inside the dataset folder, and creates a tf.data.Dataset from the imagesTr and labelsTr samples. It requires the orignal dataset directory path as argument, e.g. /home/Task01_BrainTumor/. If the target directory is provided, the tf.data.Dataset is serialized and saved in TFRecord format into that directory. Running this script is **the first step** in order to run the other scripts. Serializing the dataset the memory footprint will be larger but it has good benefits:
* Perform offline preprocessing and data augmentation.
* Optimize data reading, online preprocessing and data augmentation.
It can seem the benefits are poor, but in Deep Learning applications with this type of data, these optimization techniques save hours of training.

##### Usage:
```console
foo@bar:~$ python tfrecord.py --help
usage: tfrecord.py [-h] --source-dir SOURCE_DIR [--target-dir TARGET_DIR] [--split SPLIT] [--reshape RESHAPE]

optional arguments:
  -h, --help                   Show this help message and exit
  --source-dir SOURCE_DIR      Path: Source data directory. The directory must contain the dataset.json file, 
                               and the two folders imagesTr, labelsTr.
  --target-dir TARGET_DIR      Path: Target data directory. It must exist. It is where the TFRecord data will be 
                               written to.
  --split SPLIT                Tuple: Ratios into which the dataset will be divided (all sum up to 1). Train, 
                               validation and test set. Default=(0.7, 0.15, 0.15).
  --reshape RESHAPE            Tuple: Shape of the written data. Default=(240, 240, 152) which is its original 
                               shape. If a different shape is provided, a trilinear resize will be applied to 
                               the data.
```

##### Examples:
Creating the target directory
```console
foo@bar:~$ mkdir dataset

```
Creating the tfrecord dataset into the created directory _dataset_.
```console
foo@bar:~$ python tfrecord.py --source-dir /home/Task01_BrainTumor/ --target-source /home/dataset/

```
Creating a tfrecord dataset with smaller size data, and different split sets.

```console
foo@bar:~$ python tfrecord.py --source-dir /home/Task01_BrainTumor/ --target-source /home/dataset/ --reshape (120, 120, 152) --split (0.8, 0.1, 0.1)
```

<br/><br/>

### Visualize
The `visualize` script is just an auxiliary script for visualizing the data after doing the tfrecord
and other possible transformations, e.g. offline data_augmentation. It is also useful for debugging
purposes, e.g. testing some transformation or preprocessing functions, before deploying.

##### Usage:
```console
foo@bar:~$ python visualize.py --help
usage: visualize.py [-h] --dataset-dir DATASET_DIR [--sample SAMPLE] [--data-shape DATA_SHAPE] 
                         [--no-screen NO_SCREEN]

optional arguments:
  -h, --help                    Show this help message and exit
  --dataset-dir DATASET_DIR     Path: TFRecord dataset directory.
  --sample SAMPLE               Int: Sample to visualize. Default=0. It has to be 
                                0 <= sample <= size_dataset.
  --data-shape DATA_SHAPE       Tuple: Shape of the data in the dataset path provided. 
                                Default=(240, 240, 152) which is the orginal data shape.
  --no-screen                   No X session (graphical Linux desktop) mode. If it used, a 
                                GIF file will be saved in the current directory ('./') 
                                containing the plotted image.
```

##### Examples:
Visualizing the written data in TFRecord format, the default sample is the number 0.
```console
foo@bar:~$ python visualize.py --dataset-dir /home/dataset/
```
Visualizing the sample number 350, since we are working via ssh with no x session, we enable _--no-screeen_ flag to save a GIF file.
```console
foo@bar:~$ python visualize.py --dataset-dir /home/dataset/ --sample 350 --no-screen
```
<br/>
<p align="center">
<img src="./images/sample_2.gif" alt="Visualization of one sample">
</p>

<br/><br/>

### Data Parallelism
The `data_parallel` script is the first approach presented in the paper, given a model in tensorflow and a TFRecord dataset it performs data parallelism. 
Data parallelism consists in, given n GPUs, the model is replicated n times and each replica is sent to a GPU. After that, the data is split into n chunks, i.e. the batch size is diveded by n, these chunks are distributed across the GPUs, where each chunk is assigned to a GPU. If we are training m models and m >= n, then we proceed sequentially for each model. Since we are using more than 1 GPU we are speeding-up the training of each model, and therefore, the m models.
Our cluster has 4 GPUs per node, so if the number of GPUs used is less than 4, i.e. we are using only one node, tf.MirroredStrategy is used. For multi-node, i.e. >= 4 GPUs, we use ray.cluster which handles all the comunications between nodes and ray.sgd which is a wrapper around tf.MultiWorkerMirroredStrategy.
Both tf.MirroredStrategy and tf.MultiWorkerMirroredStrategy are built-in functions from tensorflow distributed API.

##### Usage:
First of all a configuration JSON file is required to execute the script. This configuration file has some parameters which are required shown below:

- num_replicas
  > (int) Number of GPUs to train each model. The model will be replicated this number of times.
- batch_size_per_replica
  > (int) Batch size handled by each replica, i.e. GPU. In data parallelism the total_batch_size is batch_size_per_replica * num_replicas.
- num_epochs:
  >(int) Number of epochs each model will train for.
- debug:
  > (bool) Mode debug. If true, no tensorboard files will be saved and the training verbosity is set for every step. Otherwise the training verbosity is set for epoch and the tensorboard files will be saved.

Once the configuration file is ready, if we are using a **single node** we can just call the script.
```console
foo@bar:~$ python data_parallel.py --help
usage: data_parallel.py [-h] --config CONFIG

optional arguments:
  -h, --help       Show this help message and exit
  --config CONFIG  Path: Json file configuration
```

If we are using **multi node**, we first need to initialize a ray cluster and then execute the script as above. Please refer to the section [Multi-node Ray Cluster](#multi-node-ray-cluster).

##### Examples:
So first, let's define our config JSON file named _config.json_.
```python
{
    "num_replicas": 4,
    "batch_size_per_replica": 2,
    "num_epochs": 20,
    "debug": false
}
```
Afterwards, we can simply call the script with our config file.
```console
foo@bar:~$ python exp_parallel.py --config ./config.json
```

<br/><br/>

### Experiment Parallelism
The `exp_parallel` script is the second approach presented in the paper, given a model in tensorflow and a TFRecord dataset it performs experiment parallelism using ray.tune which manages all the low level parallelism implementaion.
Experiment parallelism consists in, given n GPUs, m models and m >= n, assigning a model to each GPU available. Hence, we are training n models at the same time, speeding-up the computations of all the m models.
As mentioned above our cluster has 4 GPU per node, so if the number of GPUs used is less than 4, i.e. we are using only one node, ray.tune is used. For multi-node, i.e. >= 4 GPUs, we use ray.cluster which handles all the comunications between nodes and ray.tune.

##### Usage:
First of all a configuration JSON file is required to execute the script. This configuration file has some parameters which are required shown below:

- batch_size_per_replica       
  > (int) Batch size handled by each replica, i.e. GPU. The total_batch_size is batch_size_per_replica * num_replicas. Since in experiment parallelism we are not applying data parallelism and we train each model with a GPU, num_replicas = 1 and total_batch_size = batch_size_per_replica.
- num_epochs:
  >(int) Number of epochs each model will train for.
- debug:
  > (bool) Mode debug. If true, no tensorboard files will be saved and the training verbosity is set for every step. Otherwise the training verbosity is set for epoch and the tensorboard files will be saved.

In order to execute the script first we need to start a ray.cluster with the required resources, i.e. we want to use NUM_GPUS and NUM_CPUS. If we are using a **single node** then we can type the following command. If we are using **multi-node**, please ignore this command and refer to the to the section [Multi-node ray cluster](#multi-node-ray-cluster).
```console
foo@bar:~$ ray start --head --num-cpus=NUM_CPUS --num-gpus=NUM_GPUS
```
Once the ray cluster is started, we can call our script with our configuration json file.
```console
foo@bar:~$ python exp_parallel.py --help
usage: exp_parallel.py [-h] --config CONFIG

optional arguments:
  -h, --help       Show this help message and exit
  --config CONFIG  Path: Json file configuration
```
It is a well practice to shutdown the ray cluster when the work is done.
```console
foo@bar:~$ ray stop
```


##### Examples:
First we define out config as JSON file named _config.json_ and afterwards we initialize a ray cluster with 20 CPUs and 2 GPUs.
```python
{
    "batch_size_per_replica": 2,
    "num_epochs": 20,
    "debug": false
}
```
```console
foo@bar:~$ ray start --head --num-cpus=20 --num-gpus=2
```
Finally we can call the script with our config file.
```console
foo@bar:~$ python exp_parallel.py --config ./config.json
```

<br/><br/>

### Multi-node Ray Cluster
In our case we are using a cluster with 4 GPUs per node, so given n GPUs for n >= 4, we are using multi-node.
If you are using **multi-node**, you need to start a ray cluster in a different way from what is shown in the previous sections. Once the cluster is initialized you can run the script as usual.
Here we show an example, a bash script to start a ray cluster using **SLURM**.

```bash
#!/bin/bash
# shellcheck disable=SC2206
# THIS FILE IS GENERATED BY AUTOMATION SCRIPT! PLEASE REFER TO ORIGINAL SCRIPT!
# THIS FILE IS MODIFIED AUTOMATICALLY FROM TEMPLATE AND SHOULD BE RUNNABLE!

#SBATCH --job-name 16_gpus
#SBATCH -D .
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err

### This script works for any number of nodes, Ray will find and manage all resources
#SBATCH --nodes=4
### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=160
#SBATCH --gres='gpu:4'
#SBATCH --time 48:00:00
#SBATCH --exclusive

# Load modules or your own conda environment here
module purge; module load gcc/8.3.0 cuda/10.2 cudnn/7.6.4 nccl/2.4.8 tensorrt/6.0.1 \
openmpi/4.0.1 atlas/3.10.3 scalapack/2.0.2 fftw/3.3.8 szip/2.1.1 ffmpeg/4.2.1 opencv/4.1.1 \
python/3.7.4_ML ray/1.4.1

# ===== DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING =====
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") # Getting the node names
nodes_array=($nodes)

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w "$node_1" hostname --ip-address) # making redis-address

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$ip" == *" "* ]]; then
  IFS=' ' read -ra ADDR <<< "$ip"
  if [[ ${#ADDR[0]} -gt 16 ]]; then
    ip=${ADDR[1]}
  else
    ip=${ADDR[0]}
  fi
  echo "IPV6 address detected. We split the IPV4 address as $ip"
fi

port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w "$node_1" \
  ray start --head --num-cpus 40 --node-ip-address="$ip" --port=$port --redis-password="$redis_password" --block &
sleep 30

worker_num=$((SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((i = 1; i <= worker_num; i++)); do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w "$node_i" ray start --num-cpus 40 --address "$ip_head" --redis-password="$redis_password" --block &
  sleep 5
done
# ===== Call your code below =====
```
With the first part of the script we have requested the resources via slurm and we have started a ray cluster. The script continues with the following lines where you can call the script you wish.
```bash
export PYTHONUNBUFFERED=1
python data_parallelism.py --config ./config.json
```
