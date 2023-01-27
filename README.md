# Wind farm wake modeling using cDCGAN

Predict the velocity field around a wind turbine given an inflow wind profile.

Based on the work by Zhang J, Zhao X, "Wind farm wake modeling based on deep convolutional
conditional generative adversarial network", Energy, [https://doi.org/10.1016/j.energy.2021.121747](https://doi.org/10.1016/j.energy.2021.121747)

Limitations:

- Only trained with streamwise velocity component (Ux) for now
- It generates only horizontal planes at hub's height (90m)

![Image comparison test](https://github.com/maxibove13/wakeGAN/blob/main/figures/test/image_comparison_test.png)

# Repo usage

## Pull preprocessed data from remote storage

### enable use of gdrive service account:
```
dvc remote modify storage gdrive_use_service_account true
```
### add gdrive credentials to dvc remote:
```
dvc remote modify storage --local gdrive_service_account_json_file_path <credentials_file_name>.json
```
### pull dataset from remote:
```
dvc pull
```
<details>
<summary>(Optional) Preprocess dataset from raw data (CFD simulations)</summary>

## Generate preprocessed dataset 

- For now, only in `medusa16` server where the raw data is stored, that is, CFD (caffa3d) simulation outputs.

### extract images from CFD simulations outputs:
```
./src/data/make_dataset.py
```
### Split data between training and testing:

```
./src/data/split_data.py --ratio 0.9 0.1
```
</details>



## Train the wakeGAN:

```
./train.py
```

- You can modify hyperparameters in `config.yaml` file
- You can monitor the training both watching the prints in the shell and checking the `/figures/monitor` folder. This folder contains three figures that update every epoch:
    - `image_comparison.png`: watch real vs. synth generated images for both training and testing set
    - `metrics.png`: track adversarial loss and RMSE evolution over epochs
    - `err_evol.png`: watch error (just the difference in m/s) between real and synth images for training and testing set  

## Test the wakeGAN:

```
./test.py
```

Check generated files in `/figures/test`:
- `image_comparison_test.png`: flow field comparison between real and synth images for four random samples
- `image_comparison_err_test.png`: error for those four random samples
- `profiles.png`: wind profiles at different streamwise position in relation to the wind turbine for both the ground truth and the generated flow.


# Data pipeline overview:

1) CFD simulations of a WF
2) Horizontal slices at hub's height of mean horizontal velocity ( $U_x$ , $U_y$ )
3) Crop slices into several images around each WT of the WF.
4) Save them as image files mapped with a certain $v_{min}$ and $v_{max}$. ( $v_{min}$ , $v_{max}$ ) -> ( $0$ , $255$ )
5) Read them, convert them to float32, rescale them to ( $-1$ , $1$ )
6) Extract first column of pixels on each channel (inflow velocity)
7) Transform to tensor
8) For each fold:
    For each epoch:
        For each minibatch:
            - Generate fake image given inflow
            - Pass real, fake and inflows to discriminator
            - Evaluate loss, backprop on Disc and Gen


# cDCGAN architecture

## Generator

<p align='center'>
    <img src="https://github.com/maxibove13/wakeGAN/blob/main/figures/reference/gen_arch.png" alt="generator" width="800"/>
</p>

## Discriminator

<p align='center'>
    <img src="https://github.com/maxibove13/wakeGAN/blob/main/figures/reference/disc_arch.png" alt="discriminator" width="800"/>
</p>


## Results

### Error between real and synthetic flow field

![Image comparison test](https://github.com/maxibove13/wakeGAN/blob/main/figures/test/image_comparison_err_test.png)

### Real vs. generated Wind profiles at different streamwise positions relative to WT diameter

<p align='center'>
    <img src="https://github.com/maxibove13/wakeGAN/blob/main/figures/test/profiles.png" alt="wind profiles" width="500"/>
</p>

### Adversarial loss and RMSE evolution over epochs

<p align='center'>
    <img src="https://raw.githubusercontent.com/maxibove13/wakeGAN/main/figures/reference/metrics_ref.png" alt="loss & rmse" width="500"/>
</p>

## data

### raw

Contains the chaman LES simulation outputs of the WF for different precursors and turns.
Each simulation is composed of 18 regions.
The data is present through a symbolic link between the actual storage folder and `/data/raw` directory.

## Random comments on the approach

Generator takes the parameters $\mu$ as input and outputs the flow field prediction U as the output

Discriminator takes the data pair of the embedded flow parameter Z and the corresponding flow field $U$ or $U_{gen}$ (real or generated) as the input, and zeros or ones as outputs.

The main difference between CGAN and GAN is that the labels (here the flow parameters u) are combined with the corresponding flow field for the examination by the discriminator, while GAN only distinguishes the generated flow field from the real flow field without the labeling information.

The $\mu$ parameters are the input of the CFD simulations.
These parameters are collected in a input tensor X of shape [ $N$ , $N_{\mu}$ ]

input: [ $N$ , $X_3$, $C$ ] (they use profile along y axis)
output: [ $N$ , $X_1$ , $X_2$, $X_3$ , $C$ ] (flow field data, C is 2 - $U_x$ and $U_y$ -, $N$ samples)

The loss is just the common adversarial loss but for the Discriminator instead of x we use [ $U$ , $\mu$ ] and for the Generator we use [ $G(U)$ , $\mu$ ] 

# Versioning data with DVC

Let's track our splited data using [DVC](https://dvc.org/)

First initialize DVC, it behaves similar to git:

```
dvc init
```

Note: Check that the data you want to tracked isn't in `.gitignore`

In my case, I'm going to track only the splitted data (splitted between `train` and `test`) which lives in `data/preprocessed/tracked`

Then, let's add the data we are going to track to dvc staging area: 

```
dvc add data/preprocessed/tracked/
```

With this command, dvc creates a file (`*.dvc`) that contains metadata about your tracked data, let's git add it and ignore the folder that contains the tracked data

```
git add data/preprocessed/tracked.dvc data/preprocessed/.gitignore
```

We can keep track of our data with the actual data being storage in the cloud. We'll use google drive for this:

```
dvc remote add -d storage gdrive:<gdrive_folder_id> 
```

Note: `gdrive_folder_id` corresponds to the id that the URL shows when you are in the folder that you would like to store your tracked data.

This configuration lives in `.dvc/config` file

Now, let's push the data to our remote storage:

```
dvc push
```

If you make changes to the data, you can track them with

```
dvc add <path_to_tracked_data>
```

Then git add the changes on `*.dvc` file, and commit.

```
git add <path_to_tracked_data>/*.dvc
git commit -m 'updating data'
```

For example, you can recover the last data modification going back one commit

```
git checkout HEAD^1 <path_to_tracked_data>
```

And go back and forth with:

```
git stash
git checkout HEAD
dvc checkout
```

### branches to track datasets

Each branch represents a different dataset.
In order to have the changes in `main` in any feature (dataset) branch we need to `git rebase main` on each branch when we make changes in the code (only changes in `main` allowed).
However, we don't want to obtain the changes made to `tracked.dvc`, `figures/test/*` and `config.yaml`, we use `.gitattributes` for this.

In the root file `.gitattributes` we specify which file pattern we will exclude from the merging. In this case, we add the following to `.gitattributes`:

```
data/preprocessed/tracked.dvc merge=ours
figures/test/* merge=ours
config.yaml merge=ours
```

After that modify `.gitconfig`:

```
git config --global merge.ours.driver true
```

With this changes we can safely move between branches to keep a certain configuration, dataset and figures.
Remember that on each branch the we keep track of `tracked.dvc` not of the data itself which is store in `gdrive`.

Rebase tu upgrade the feature branch with main changes


A general steady-state parametrized fluid system can be described by:

```math
P[u] = 0, x ~ E ~ \Omega 
```
```math
B[u] = 0, x ~ E ~ \delta ~ \Omega
```

Where $u$ is the state of the system while the differential operator $P$ (parametrized by $\mu_p$) represent the PDEs describing the fluid systems, the boundary conditions and the flow domain respectively.

The flow parameters arising from governing equations, the domain geometry and the boundary conditions are denoted as $\mu$.
Given a specific value of $\mu$ the flow field in the domain \Omega, denoted as $U$, can be obtained by solving the above equation numerically.

Zhang & Zhao, develop a surrogate modeling method to approximate the mapping between \mu and U so that fast and accurate predictions of U can be achieved.

# Reproducibility

In order to make the results reproducible, a random seed has to be set at the beginning of the code:

```python
torch.manual_seed(42)
```

It is also recommended to force PyTorch to check that all operations are deterministic:

```python
torch.use_deterministic_algorithms(True)
```

If using a GPU it is necessary to set the `CUBLAS_WORKSPACE_CONFIG` environment variable to `:4096:8` or `:16:8` as suggested in the [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility) documentation.
