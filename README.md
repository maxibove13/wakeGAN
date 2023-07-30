# Wind farm wake modeling using cDCGAN

## Predict the velocity field around a wind turbine given an inflow wind profile.

Based on the work by Zhang J, Zhao X, "Wind farm wake modeling based on deep convolutional
conditional generative adversarial network", Energy, [https://doi.org/10.1016/j.energy.2021.121747](https://doi.org/10.1016/j.energy.2021.121747)

<details>
<summary >Limitations:</summary>

- Only trained with streamwise velocity component (Ux) for now
- It generates only horizontal planes at hub's height (90m)
</details>


![Image comparison test](https://github.com/maxibove13/wakeGAN/blob/main/figures/reference/images_test_ref.png)

# Installation

1. Install packages.
    
    <details>
    intended for a GPU NVIDIA RTX 3070, check the pytorch specific installation for your specific GPU. You may need to change this line in the requirements.txt: 
    `--extra-index-url https://download.pytorch.org/whl/cu116`


```
pip install -r requirements.txt
```

2. Make sure you have `dvc` remote data credentials `.json` in the root directory of the repository.

3. Enable use of gdrive service account:

    ```
    dvc remote modify storage gdrive_use_service_account true
    ```

4. add gdrive credentials to dvc remote:

    ```
    dvc remote modify storage --local gdrive_service_account_json_file_path <credentials_file_name>.json
    ```

5. pull data from gdrive (through dvc which keeps track of the changes)

```
dvc pull
```

6. There are two main branches:

    1. `main` (t_window_1000)
    2. `t_window_4000`

`main` uses data from a temporal window of 1000 steps, and `t_window_4000` of 4000 steps.

5. You can `git checkout` and `dvc pull` (one after the other) in order to train/test/eval one or the other dataset.

# Usage

## data

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


## train the wakeGAN:

```
./scripts/train.py
```

- You can modify hyperparameters in `config.yaml` file
- You can monitor the training by watching the output in the shell, checking the logs in `logs/train.log` and checking the `figures/monitor` directory. This folder contains three figures that update every epoch:
    - `images.png`: watch real vs. synth generated images for both training and testing set
    - `metrics.png`: track RMSE and FID metrics.
    - `losses.png`: track generator and discriminator loss.  
    - `pixel_diff.png`: watch error (just the difference in m/s) between real and synth images for training and testing set.

## test the wakeGAN:

```
./scripts/test.py
```

Check generated files in `figures/test`:
- `images.png`: flow field comparison between real and synth images for four random samples.
- `pixel_diff.png`: error for those four random samples.
- `profiles.png`: wind profiles at different streamwise position in relation to the wind turbine for both the ground truth and the generated flow.

## data versions

We use different branches to track different versions of the dataset. We could also track the versions with `git commit`, but as these versions aren't an "improvement" of the previous version but just a different way of preprocessing the raw dataset we mantain the versions in branches.

Currently we are using two types of preprocessed dataset with velocity means taken at different average windows:

1. temporal window of 1000 time steps - branch: `main`
2. temporal window of 4000 time steps - branch: `t_window_4000`

## change between data versions

In order to use a different version first check that your git working directory is clean, and then checkout to your target branch:

```
git checkout t_window_4000
```

tell dvc to checkout to this branch:


```
dvc checkout
```

Now you should see the oter version of the dataset.

You can go back to the previous state with the same commands:

```
git checkout main
dvc checkout
```

# data pipeline overview:

1) CFD simulations of a WF
2) Horizontal slices at hub's height of mean horizontal velocity ($U_x$)
3) Crop slices into several images around each WT of the WF.
4) Save them as image files mapped with a certain $v_{min}$ and $v_{max}$. 
    
    ( $v_{min}$ , $v_{max}$ ) -> ( $0$ , $255$ )

5) Load the images, convert them to `float32` and rescale them to [ $-1$ , $1$ ]
6) Extract first column of pixels for each image (inflow velocity).
7) Training loop:
    
        for each epoch:
            for each minibatch:
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

### error between real and synthetic flow field

![Image comparison test](https://github.com/maxibove13/wakeGAN/blob/main/figures/reference/pixel_diff_test_ref.png)

### real vs. generated wind profiles at different streamwise positions relative to WT diameter

<p align='center'>
    <img src="https://github.com/maxibove13/wakeGAN/blob/main/figures/reference/profiles_test_ref.png" alt="wind profiles" width="500"/>
</p>

### complete wf real and synthetic flow

<p align='center'>
    <img src="https://github.com/maxibove13/wakeGAN/blob/main/figures/reference/wf_flow_5.76_-7.5_0.png" alt="wf flow" width="500"/>
</p>



## data

### raw

Contains the chaman LES simulation outputs of the WF for different precursors and turns.
Each simulation is composed of 18 regions.

# Versioning data with DVC (only for non-tracked data)

Let's track our splited data using [DVC](https://dvc.org/)

First initialize DVC, it behaves similar to git:

```
dvc init
```

Note: Check that the data you want to tracked isn't in `.gitignore`

We're going to track the splitted data (splitted between `train`, `val` `test`) which lives in `data/preprocessed/tracked`

Let's add the data we are going to track to dvc staging area: 

```
dvc add data/preprocessed/tracked/
```

With this command, dvc creates a file (`*.dvc`) in the tracked folder that contains metadata about your tracked data, let's git add it and ignore the folder that contains the tracked data

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


# Reproducibility (for plain pytorch)

In order to make the results reproducible, a random seed has to be set at the beginning of the code:

```python
torch.manual_seed(42)
```

It is also recommended to force PyTorch to check that all operations are deterministic:

```python
torch.use_deterministic_algorithms(True)
```

If using a GPU it is necessary to set the `CUBLAS_WORKSPACE_CONFIG` environment variable to `:4096:8` or `:16:8` as suggested in the [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility) documentation.
