# Wind farm wake modelling using cDCGAN

Based on the work by Zhang J, Zhao X, "Wind farm wake modeling based on deep convolutional
conditional generative adversarial network", Energy, [https://doi.org/10.1016/j.energy.2021.121747](https://doi.org/10.1016/j.energy.2021.121747)

# Repo usage

## Generate dataset:

```
./src/data/make_dataset.py
```

## Split data between training and testing:

```
./src/data/split_data.py --ratio 0.9 0.1
```

## Train the cDCGAN:

```
./src/train.py
```


# Workflow:

1) CFD simulations of a WF
2) Horizontal slices at hub's height of mean horizontal velocity (Ux, Uy)
3) Crop slices into several images around each WT of the WF.
4) Save them as image files mapped with a certaing vmin and vmax. (vmin, vmax) -> (0, 255)
5) Read them, convert them to float32, rescale them to (0, 1)
6) Extract first column of pixels on each channel (inflow velocity)
7) Transform to tensor
8) For each fold:
    For each epoch:
        For each minibatch:
            - Generate fake image given inflow
            - Pass real, fake and inflows to discriminator
            - Evaluate loss, backprop on Disc and Gen



### Flowchart of the proposed surrogated model

![flowchart](https://github.com/maxibove13/ZZ_DC_CGAN/blob/main/figures/flowchart?raw=true)

A general steady-state parametrized fluid system can be described by:

P[u] = 0, x E \Omega
B[u] = 0, x E \delta \Omega

Where u is the state of the system while the differential operator P (parametrized by \mu_p) represent the PDEs describing the fluid systems, the boundary conditions and the flow domain respectively.

The flow parameters arising from governing equations, the domain geometry and the boundary conditions are denoted as \mu.
Given a specific value of \mu the flow field in the domain \Omega, denoted as U, can be obtained by solving the above equation numerically.

Zhang & Zhao, develop a surrogate modeling method to approximate the mapping between \mu and U so that fast and accurate predictions of U can be achieved.

## data

### raw

Contains the chaman LES simulation outputs of the WF for different precursors and turns.
Each simulation is composed of 18 regions.
The data is present through a symbolic link between the actual storage folder and `/data/raw` directory.

## Comments

Generator takes the parameters \mu as input and outputs the flow field prediction U as the output

Discriminator takes the data pair of the embedded flow parameter Z and the corresponding flow field U or U_gen (real or generated) as the input, and zeros or ones as outputs.

The main difference between CGAN and GAN is that the labels (here the flow parameters u) are combined with the corresponding flow field for the examination by the discriminator, while GAN only distinguishes the generated flow field from the real flow field without the labeling information.

The \mu parameters are the input of the CFD simulations.
These parameters are collected in a input tensor X of shape [N, N_mu]

input: [N, X3, C] (they use profile along y axis)
output: [N,X1,X2,X3,C] (flow field data, C is 2 -Ux and Uy-, N samples)

The loss is just the common adversarial loss but for the Discriminator instead of x we use [U, \mu] and for the Generator we use [G(U), \mu] 

They use Adam optimizer

### Case study of ZZ paper

Case of 3 turbines operating in a row, and the 2D velocity field around each turbine at the turbine hub height is extracted.

3 groups of different freestream mean wind speeds (8, 9, 10 m/s)

For each group 30 simulations varying the turbine yaw angles.
So for each inflow wind speed group we have 90 simulations, making a total of 270 training samples (each group has 3 turbines) 

\mu.shape = [33], 32 wind speed points and 1 yaw angle.
U.shape = [32, 32, 2] (32x32 uniform grid points and two channels)

75% training
25% testing

The inflow wind profiles (\mu) are the ones at the start of each subdomain containing each turbine.

## Issues

## Questions