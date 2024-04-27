# Adaptive Basis Function Selection for Computationally Efficient Predictions
Contains code for reproducing the results in our paper **Adaptive Basis Function Selection for Computationally Efficient Predictions**.
Requires the fast-hgp submodule which contains a reference implementation of the selection for the HGP.

### Getting started
A Dockerfile is provided which greatly simplifies the installation process.
Make sure to install Docker which should guarantee a simple installation for you.

Clone the repo:
```
git clone --recurse-submodules git@github.com:AOKullberg/adaptive-bf-selection.git
```

Build the Docker image:
```
docker build -t adaptive-bf .
```

Run the Docker container:
```
docker-compose up
```

#### Running the timing experiments
The timing experiments are produced by the `run_timing.py` script.
It uses Hydra multirun to run all the different configurations.
Specifically, run it with
```
python run_timing.py -m
```
This will save the data in a directory in `multirun`.
The notebook `timing-results.ipynb` can now be used to generate mock-ups of Figure 2 of the paper and should be self-explanatory.

#### Generating conceptual plot
The notebook `conceptual_idea.ipynb` generates a mock-up of Figure 1, explaining the conceptual idea of the paper and validating the approach.