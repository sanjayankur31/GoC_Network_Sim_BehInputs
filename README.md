# NeuroML2

This folder contains the NeuroML model code.

## Requirements

All supported versions of Python should be fine.
Please see the [GitHub actions configuration](https://github.com/sanjayankur31/GoC_Network_Sim_BehInputs/blob/development/.github/workflows/omv-ci.yml) in the repository for the current version that it is being tested against:

The required Python packages can be installed using the `requirements.txt` file:

```
pip install -r ./requirements.txt
```

### Mechanisms
This contains NeuroML or LEMS descriptions of ion channels, synaptic conductances and input spike trains/generators.

### Cells
This contains NeuroML descriptions of single Golgi cells, with different ion channel densities, constrained to match spontaneous firing rates between 2-9Hz, and input/output firing rates to be 14-25Hz/nA.

### Python Utils
This contains general Python scripts for generating networks and connectivity (`network_utils.py`) and input trains.

### Network_XXX
There are separate folders for generating networks with different input structures.
To generate the simulation scripts, either execute `generate_all.py` provided in each *Network_XX* folder, or follow the syntax to generate a single network model using `create_GoC_network` in `generate_beh_network_main.py`.
This will generate all necessary files:

- channel description as .mod files (need to be compiled using `nrnivmodl`)
- GoC descriptions (morphology and cellular mechanism) as .hoc files
- Network descriptors as .nml files
- Simulation description as LEMS files
- Simulation scripts as .py files (which use `neuron` python library for simulation)

To run the relevant simulation, run `LEMS_XXX_nrn.py` files.
