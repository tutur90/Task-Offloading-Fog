# Task Offloading in Fog Environment


##  Requirements & Installation

Main Dependent Modules:

- **python >= 3.8**: Previous versions might be OK but without testing.
- **networkx**: NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
- **simpy**: SimPy is a process-based discrete-event simulation framework based on standard Python.
- **numpy**: NumPy is a Python library used for working with arrays.
- **pandas**: Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.
  
The following modules are used for visualization tools:

- **matplotlib**
- **cv2**
- **tensorboard**

Users are recommended to use the Anaconda to configure the RayCloudSim:

```text
conda create --name raycloudsim python=3.8
conda activate raycloudsim
pip install -r requirements.txt
```


## Usage

To run the simulation, navigate to the `main` directory and execute the `main.py` script. You can modify the configuration parameters directly in the script or through a configuration file.

```bash
cd main
python main.py
```

## Configuration

The simulation can be configured through the `config` dictionary in the `main.py` file. Key parameters include:


