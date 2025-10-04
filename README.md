# Task Offloading in Fog Environment


##  Requirements & Installation

Main Dependent Modules:

- **python >= 3.8**: Previous versions might be OK but without testing.
- **networkx**: NetworkX is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
- **simpy**: SimPy is a process-based discrete-event simulation framework based on standard Python.
- **numpy**: NumPy is a Python library used for working with arrays.
- **pandas**: Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.
  

Users are recommended to use the Anaconda to configure the RayCloudSim:

```text
conda create --name raycloudsim python=3.12
conda activate raycloudsim
pip install -r requirements.txt
```


## Usage

To run the simulation, execute the `main.py` script. You can modify the configuration file path to use different scenarios and policies. For example, to run the simulation with the NOTE policy in the Pakistan scenario, use the following command:

```bash
python main.py --config configs/Pakistan/DQL/NOTE.yaml
```


## Policies
The framework supports various task offloading policies, including:
- **Random**: Tasks are offloaded to a randomly selected node.
- **Greedy**: Tasks are offloaded to the node with the most available resources.
- **Round Robin**: Tasks are offloaded in a round-robin fashion among available nodes
- **MLP DQL**: A Deep Q-Learning based policy using a Multi-Layer Perceptron.
- **MLP NPGA/NSGA-II**: A policy using Multi-Layer Perceptron with Non-dominated Sorting Genetic Algorithm.
- **NOTE**: Node Offloading Transformer-based Encoder policy using the DQL algorithm.
- **T-NOTE**: Task-aware Node Offloading Transformer-based Encoder policy using the DQL algorithm.


## Informations

This framework is based on the RayCloudSim project available at [RayCloudSim GitHub Repository](https://github.com/ZhangRui111/RayCloudSim). For more details on the simulation environment and additional functionalities please refer to the original repository.

This repository is developed and maintained by Arthur GARON as part of his research project. For any questions or contributions, please feel free to open an issue or submit a pull request.
