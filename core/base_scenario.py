import json
from abc import ABCMeta, abstractmethod
from typing import Optional, Union, Tuple, List, Dict, Union
from core.infrastructure import Infrastructure, Link, DataFlow, Node, Location

__all__ = ["BaseScenario"]

class BaseScenario(metaclass=ABCMeta):
    """Base class for customized scenarios."""

    def __init__(self, config_file: str):
        """Initialize the scenario by loading the configuration file and setting up the infrastructure."""
        self.json_object = self.load_config(config_file)
        self.json_nodes, self.json_edges = self.json_object['Nodes'], self.json_object['Edges']
        
        self.base_latency_type = self.json_object.get('BaseLatencyType', 'None')
        
        # Initialize infrastructure and node mapping
        self.infrastructure = Infrastructure()
        self.node_id2name = {}
        self.node_name2id = {}

        # Signal speed and hop delay constants
        self.signal_speed = 2.0e5 - 1.5e5 # Signal speed in fiber (km/s) - Hop delay + Router delay
        self.base_energy_coef = 0.8/10000 # Energy coefficient for the link (J/mb/km)

        # Initialize infrastructure with nodes and links
        self.init_infrastructure_nodes()
        self.init_infrastructure_links()

    def load_config(self, config_file: str) -> dict:
        """Load the configuration file and return its content as a JSON object."""
        with open(config_file, 'r') as fr:
            return json.load(fr)

    def init_infrastructure_nodes(self):
        """Initialize nodes in the infrastructure."""
        for node_info in self.json_nodes:
            assert node_info['NodeType'] == 'Node', (
                f"Invalid NodeType {node_info['NodeType']}. "
                "Ensure it is 'Node' or override init_infrastructure_nodes()."
            )
            
            location = self.get_location(node_info)
            node = Node(
                node_id=node_info['NodeId'],
                name=node_info['NodeName'],
                max_cpu_freq=node_info['MaxCpuFreq'],
                max_buffer_size=node_info['MaxBufferSize'],
                location=location,
                idle_energy_coef=node_info['IdleEnergyCoef'],
                exe_energy_coef=node_info['ExeEnergyCoef']
            )

            self.infrastructure.add_node(node)
            self.node_id2name[node_info['NodeId']] = node_info['NodeName']
            self.node_name2id[node_info['NodeName']] = node_info['NodeId']

    def get_location(self, node_info: dict) -> Optional[Location]:
        """Return a Location object if coordinates are provided, otherwise None."""
        if 'LocX' in node_info and 'LocY' in node_info:
            return Location(node_info['LocX'], node_info['LocY'])
        return None

    def init_infrastructure_links(self):
        """Initialize links between nodes in the infrastructure."""
        nodes = self.infrastructure.get_nodes()

        for edge_info in self.json_edges:
            assert edge_info['EdgeType'] in ['Link', 'SingleLink'], (
                f"Invalid EdgeType {edge_info['EdgeType']}. "
                "Ensure it is either 'Link' or 'SingleLink'."
            )

            src_node_id, dst_node_id = edge_info['SrcNodeID'], edge_info['DstNodeID']
            base_latency = self.calculate_base_latency(edge_info, src_node_id, 
                                                       dst_node_id, nodes)
            
            
            energy_coef = self.calculate_energy_coef(edge_info, src_node_id,
                                                     dst_node_id, nodes)

            if edge_info['EdgeType'] == 'SingleLink':
                self.add_unilateral_link(
                    self.node_id2name[src_node_id], 
                    self.node_id2name[dst_node_id], 
                    edge_info['Bandwidth'], 
                    base_latency,
                    energy_coef
                )
            else:
                self.add_bilateral_links(
                    self.node_id2name[src_node_id], 
                    self.node_id2name[dst_node_id], 
                    edge_info['Bandwidth'], 
                    base_latency,
                    energy_coef
                )

    def calculate_base_latency(
        self, edge_info: dict, src_node_id: int, dst_node_id: int, nodes: dict
    ) -> float:
        """Calculate the base latency for the link, either from config or based on node distances."""
        if 'BaseLatency' in edge_info:
            return edge_info['BaseLatency']
        
        src_node = nodes[self.node_id2name[src_node_id]]
        dst_node = nodes[self.node_id2name[dst_node_id]]
        
        if self.base_latency_type == 'None':
            return 0

        if src_node.location and dst_node.location:
            distance = src_node.distance(dst_node, type=self.base_latency_type) * 2  # Round trip distance in meters

            return round(distance / self.signal_speed, 3)
        
        return 0
    def calculate_energy_coef(
        self, edge_info: dict, src_node_id: int, dst_node_id: int, nodes: dict
    ) -> float:
        """Calculate the base latency for the link, either from config or based on node distances."""
        if 'BaseLatency' in edge_info:
            return edge_info['EnergyCoef']
        
        src_node = nodes[self.node_id2name[src_node_id]]
        dst_node = nodes[self.node_id2name[dst_node_id]]
        
        if self.base_latency_type == 'None':
            return 0

        if src_node.location and dst_node.location:
            distance = src_node.distance(dst_node, type=self.base_latency_type) * 2  # Round trip distance in meters
            return round(distance * self.base_energy_coef, 3)
        
        return 0

    @abstractmethod
    def status(self, node_name: Optional[str] = None, link_args: Optional[Tuple] = None):
        """User-defined Scenario status."""
        nodes = self.get_nodes()
        links = self.get_links()
        return nodes, links

    def avg_node_energy(self, node_name_list: Optional[List[str]] = None) -> float:
        """Calculate the average energy consumption of specified nodes."""
        if not node_name_list:
            node_list = self.get_nodes().values()
        else:
            node_list = [self.get_node(node_name) for node_name in node_name_list]
        
        total_energy = sum(node.energy_consumption for node in node_list)
        return total_energy / len(node_list)

    def node_energy(self, node_name: str) -> float:
        """Return the energy consumption of a specific node."""
        return self.get_node(node_name).energy_consumption
    
    def node_power(self, node_name: Optional[str] = None) -> Union[float , dict]:
        """Return the power consumption of a specific node."""
        if node_name:
            return self.get_node(node_name).energy_consumption/self.get_node(node_name).clock
        else:
            return {node_name: node.energy_consumption/node.clock for node_name, node in self.get_nodes().items()}
        
    def avg_node_power(self, node_name_list: Optional[List[str]] = None) -> float:
        """Calculate the average power consumption of specified nodes."""
        if not node_name_list:
            node_list = self.get_nodes().values()
        else:
            node_list = [self.get_node(node_name) for node_name in node_name_list]
        
        total_power = sum(node.energy_consumption/node.clock for node in node_list)
        return total_power / len(node_list)
        

    def get_node(self, name: str) -> Node:
        """Return the node by its name."""
        return self.infrastructure.get_node(name)

    def get_link(self, src_name: str, dst_name: str, key=0) -> Link:
        """Return the link between two nodes."""
        return self.infrastructure.get_link(src_name, dst_name, key)

    def get_nodes(self):
        """Return all nodes in the infrastructure."""
        return self.infrastructure.get_nodes()

    def get_links(self):
        """Return all links in the infrastructure."""
        return self.infrastructure.get_links()

    def add_unilateral_link(self, src_name: str, dst_name: str, bandwidth: float, base_latency: float = 0, energy_coef: float = 0.0):
        """Add a unilateral link between two nodes."""
        self.infrastructure.add_link(
            Link(self.infrastructure.get_node(src_name), self.infrastructure.get_node(dst_name),
                 max_bandwidth=bandwidth, base_latency=base_latency, energy_coef=energy_coef)
        )

    def add_bilateral_links(
        self, src_name: str, dst_name: str, bandwidth: Union[float, List], base_latency: float = 0, energy_coef: float = 0.0
    ):
        """Add bilateral links between two nodes."""
        self.infrastructure.add_link(
            Link(self.infrastructure.get_node(src_name), self.infrastructure.get_node(dst_name),
                 max_bandwidth=bandwidth[0] if isinstance(bandwidth, List) else bandwidth, 
                 base_latency=base_latency, energy_coef=
                    energy_coef)
        )
        self.infrastructure.add_link(
            Link(self.infrastructure.get_node(dst_name), self.infrastructure.get_node(src_name),
                 max_bandwidth=bandwidth[1] if isinstance(bandwidth, List) else bandwidth, 
                 base_latency=base_latency, energy_coef=energy_coef)
                 
        )

    def reset(self):
        """Reset all nodes and links in the infrastructure."""
        for node in self.get_nodes().values():
            node.reset()
            if node.flag_only_wireless:
                node.update_access_dst_nodes(self.get_nodes())

        for link in self.get_links().values():
            link.reset()

    def send_data_flow(
        self, data_flow: DataFlow, links=None, src_name: str = None, dst_name: str = None, weight=None
    ):
        """Simulate a data flow in the infrastructure from src to dst."""
        if not links:
            links = self.infrastructure.get_shortest_links(src_name, dst_name, weight)
        data_flow.allocate(links)
