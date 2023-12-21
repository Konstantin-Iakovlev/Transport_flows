import json
import networkx as nx
import numpy as np
import pandas as pd
import xmltodict
from pathlib import Path

FLOAT = np.float32

def read_graph_csv(folder: Path) -> nx.DiGraph:
    node = pd.read_csv(f"{folder}/Node.csv", sep=",")
    topo = pd.read_csv(f"{folder}/topo.csv", sep=",")
    graph = nx.DiGraph()
    graph.add_nodes_from(node.index)
    edges = topo.loc[:, ["src", "dst"]].to_numpy()
    cost = topo["IGP cost"].astype(FLOAT).to_numpy()
    bandwidth = topo["cap"].astype(FLOAT).to_numpy()
    for edge, c, b in zip(edges, cost, bandwidth):
        graph.add_edge(edge[0], edge[1], cost=c, bandwidth=b)
    return graph


def read_traffic_mat_csv(filename: Path) -> np.ndarray:
    tunnel = pd.read_csv(filename, sep=",")
    traffic_mat = np.zeros((tunnel["src"].max() + 1, tunnel["dst"].max() + 1), dtype=FLOAT)
    traffic_mat[tunnel["src"], tunnel["dst"]] = tunnel["bandwidth"]
    return traffic_mat / 1e6


def read_graph_sndlib_xml(filename: Path) -> nx.Graph:
    with open(filename, "r") as file:
        graph_dct = xmltodict.parse(file.read())["network"]["networkStructure"]

    graph = nx.DiGraph()

    for node in graph_dct["nodes"]["node"]:
        graph.add_node(
            node["@id"], 
            x=FLOAT(node["coordinates"]["x"]), 
            y=FLOAT(node["coordinates"]["y"])
        )

    for edge in graph_dct["links"]["link"]:
        cost = FLOAT(edge.get("routingCost", 1.))
        if "preInstalledModule" in edge:
            bandwidth = FLOAT(edge["preInstalledModule"]["capacity"])
        elif "additionalModules" in edge and "addModule" in edge["additionalModules"]:
            module = edge["additionalModules"]["addModule"]
            if isinstance(module, list):
                module = module[0]
            bandwidth = FLOAT(module["capacity"])
        else:
            bandwidth = FLOAT(1.)
        graph.add_edge(edge["source"], edge["target"], cost=cost, bandwidth=bandwidth)
        graph.add_edge(edge["target"], edge["source"], cost=cost, bandwidth=bandwidth)

    return graph


def read_traffic_mat_sndlib_xml(filename) -> np.ndarray:
    with open(filename, "r") as file:
        xml_dct = xmltodict.parse(file.read())["network"]

    node_label_to_num = {node["@id"]: i for i, node in enumerate(xml_dct["networkStructure"]["nodes"]["node"])}
    traffic_mat = np.zeros((len(node_label_to_num), len(node_label_to_num)), dtype=FLOAT)
    for demand in xml_dct["demands"]["demand"]:
        source = node_label_to_num[demand["source"]]
        target = node_label_to_num[demand["target"]]
        traffic_mat[source, target] = demand["demandValue"]
    return traffic_mat


def read_metadata_networks_tntp(filename: Path) -> dict:
    with open(filename, "r") as file:
        zones = int(file.readline()[len("<NUMBER OF ZONES>"):].strip())
        nodes = int(file.readline()[len("<NUMBER OF NODES>"):].strip())
        can_pass_through_zones = (int(file.readline()[len("<FIRST THRU NODE>"):].strip()) == 1)
    return dict(zones=zones, nodes=nodes, can_pass_through_zones=can_pass_through_zones)


def read_graph_transport_networks_tntp(filename: Path) -> nx.DiGraph:
    # Made on the basis of
    # https://github.com/bstabler/TransportationNetworks/blob/master/_scripts/parsing%20networks%20in%20Python.ipynb
    
    metadata = read_metadata_networks_tntp(filename)

    net = pd.read_csv(filename, skiprows=8, sep='\t')
    net.columns = [col.strip().lower() for col in net.columns]
    net = net.loc[:, ["init_node", "term_node", "capacity", "free_flow_time"]]
    net.loc[:, ["init_node", "term_node"]] -= 1

    graph = nx.DiGraph()
    graph.add_nodes_from(
        range(metadata["nodes"] + (0 if metadata["can_pass_through_zones"] else metadata["zones"])))

    for row in net.iterrows():
        source = row[1].init_node
        dest = row[1].term_node
        if not metadata["can_pass_through_zones"] and dest < metadata["zones"]:
            dest += metadata["nodes"]
        graph.add_edge(
            source,
            dest,
            cost=FLOAT(row[1].free_flow_time),
            bandwidth=FLOAT(row[1].capacity)
        )
    
    return graph


def read_traffic_mat_transport_networks_tntp(filename: Path, metadata: dict) -> np.ndarray:
    # Made on the basis of
    # https://github.com/bstabler/TransportationNetworks/blob/master/_scripts/parsing%20networks%20in%20Python.ipynb
    
    with open(filename, "r") as file:
        blocks = file.read().split("Origin")[1:]
    matrix = {}
    for block in blocks:
        demand_data_for_origin = block.split("\n")
        orig = int(demand_data_for_origin[0])
        destinations = ";".join(demand_data_for_origin[1:]).split(";")
        matrix[orig] = {}
        for dest_str in destinations:
            if len(dest_str.strip()) == 0:
                continue
            dest, demand = dest_str.split(":")
            matrix[orig][int(dest)] = FLOAT(demand)

    zones = metadata["zones"]
    zone_demands = np.zeros((zones, zones))
    for i in range(zones):
        for j in range(zones):
            zone_demands[i, j] = matrix.get(i + 1, {}).get(j + 1,0)

    num_nodes = metadata["nodes"]
    num_nodes += (0 if metadata["can_pass_through_zones"] else metadata["zones"])
    traffic_mat = np.zeros((num_nodes, num_nodes), dtype=FLOAT)
    if metadata["can_pass_through_zones"]:
        traffic_mat[:zones, :zones] = zone_demands
    else:
        traffic_mat[:zones, -zones:] = zone_demands
    return traffic_mat


def update_node_coordinates(node_coords: dict, metadata: dict):
    if not metadata["can_pass_through_zones"]:
        for key in range(metadata["zones"]):
            node_coords[key + metadata["nodes"]] = node_coords[key].copy()


def read_node_coordinates_transport_networks_tntp(filename: Path, metadata: dict) -> dict:
    with open(filename, "r") as file:
        try:
            data = pd.read_csv(filename, delim_whitespace=True, header=0, names=["node", "x", "y", "semicolon"])
        except pd.errors.ParserError:
            data = pd.read_csv(filename, delim_whitespace=True, header=0, names=["node", "x", "y"])
    data = data.loc[:, ["x", "y"]]

    node_coords = {}
    for row in data.iterrows():
        node_coords[row[0]] = {"x": FLOAT(row[1].x), "y": FLOAT(row[1].y)}

    update_node_coordinates(node_coords, metadata)
    return node_coords


def read_node_coordinates_transport_networks_geojson(filename: Path, metadata: dict) -> dict:
    with open(filename, "r") as file:
        geodata = json.load(file)
    
    node_coords = {}
    for node, feature in enumerate(geodata["features"]):
        coords = feature["geometry"]["coordinates"]
        node_coords[node - 1] = {"x": FLOAT(coords[0]), "y": FLOAT(coords[1])}

    update_node_coordinates(node_coords, metadata)
    return node_coords
