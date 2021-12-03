#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to decide what's the best order to visit pubs.

@author: Matt Bailey
"""

import argparse
import collections
import itertools
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from osm import read_osm
from pub_data import Pub, import_pubs
from pub_drawing import plot_highlighted_paths, plot_pub_map, plot_pub_voronoi
from pub_satisfier import (
    find_all_closest_nodes,
    find_closest_node,
    import_pubs,
    load_graph,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)


PTCL_LOCATION = (-1.2534844873736306, 51.7590386380762)


def find_shortest_crawl_path(nx_graph, pubs):
    closest_nodes = find_all_closest_nodes(
        pubs, nx.get_node_attributes(nx_graph, "pos")
    )
    node_to_name = {val: key for key, val in closest_nodes.items()}

    node_ids = list(closest_nodes.values())

    # Using networkx directly absolutely thrashes my PC, as the
    # underlying graph is way too big.
    # Instead, do some pre-processing ourselves.
    ip_graph = nx.Graph()
    for pub_i, pub_j in itertools.combinations(pubs, 2):

        shortest_distance: float = nx.shortest_path_length(
            nx_graph,
            source=closest_nodes[pub_i.name],
            target=closest_nodes[pub_j.name],
            weight="length",
        )
        logging.info(
            f"Shortest distance from {pub_i} to {pub_j} is {shortest_distance}"
        )
        ip_graph.add_edge(pub_i.name, pub_j.name, length=shortest_distance)

    closest_nodes["PTCL"] = find_closest_node(
        PTCL_LOCATION, nx.get_node_attributes(nx_graph, "pos")
    )
    for pub in pubs:
        dist_to_ptcl = nx.shortest_path_length(
            nx_graph,
            source=closest_nodes[pub.name],
            target=closest_nodes["PTCL"],
            weight="length",
        )
        ip_graph.add_edge(pub.name, "PTCL", length=dist_to_ptcl)
    path = nx.approximation.traveling_salesman_problem(
        ip_graph, weight="length", cycle=True
    )

    total_path = []
    for i in range(len(path) - 1):
        node_i, node_j = closest_nodes[path[i]], closest_nodes[path[i + 1]]
        between_path = nx.shortest_path(
            nx_graph, source=node_i, target=node_j, weight="length"
        )
        total_path.append(between_path)

    name_path = collections.deque(path)
    ptcl_index = name_path.index("PTCL")
    name_path.rotate(-ptcl_index)
    return total_path, name_path


def main():

    parser = argparse.ArgumentParser(description="Decide which pub to go to.")

    parser.add_argument(
        "--temperature",
        default=1.0,
        type=float,
        help="Monte Carlo temperature for deciding on which pub. Higher temperature means less optimal choices.",
    )

    parser.add_argument(
        "--datafile",
        default="./oxford-pub-data.csv",
        type=str,
        help="Name of the file containing the pub data",
    )

    parser.add_argument(
        "--mapfile",
        default="oxford.osm",
        type=str,
        help="Name of the OpenStreetMap file",
    )

    def LoggingTypeVerifier(arg: str):
        _logging_types = {"CRITICAL", "ERROR", "WARNING", "NOTSET", "INFO", "DEBUG"}
        arg = str(arg).upper()
        if arg in _logging_types:
            return arg
        raise argparse.ArgumentTypeError(f"{arg} is not a valid logging type.")

    def list_str(values):
        return values.split(",")

    parser.add_argument(
        "pubs",
        help="Which pubs are we going to?",
        nargs="*",
        default=[
            "The Turf Tavern",
            "The Chequers",
            "The Kings Arms",
            "White Horse",
            "The Royal Oak",
            "The Bear Inn",
        ],
    )

    parser.add_argument(
        "--logging",
        default="NOTSET",
        type=LoggingTypeVerifier,
        help="Level of logging messages to show",
    )

    args = parser.parse_args()
    logger.setLevel(args.logging.upper())
    print(args.pubs)
    nx_graph = load_graph(args.mapfile)

    positions = {
        node: np.array([nx_graph.nodes[node]["lon"], nx_graph.nodes[node]["lat"]])
        for node in nx_graph.nodes
    }
    nx.set_node_attributes(nx_graph, positions, name="pos")
    pubs = [pub for pub in import_pubs(args.datafile) if pub.is_open]
    pub_names = set(pub.name for pub in pubs)

    valid_crawl_pubs = []
    for crawl_pub_name in args.pubs:
        if crawl_pub_name not in pub_names:
            logger.warning(f"{crawl_pub_name} is not in the data file.")
        else:
            matching_pub = [pub for pub in pubs if pub.name == crawl_pub_name][0]
            valid_crawl_pubs.append(matching_pub)

    logger.info(
        "Creating an intra-pub graph of "
        + ",".join(pub.name for pub in valid_crawl_pubs)
    )

    shortest_path, name_path = find_shortest_crawl_path(nx_graph, valid_crawl_pubs)
    paths_highlight = [
        [(sublist[node], sublist[node + 1]) for node in range(len(sublist) - 1)]
        for sublist in shortest_path
    ]
    _, ax = plt.subplots()
    ax = plot_pub_map(nx_graph, valid_crawl_pubs, with_labels=True, ax=ax)
    # ax = plot_pub_voronoi(pubs, ax=ax, with_labels=True)
    ax = plot_highlighted_paths(nx_graph, paths_highlight, ax=ax, styles="dashed")
    plt.show()
    print("The shortest path is", name_path)


if __name__ == "__main__":
    main()
