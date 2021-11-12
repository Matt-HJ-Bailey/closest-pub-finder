#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to decide which pub to go to in Oxford.

@author: Matt Bailey
"""

import argparse
import logging
import os
import sys
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from osm import read_osm
from pub_data import Pub, import_pubs
from pub_drawing import plot_highlighted_paths, plot_pub_map, plot_pub_voronoi

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.INFO)

STANDARD_LOCATION = (-1.2664722821594505,51.760319078103194)
PTCL_LOCATION = (-1.2534844873736306, 51.7590386380762)
PERSON_VETOS = defaultdict(set)
PERSON_VETOS["Max"] = {
    "The Lamb and Flag",
}
PERSON_VETOS["Matt"] = {"The Half Moon", "The Black Swan", "The Wig and Pen"}
PERSON_VETOS["Tristan"] = {"The Kings Arms"}

Coordinate = Tuple[float, float]
_closest_node_cache: Dict[Coordinate, int] = {}

_CACHED_HELPER = None


class ClosestHelper:
    """Helper class that automates some of the KD-tree bookkeeping."""

    def __init__(self, positions: Dict[int, Coordinate]):
        """
        Set up the KD-tree for future usage.

        Parameters
        ----------
        positions
            Positions of every node in the graph, as a dict.
        """
        positions_arr = np.empty([len(positions), 2])
        self._idx_label = {}
        for idx, (label, pos) in enumerate(positions.items()):
            self._idx_label[idx] = label
            positions_arr[idx, :] = pos

        self._tree = KDTree(positions_arr)

    def query(
        self, coordinates: Union[Coordinate, Iterable[Coordinate]]
    ) -> Union[int, List[int]]:
        """
        Find the node index closest to these coordinates.

        Returns just one node index for a single coordinate, or an array
        for an array for coordinates.

        Parameters
        ----------
        coordinates
            Either one or many lat, lon coordinates

        Returns
        -------
            labels of the closest nodes in the graph
        """
        _, neighbours = self._tree.query(coordinates)
        if isinstance(neighbours, np.int64):
            # We've just got one
            return self._idx_label[neighbours]
        return [self._idx_label[neighbour] for neighbour in neighbours]


def find_closest_node(current_pos: Coordinate, positions: Dict[int, Coordinate]) -> int:
    """
    Find the closest node to the specified lat, lon pair.

    This is cached and will not take a change of positions into account
    Parameters
    ----------
    current_pos
        a (lat, lon) pair.
    positions
        A dict keyed by node id containing node lat, lon pairs

    Returns
    -------
        a node id of the closest graph node to this position
    """
    global _CACHED_HELPER
    if _CACHED_HELPER is None:
        _CACHED_HELPER = ClosestHelper(positions)

    kdtree_helper = _CACHED_HELPER
    neighbour = kdtree_helper.query(current_pos)
    return neighbour


def load_graph(filename: str) -> nx.Graph:
    """
    Try loading the graph, either from the gpickle or the file.

    If the pickle exists, read the graph from that.
    If not, use read_osm to read it.
    Trim out unconnected components and slightly tidy up the graph before returning.
    Parameters
    ----------
    filename
        The name of the file to read from.
    """

    if os.path.exists(filename + ".pkl"):
        nx_graph = nx.read_gpickle(filename + ".pkl")
        logger.info("Loading graph from pickle")
    else:
        nx_graph = read_osm(filename)
        nx.write_gpickle(nx_graph, filename + ".pkl")

    largest_component = list(nx.connected_components(nx_graph))[0]
    nx_graph = nx_graph.subgraph(largest_component).copy()
    return nx_graph


def find_all_closest_nodes(
    pubs: List[Pub], positions: Dict[int, Tuple[float, float]], regenerate: bool = False
) -> Dict[str, int]:
    """
    For each pub, find the closest node in the graph to it.

    Parameters
    ----------
    pubs
        a list of pubs that have coordinates
    positions
        A dict keyed by node id containing node lat, lon pairs

    Returns
    -------
        A dict mapping pub names to the closest graph node
    """
    closest_nodes = {}
    global _CACHED_HELPER
    if regenerate or _CACHED_HELPER is None:
        _CACHED_HELPER = ClosestHelper(positions)
    kdtree_helper = _CACHED_HELPER
    pub_positions = np.array([pub.coordinates for pub in pubs])
    neighbours = kdtree_helper.query(pub_positions)

    for idx, pub in enumerate(pubs):
        closest_nodes[pub.name] = neighbours[idx]
    return closest_nodes


def find_satisfied_constraints(
    pubs: List[Pub], constraints: Tuple[str, Callable]
) -> List[Pub]:
    """
    Find which pub satisfy a list of constraints.

    Provide the constraints as a list of (attr, predicate) tuples
    where the attribute is something we can look up in a pub
    as getattr(pub, attr), and the predicate operates on that.

    Parameters
    ----------
    pubs
        A list of pubs with various attributes
    constraints
        A list of (attribute, callable) pairs, where each pub must have each attribute and callable operates on that.

    Returns
    -------
        Pubs where all constraints are met.
    """
    total_matches_arr = np.ones(len(pubs), dtype=bool)
    for attr, pred in constraints:
        matches_arr = np.ones(len(pubs), dtype=bool)
        for idx, pub in enumerate(pubs):
            matches_arr[idx] = pred(getattr(pub, attr))
        print(f"{np.sum(matches_arr)} pubs meet constraint {attr}.")
        total_matches_arr = np.logical_and(matches_arr, total_matches_arr)

    return [pubs[idx] for idx in range(len(pubs)) if total_matches_arr[idx]]


def populate_distances(
    pubs: List[Pub],
    pubgoers: Iterable[str],
    person_locations: Dict[str, Tuple[float, float]],
    home_locations: Dict[str, Tuple[float, float]],
    graph: nx.Graph,
) -> List[Pub]:
    """
    Populate the distance attribute of the provided pubs.

    Parameters
    ----------
    pubs
        An iterable of pubs with attributes 'name' and 'distance'. Will be mutated.
    pubgoers
        A list of people going to the pub
    person_locations
        And a defaultdict of where to find them
    graph
        A road graph with edges that have attribute 'length'

    Returns
    -------
        mutated list of pubs
    """
    positions = nx.get_node_attributes(graph, "pos")
    closest_nodes = find_all_closest_nodes(pubs, positions)
    _distance_cache: Dict[Tuple[str, Coordinate], float] = {}
    for pub in pubs:
        distance = 0.0
        for pubgoer in pubgoers:

            for loc in (person_locations[pubgoer], home_locations[pubgoer]):
                if (pub.name, loc) in _distance_cache:
                    shortest_distance = _distance_cache[(pub.name, loc)]
                    distance += shortest_distance
                    logger.info(
                        f"shortest path from {pub.name} to {pubgoer} is {shortest_distance}"
                    )
                else:
                    # Fall back to finding the actual path
                    source_node = find_closest_node(loc, positions)
                    try:
                        shortest_distance: float = nx.shortest_path_length(
                            graph,
                            source=source_node,
                            target=closest_nodes[pub.name],
                            weight="length",
                        )
                        _distance_cache[(pub.name, loc)] = shortest_distance
                        distance += shortest_distance
                        logger.info(
                            f"Uncached shortest path from {pub.name} to {pubgoer} is {shortest_distance}"
                        )
                    except nx.NetworkXNoPath:
                        logger.warning(f"Could not find a path between {pub.name} and {pubgoer}. Marking it as closed.")
                        pub.is_open = False
                
        pub.distance = distance
    return pubs


def argsort_preserve_ties(values: np.ndarray) -> np.ndarray:
    """
    Provide indices that would sort this array, with ties having the same index.

    For example, argsort_preserve_ties([3, 3, 2, 2, 1, 1]) would give
    [2, 2, 1, 1, 0, 0]

    Parameters
    ----------
    values
        The array to sort
    Returns
    -------
        indices into sorted array
    """
    values = np.asarray(values)
    uniques = np.unique(values)
    new_order = np.empty(values.shape, dtype=int)
    for idx, value in enumerate(values):
        pos = np.where(uniques == value)
        new_order[idx] = pos[0]
    return new_order


def main():
    parser = argparse.ArgumentParser(description="Decide which pub to go to.")
    parser.add_argument(
        "pubgoers",
        help="Who is coming to the pub?",
        nargs="*",
        default=["Matt", "Max", "Martin"],
    )
    parser.add_argument(
        "--martin",
        action="store_true",
        help="Is Martin paying? If so, reverse the price ordering",
    )
    parser.add_argument(
        "--distance", default=1.0, type=float, help="Maximum average distance to travel"
    )
    parser.add_argument(
        "--print", default=5, type=int, help="How many pubs to print in the table"
    )
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
        help="Name of the file containing the pub data"
    )
    
    parser.add_argument(
        "--mapfile",
        default="oxford.osm",
        type=str,
        help="Name of the OpenStreetMap file"
    )
    
    def LoggingTypeVerifier(arg: str):
        _logging_types = {"CRITICAL", "ERROR", "WARNING", "NOTSET", "INFO", "DEBUG"}
        arg = str(arg).upper()
        if arg in _logging_types:
            return arg
        raise argparse.ArgumentTypeError(f"{arg} is not a valid logging type.")
    parser.add_argument(
        "--logging",
        default="NOTSET",
        type=LoggingTypeVerifier,
        help="Level of logging messages to show"
    )
    
    
    args = parser.parse_args()
    logger.setLevel(args.logging.upper())

    temperature = args.temperature
    pubgoers = args.pubgoers

    person_locations = defaultdict(lambda: PTCL_LOCATION)
    home_locations = defaultdict(lambda: STANDARD_LOCATION)

    if os.path.exists("./locations.csv"):
        person_df = pd.read_csv("./locations.csv")
        for idx, row in person_df.iterrows():
            person_locations[row["name"]] = (row["lon"], row["lat"])

    logger.info("Loading OpenStreetMap graph...")
    nx_graph = load_graph(args.mapfile)

    positions = {
        node: np.array([nx_graph.nodes[node]["lon"], nx_graph.nodes[node]["lat"]])
        for node in nx_graph.nodes
    }
    nx.set_node_attributes(nx_graph, positions, name="pos")
 
    pubs = [pub for pub in import_pubs(args.datafile) if pub.is_open]
    logger.info("Calculating distances...")
    pubs = populate_distances(pubs, pubgoers, person_locations, home_locations, nx_graph)

    pub_vetos = set(name.lower().strip() for pubgoer in pubgoers for name in PERSON_VETOS[pubgoer])
    print("Veto'd pubs are", pub_vetos)
    logger.info("Satisfying constraints...")
    valid_pubs = find_satisfied_constraints(
        pubs,
        [
            ("has_beer", lambda has_beer: has_beer),
            ("is_spoons", lambda is_spoons: not is_spoons),
            ("distance", lambda distance: distance < args.distance * len(pubgoers)),
            (
                "cheapest_pint",
                lambda cheapest_pint: np.isinf(cheapest_pint) or cheapest_pint < 5.0,
            ),
            ("has_pub_quiz", lambda has_pub_quiz: not has_pub_quiz),
            ("has_live_music", lambda has_live_music: not has_live_music),
            ("name", lambda name: name.lower().strip() not in pub_vetos),
            ("has_funny_smell", lambda has_funny_smell: not has_funny_smell),
            ("is_college", lambda is_college: not is_college),
        ],
    )

    if len(valid_pubs) > 1:
        print(f"We are left with {len(valid_pubs)} pubs. These are:")
        print(
            ", ".join(pub.name for pub in sorted(valid_pubs, key=lambda pub: pub.name))
        )
    elif len(valid_pubs) == 1:
        print(f"We are left with one pubs. It is {valid_pubs[0]}")
        sys.exit()
    else:
        print("We are left with zero valid pubs. We are having tinnies in the office.")
        sys.exit()

    if args.martin:
        price_weights = argsort_preserve_ties(
            [-pub.cheapest_pint for pub in valid_pubs]
        )
    else:
        price_weights = argsort_preserve_ties([pub.cheapest_pint for pub in valid_pubs])

    distance_weights = argsort_preserve_ties([pub.distance for pub in valid_pubs])

    weights = np.exp(-(distance_weights + price_weights) / temperature)
    weights = weights / np.sum(weights)
    weight_order = np.argsort(-weights)
    max_name_len = max(len(valid_pubs[pub_id].name) for pub_id in weight_order[: min(args.print, len(valid_pubs))]) + 1
    
    print("-" * (max_name_len) + "-⊤------⊤------⊤------⊣")
    print("Name" + " " * (max_name_len-4)+" | Dist.| Pint |Chance|")
    print("-" * (max_name_len) + "-+------+------+------⊣")

    for pub_id in weight_order[: min(args.print, len(valid_pubs))]:
        print(
            f"{valid_pubs[pub_id].name:{max_name_len}} | {valid_pubs[pub_id].distance/len(pubgoers):4.1f} | {valid_pubs[pub_id].cheapest_pint:4.2f} | {weights[pub_id]*100:4.1f}%|"
        )
    print("-" * (max_name_len) + "-⊥------⊥------⊥------⊣")

    rng = np.random.default_rng()
    random_idx = rng.choice(list(range(len(valid_pubs))), p=weights)
    random_pub = valid_pubs[random_idx]
    print(
        f"I have randomly chosen {random_pub.name} with probability {weights[random_idx]*100:.1f}%"
    )

    logger.info("Plotting shortest paths...")
    paths_highlight = []
    for pubgoer in pubgoers:
        person_node = find_closest_node(person_locations[pubgoer], positions)
        pub_node = find_closest_node(random_pub.coordinates, positions)
        choice_path = nx.shortest_path(
            nx_graph, source=person_node, target=pub_node, weight="length"
        )
        
        home_node = find_closest_node(home_locations[pubgoer], positions)
        home_path = nx.shortest_path(
            nx_graph, source=home_node, target=pub_node, weight="length"
        )
        
        paths_highlight.append(
            [
                (choice_path[node], choice_path[node + 1])
                for node in range(len(choice_path) - 1)
            ]
        )
        paths_highlight.append(
            [
                (home_path[node], home_path[node + 1])
                for node in range(len(home_path) - 1)
            ]
        )
    logger.info("Visualising pub map...")
    _, ax = plt.subplots()
    ax = plot_pub_map(nx_graph, pubs, ax=ax)
    ax = plot_pub_voronoi(pubs, ax=ax, with_labels=True)
    ax = plot_highlighted_paths(nx_graph, paths_highlight, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
