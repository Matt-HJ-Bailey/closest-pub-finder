import sys

import pickle as pkl

from collections import defaultdict

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from geopy.distance import geodesic
from scipy.spatial import Voronoi
from typing import Tuple, List, Callable

from osm import read_osm
from pub_data import PUBS


PTCL_LOCATION = (-1.2535525, 51.7593620)
GLOUCESTER_GREEN = (-1.26248, 51.75397)
PERSON_VETOS = defaultdict(set)
PERSON_VETOS["Max"] = {
    "The Lamb and Flag",
}
PERSON_VETOS["Matt"] = {"The Half Moon", "The Black Swan"}

DISTANCE_CUTOFF = 2.0

_closest_node_cache = {}


def find_closest_node(current_pos, positions):
    current_distance = np.inf
    node_index = None
    if current_pos in _closest_node_cache:
        return _closest_node_cache[current_pos]
    for key, val in positions.items():
        distance = geodesic(current_pos, val)
        if distance < current_distance:
            current_distance = distance
            node_index = key
    _closest_node_cache[current_pos] = node_index
    return node_index


def load_graph(filename):
    try:
        nx_graph = nx.read_gpickle(filename)
        print("Loading graph from pickle")
    except FileNotFoundError:
        nx_graph = read_osm("planet_-1.275,51.745_-1.234,51.762.osm")
        nx.write_gpickle(nx_graph, filename)
    return nx_graph


def find_all_closest_nodes(pubs, positions, regenerate=False):
    closest_nodes = dict()
    try:
        if not regenerate:
            print("Loading closest nodes from pickle")
            return pkl.load(open("all-closest-nodes.pkl", "rb"))

    except FileNotFoundError:
        print("Could not find all-closest-nodes.pkl")
    except TypeError as e:
        print(e)

    for pub in pubs:
        print("Finding closest", pub.name)
        closest_node = find_closest_node(pub.coordinates, positions)
        closest_nodes[pub.name] = closest_node
    pkl.dump(closest_nodes, open("all-closest-nodes.pkl", "wb"))
    return closest_nodes


def plot_pub_map(graph, highlight_edges, pubs, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    positions = nx.get_node_attributes(graph, "pos")
    # voronoi_plot_2d(vor, ax=AX, show_points=True, show_vertices=False, line_colors="red")
    nx.draw_networkx_edges(graph, pos=positions, ax=ax)

    min_x = min(pos[0] for pos in positions.values())
    max_x = max(pos[0] for pos in positions.values())
    min_y = min(pos[1] for pos in positions.values())
    max_y = max(pos[1] for pos in positions.values())

    nx.draw_networkx_edges(
        graph,
        pos=nx.get_node_attributes(graph, "pos"),
        ax=ax,
        edgelist=highlight_edges,
        edge_color="red",
        width=2.0,
    )
    ax.scatter(
        [pub.coordinates[0] for pub in pubs],
        [pub.coordinates[1] for pub in pubs],
        s=25,
        c="blue",
    )
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.axis("off")

    return ax


def plot_pub_voronoi(pubs, ax=None, with_labels=True):
    if ax is None:
        _, ax = plt.subplots()

    points = np.array([pub.coordinates for pub in pubs])

    points = np.append(
        points, [[999, 999], [-999, 999], [999, -999], [-999, -999]], axis=0
    )
    vor = Voronoi(points)

    dual_graph = nx.Graph()
    for pub_id, point_region in enumerate(vor.point_region):
        vertices_a = set(vor.regions[point_region])
        for other_pub_id, other_point_region in enumerate(vor.point_region):
            vertices_b = set(vor.regions[other_point_region])
            overlap = vertices_a.intersection(vertices_b)
            if len(overlap) == 2:
                dual_graph.add_edge(pub_id, other_pub_id)
    coloration = nx.greedy_color(dual_graph, interchange=True)

    print(coloration)
    polys = []
    colors = []
    regions = []
    for point_region in vor.point_region:
        region = vor.regions[point_region]
        if -1 in region:
            # TODO: Find the effective infinite point for this
            continue
        coordinates = [
            np.array([vor.vertices[vertex][0], vor.vertices[vertex][1]])
            for vertex in region
        ]
        if coordinates:
            coordinates = np.vstack(coordinates)
            polys.append(
                mpl.patches.Polygon(coordinates, linewidth=1.0, edgecolor="black")
            )
    colors = [coloration[key] for key in coloration.keys()]
    for pub_id, point_region in enumerate(vor.point_region):
        coordinates = [
            np.array([vor.vertices[vertex][1], vor.vertices[vertex][0]])
            for vertex in vor.regions[point_region]
        ]

        if coordinates:
            coordinates = np.vstack(coordinates)
        if np.any(np.abs(coordinates) > 300):
            continue

        centroid = np.mean(coordinates, axis=0)
        if with_labels:
            text = ax.text(
                centroid[1],
                centroid[0],
                pubs[pub_id].name,
                horizontalalignment="center",
                color="red",
                fontsize="x-small",
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground="black"),
                    path_effects.Normal(),
                ]
            )

    polys = mpl.collections.PatchCollection(
        polys, alpha=0.5, linewidth=2.0, edgecolor="black"
    )
    polys.set_cmap("Set3")
    polys.set_array(np.array(colors))
    ax.add_collection(polys)

    return ax


def find_satisfied_constraints(pubs, constraints: Tuple[str, Callable]):
    """
    Find which pub satisfy a list of constraints.

    Provide the constraints as a list of (attr, predicate) tuples
    where the attribute is something we can look up in a pub
    as getattr(pub, attr), and the predicate operates on that.
    """
    total_matches_arr = np.ones(len(pubs), dtype=bool)
    for attr, pred in constraints:
        matches_arr = np.ones(len(pubs), dtype=bool)
        for idx, pub in enumerate(pubs):
            matches_arr[idx] = pred(getattr(pub, attr))
        print(f"{np.sum(matches_arr)} pubs meet constraint {attr}.")
        total_matches_arr = np.logical_and(matches_arr, total_matches_arr)

    return [pubs[idx] for idx in range(len(pubs)) if total_matches_arr[idx]]


def populate_distances(pubs, pubgoers, person_locations, graph):
    positions = nx.get_node_attributes(graph, "pos")
    closest_nodes = find_all_closest_nodes(pubs, positions)
    print(f"Populating distances, {len(pubs)}, {len(person_locations)}")
    for pub in pubs:
        distance = 0.0
        for pubgoer in pubgoers:

            loc = person_locations[pubgoer]
            source_node = find_closest_node(loc, positions)
            try:
                shortest_distance = nx.shortest_path_length(
                    graph,
                    source=source_node,
                    target=closest_nodes[pub.name],
                    weight="length",
                )
                distance += shortest_distance
                print(
                    f"shortest path from {pub.name} to {pubgoer} is {shortest_distance}"
                )
            except nx.networkxnopath:
                print(f"could not find a path between {pub.name} and {pubgoer}")
        pub.distance = distance
    return pubs


def main():
    print(f"I have found {len(PUBS)} pubs.")
    if len(sys.argv) > 1:
        pubgoers = sys.argv[1:]
    else:
        pubgoers = ["Matt", "Martin", "Max"]
    person_locations = defaultdict(lambda: PTCL_LOCATION)
    person_locations["Matt"] = (-1.2951793697897114, 51.742673908496585)
    person_locations["Max"] = PTCL_LOCATION
    person_locations["Martin"] = PTCL_LOCATION

    nx_graph = load_graph("map-filter.gpickle")
    largest_component = list(nx.connected_components(nx_graph))[0]
    nx_graph = nx_graph.subgraph(largest_component).copy()

    positions = {
        node: np.array([nx_graph.nodes[node]["lon"], nx_graph.nodes[node]["lat"]])
        for node in nx_graph.nodes
    }
    nx.set_node_attributes(nx_graph, positions, name="pos")
    pubs = [pub for pub in PUBS if pub.is_open]
    pubs = populate_distances(pubs, pubgoers, person_locations, nx_graph)

    pub_vetos = set(name for pubgoer in pubgoers for name in PERSON_VETOS[pubgoer])

    valid_pubs = find_satisfied_constraints(
        pubs,
        [
            ("has_beer", lambda has_beer: has_beer),
            ("is_spoons", lambda is_spoons: not is_spoons),
            ("distance", lambda distance: distance < DISTANCE_CUTOFF * len(pubgoers)),
            (
                "cheapest_pint",
                lambda cheapest_pint: np.isinf(cheapest_pint) or cheapest_pint < 5.0,
            ),
            ("has_pub_quiz", lambda has_pub_quiz: not has_pub_quiz),
            ("has_live_music", lambda has_live_music: not has_live_music),
            ("name", lambda name: name not in pub_vetos),
            ("has_funny_smell", lambda has_funny_smell: not has_funny_smell),
        ],
    )
    print(f"We are left with {len(valid_pubs)} pubs. These are:")
    print(", ".join(pub.name for pub in valid_pubs))

    distance_weights = np.argsort([-pub.distance for pub in valid_pubs])
    print([pub.distance for pub in valid_pubs])
    price_weights = np.argsort([-pub.cheapest_pint for pub in valid_pubs])
    weights = (distance_weights + price_weights).astype(np.float64)
    weights = np.exp(weights) / np.sum(np.exp(weights))
    rng = np.random.default_rng()
    random_idx = rng.choice([idx for idx in range(len(valid_pubs))], p=weights)
    random_pub = valid_pubs[random_idx]
    print(
        f"I have randomly chosen {random_pub.name} with probability {weights[random_idx]*100:.1f}%"
    )

    FIG, AX = plt.subplots()

    ptcl_node = find_closest_node(PTCL_LOCATION, positions)
    pub_node = find_closest_node(random_pub.coordinates, positions)
    choice_path = nx.shortest_path(
        nx_graph, source=ptcl_node, target=pub_node, weight="length"
    )
    edge_list = [
        (choice_path[node], choice_path[node + 1])
        for node in range(len(choice_path) - 1)
    ]

    fig, ax = plt.subplots()
    ax = plot_pub_map(nx_graph, edge_list, pubs, ax=ax)
    ax = plot_pub_voronoi(pubs, ax=ax, with_labels=True)
    # ax.set_xlim(-1.27, -1.2540034)
    # ax.set_ylim(51.7400023, 51.7619972)
    fig.savefig("pub-map.png", dpi=500)


if __name__ == "__main__":
    main()
