import argparse
import logging
import pickle as pkl
import sys
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from geopy.distance import geodesic
from matplotlib import cm
from scipy.spatial import Voronoi

from osm import read_osm
from pub_data import PUBS, Pub

PTCL_LOCATION = (-1.2535525, 51.7593620)
PERSON_VETOS = defaultdict(set)
PERSON_VETOS["Max"] = {
    "The Lamb and Flag",
}
PERSON_VETOS["Matt"] = {"The Half Moon", "The Black Swan"}

Coordinate = Tuple[float, float]
_closest_node_cache: Dict[Coordinate, int] = {}


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
    current_distance = np.inf
    node_index = None
    # If this is cached, return the cached value
    if current_pos in _closest_node_cache:
        return _closest_node_cache[current_pos]
    for key, val in positions.items():
        distance = geodesic(current_pos, val)
        if distance < current_distance:
            current_distance = distance
            node_index = key

    # Return the cached value
    _closest_node_cache[current_pos] = node_index
    return node_index


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
    try:
        nx_graph = nx.read_gpickle(filename + ".pkl")
        logging.info("Loading graph from pickle")
    except FileNotFoundError:
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
    regenerate
        if True, reload the whole collection. if not, use a pickle.

    Returns
    -------
        A dict mapping pub names to the closest graph node
    """
    closest_nodes = dict()
    try:
        if not regenerate:
            logging.info("Loading closest nodes from pickle")
            with open("all-closest-nodes.pkl", "rb") as fi:
                loaded_dict = pkl.load(fi)
            rewrite_pickle = False
            for pub in pubs:
                if pub.name not in loaded_dict:
                    logging.info(f"{pub.name} not in pickle, finding closest.")
                    rewrite_pickle = True
                    loaded_dict[pub.name] = find_closest_node(
                        pub.coordinates, positions
                    )
            if rewrite_pickle:
                with open("all-closest-nodes.pkl", "wb") as fi:
                    pkl.dump(loaded_dict, fi)
            return loaded_dict
    except FileNotFoundError:
        logging.warn("Could not find all-closest-nodes.pkl")
    except TypeError as e:
        logging.warn(str(e))

    for pub in pubs:
        logging.info("Finding closest", pub.name)
        closest_node = find_closest_node(pub.coordinates, positions)
        closest_nodes[pub.name] = closest_node
    with open("all-closest-nodes.pkl", "wb") as fi:
        pkl.dump(closest_nodes, fi)
    return closest_nodes


def plot_pub_map(
    graph: nx.Graph, pubs: List[Pub], ax: Optional[mpl.axes.Axes] = None
) -> mpl.axes.Axes:
    """
    Plot the provided graph as a map, highlighting pub locations.

    Parameters
    ----------
    graph
        A graph of roads and footpaths. Must have attribute 'pos'
    pubs
        A list of pubs to highlight
    ax
        The axis to plot onto, will generate a new one if not provided

    Returns
    -------
        an axis with a map drawn on it.
    """
    if ax is None:
        logging.info("Plotting new fig")
        _, ax = plt.subplots()

    positions = nx.get_node_attributes(graph, "pos")
    nx.draw_networkx_edges(graph, pos=positions, ax=ax)

    min_x = min(pos[0] for pos in positions.values())
    max_x = max(pos[0] for pos in positions.values())
    min_y = min(pos[1] for pos in positions.values())
    max_y = max(pos[1] for pos in positions.values())

    ax.scatter(
        [pub.coordinates[0] for pub in pubs],
        [pub.coordinates[1] for pub in pubs],
        s=25,
        c="blue",
    )
    width = max_x - min_x
    height = max_y - min_y
    ax.set_xlim(min_x - 0.01 * width, 0.01 * width + max_x)
    ax.set_ylim(min_y - 0.01 * height, max_y + 0.01 * height)
    # ax.axis("off")

    return ax


def plot_highlighted_paths(
    graph: nx.Graph,
    path_lists: List[List[Tuple[int, int]]],
    ax: Optional[mpl.axes.Axes] = None,
) -> mpl.axes.Axes:
    """
    Highlight certain paths on the map.

    Parameters
    ----------
    graph
        A road / path map you want plotted with attribute 'pos'
    path_lists
       A list of list of (u, v) edge tuples to highlight.
    ax
        The axes to plot onto
    Returns
    -------
        An axis with highlighted paths on it, each uniquely coloured.
    """
    if ax is None:
        logging.info("Plotting new fig")
        _, ax = plt.subplots()

    seen = set()
    unique_path_lists = []
    for path_list in path_lists:
        if set(path_list) in seen:
            continue
        else:
            unique_path_lists.append(path_list)
            seen.add(frozenset(path_list))
    highlight_cmap = cm.get_cmap("gist_rainbow", lut=len(unique_path_lists))
    for idx, path_list in enumerate(unique_path_lists):

        nx.draw_networkx_edges(
            graph,
            pos=nx.get_node_attributes(graph, "pos"),
            ax=ax,
            edgelist=path_list,
            edge_color=highlight_cmap(idx),
            width=2.0,
        )
    return graph


def sort_coordinates_anticlockwise(coordinates: np.ndarray) -> np.ndarray:
    """
    sort a list of coordinates to be in anticlockwise order.

    Parameters
    ----------
    coordinates
        Vertex coordinates of a polygon

    Returns
    -------
        Coordinates sorted in anticlockwise order
    """
    centroid = coordinates.mean(axis=0)
    angles = np.arctan2(
        coordinates[:, 1] - centroid[1], coordinates[:, 0] - centroid[0]
    )
    return coordinates[np.argsort(angles)]


def plot_pub_voronoi(
    pubs: List[Pub], ax: Optional[mpl.axes.Axes] = None, with_labels: bool = True
) -> mpl.axes.Axes:
    """
    Plot a voronoi diagram of pubs.

    Will fill in polygons at infinity and do a minimal colouring.
    Parameters
    ----------
    pubs
        List of pubs with attribute 'coordinates'
    ax
        The axis to plot onto
    with_labels
        Should we plot the names of pubs?

    Returns
    -------
        An axis with a voronoi diagram on it
    """
    if ax is None:
        logging.info("plotting new fig")
        _, ax = plt.subplots()

    points = np.array([pub.coordinates for pub in pubs])

    vor = Voronoi(points)

    dual_graph = nx.Graph()
    polys = []
    name_order = []

    all_ridges: Dict[int, List[int]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    center = vor.points.mean(axis=0)
    radius = max(vor.points[:, 0].ptp(), vor.points[:, 1].ptp())

    for pub_id, point_region in enumerate(vor.point_region):
        vertices_a = set(vor.regions[point_region])
        for other_pub_id, other_point_region in enumerate(vor.point_region):
            vertices_b = set(vor.regions[other_point_region])
            overlap = vertices_a.intersection(vertices_b)
            if len(overlap) == 2:
                dual_graph.add_edge(pubs[pub_id].name, pubs[other_pub_id].name)

        # Create the polygons
        vertices = vor.regions[point_region]
        coords_list = [
            np.array([vor.vertices[vertex][0], vor.vertices[vertex][1]])
            for vertex in vertices
            if vertex >= 0
        ]
        if -1 in vertices:
            # reconstruct a non-finite region
            for p2, v1, v2 in all_ridges[pub_id]:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge
                t = vor.points[p2] - vor.points[pub_id]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[pub_id, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius
                coords_list.append(far_point)

        coordinates = sort_coordinates_anticlockwise(np.vstack(coords_list))
        polys.append(mpl.patches.Polygon(coordinates, linewidth=1.0, edgecolor="black"))

        # Plot the text labels at the centre of each region
        if with_labels:
            # Use the pub position if this is reconstructed
            if -1 in vertices:
                centroid = pubs[pub_id].coordinates
            else:
                centroid = np.mean(coordinates, axis=0)
            text = ax.text(
                centroid[0],
                centroid[1],
                pubs[pub_id].name,
                horizontalalignment="center",
                color="white",
                fontsize="x-small",
            )
            text.set_path_effects(
                [
                    path_effects.Stroke(linewidth=1, foreground="black"),
                    path_effects.Normal(),
                ]
            )
        name_order.append(pubs[pub_id].name)

    coloration = nx.greedy_color(dual_graph, interchange=True)
    colors = [coloration[name] for name in name_order]
    polys = mpl.collections.PatchCollection(
        polys, alpha=0.5, linewidth=2.0, edgecolor="black"
    )
    polys.set_cmap("tab10")
    polys.set_array(np.array(colors))
    ax.add_collection(polys)

    return ax


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
    # print(f"Populating distances, {len(pubs)}, {len(person_locations)}")
    _distance_cache: Dict[Tuple[str, Coordinate], float] = {}
    for pub in pubs:
        distance = 0.0
        for pubgoer in pubgoers:

            loc = person_locations[pubgoer]
            if (pub.name, loc) in _distance_cache:
                distance += _distance_cache[(pub.name, loc)]
                logging.info(
                    f"shortest path from {pub.name} to {pubgoer} is {shortest_distance}"
                )
                continue

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
                # print(
                #    f"shortest path from {pub.name} to {pubgoer} is {shortest_distance}"
                # )
            except nx.NetworkXNoPath:
                print(f"could not find a path between {pub.name} and {pubgoer}")
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
        "--temperature",
        default=1.0,
        type=float,
        help="Monte Carlo temperature for deciding on which pub. Higher temperature means less optimal choices.",
    )
    args = parser.parse_args()

    temperature = args.temperature
    pubgoers = args.pubgoers

    person_locations = defaultdict(lambda: PTCL_LOCATION)

    if os.path.exists("./locations.csv"):
        person_df = pd.read_csv("./locations.csv")
        for row in person_df.iterrows:
            person_locations[row["name"]] = (row["lat"], row["lon"])

    nx_graph = load_graph("map.osm")

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
            ("distance", lambda distance: distance < args.distance * len(pubgoers)),
            (
                "cheapest_pint",
                lambda cheapest_pint: np.isinf(cheapest_pint) or cheapest_pint < 5.0,
            ),
            ("has_pub_quiz", lambda has_pub_quiz: not has_pub_quiz),
            ("has_live_music", lambda has_live_music: not has_live_music),
            ("name", lambda name: name not in pub_vetos),
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
        exit()
    else:
        print("We are left with zero valid pubs. We are having tinnies in the office.")
        exit()

    if args.martin:
        price_weights = argsort_preserve_ties(
            [-pub.cheapest_pint for pub in valid_pubs]
        )
    else:
        price_weights = argsort_preserve_ties([pub.cheapest_pint for pub in valid_pubs])

    distance_weights = argsort_preserve_ties([pub.distance for pub in valid_pubs])

    weights = np.exp(-(distance_weights + price_weights) / temperature)
    weights = weights / np.sum(weights)
    print("-------------------------------------⊤------⊤------⊤------⊣")
    print("       Name                          | Dist.| Pint |Chance|")
    print("-------------------------------------+------+------+------⊣")
    for pub_id in range(len(valid_pubs)):
        print(
            f"{valid_pubs[pub_id].name:36} | {valid_pubs[pub_id].distance:4.1f} | {valid_pubs[pub_id].cheapest_pint:4.2f} | {weights[pub_id]*100:4.1f}%|"
        )
    print("-------------------------------------⊥------⊥------⊥------⊣")

    rng = np.random.default_rng()
    random_idx = rng.choice([idx for idx in range(len(valid_pubs))], p=weights)
    random_pub = valid_pubs[random_idx]
    print(
        f"I have randomly chosen {random_pub.name} with probability {weights[random_idx]*100:.1f}%"
    )

    paths_highlight = []
    for pubgoer in pubgoers:
        person_node = find_closest_node(person_locations[pubgoer], positions)
        pub_node = find_closest_node(random_pub.coordinates, positions)
        choice_path = nx.shortest_path(
            nx_graph, source=person_node, target=pub_node, weight="length"
        )
        paths_highlight.append(
            [
                (choice_path[node], choice_path[node + 1])
                for node in range(len(choice_path) - 1)
            ]
        )

    fig, ax = plt.subplots()
    ax = plot_pub_map(nx_graph, pubs, ax=ax)
    ax = plot_pub_voronoi(pubs, ax=ax, with_labels=True)
    ax = plot_highlighted_paths(nx_graph, paths_highlight, ax=ax)
    plt.show()


if __name__ == "__main__":
    main()
