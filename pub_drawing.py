import logging
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib import cm
from scipy.spatial import Voronoi

from pub_data import Pub


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


def plot_highlighted_paths(
    graph: nx.Graph,
    path_lists: List[List[Tuple[int, int]]],
    ax: Optional[mpl.axes.Axes] = None,
    styles: Optional[List[str]] = None,
    draw_arrows: bool = False,
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
    styles
        List of styles per line, defaults to solid if not provided.
    draw_arrows
        Should we draw arrows on the direction of travel?

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

    if styles is None:
        styles = ["solid" for _ in range(len(unique_path_lists))]

    if isinstance(styles, str):
        styles = [styles for _ in range(len(unique_path_lists))]

    highlight_cmap = cm.get_cmap("gist_rainbow", lut=len(unique_path_lists))
    xs = [item[0] for item in nx.get_node_attributes(graph, "pos").values()]
    ys = [item[1] for item in nx.get_node_attributes(graph, "pos").values()]

    x_size, y_size = np.ptp(xs), np.ptp(ys)
    for idx, path_list in enumerate(unique_path_lists):

        arcs = nx.draw_networkx_edges(
            graph,
            pos=nx.get_node_attributes(graph, "pos"),
            ax=ax,
            edgelist=path_list,
            edge_color=highlight_cmap(idx),
            style=styles[idx],
            width=2.0,
        )
        if draw_arrows:
            for ((x0, y0), (x1, y1)) in arcs.get_segments():
                ax.arrow(
                    x0,
                    y0,
                    x1 - x0,
                    y1 - y0,
                    color=highlight_cmap(idx),
                    linestyle=styles[idx],
                    head_width=0.0008 * min(x_size, y_size),
                    overhang=0.1,
                    width=0.0,
                    length_includes_head=True,
                    shape="full",
                )
    return graph


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


def plot_pub_map(
    graph: nx.Graph,
    pubs: List[Pub],
    with_labels=False,
    ax: Optional[mpl.axes.Axes] = None,
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

    if with_labels:
        for pub in pubs:
            text = ax.text(
                pub.coordinates[0],
                pub.coordinates[1],
                pub.name,
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
    width = max_x - min_x
    height = max_y - min_y
    ax.set_xlim(min_x - 0.01 * width, 0.01 * width + max_x)
    ax.set_ylim(min_y - 0.01 * height, max_y + 0.01 * height)
    # ax.axis("off")

    return ax
