import pickle as pkl
import random

import matplotlib as mpl
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from scipy.spatial import Voronoi, voronoi_plot_2d

from osm import *
from pub_data import *

## Modules


def find_closest_node(current_pos, positions):
    current_distance = np.inf
    node_index = None
    for key, val in positions.items():
        distance = geodesic(current_pos, val)
        if distance < current_distance:
            current_distance = distance
            node_index = key
    return node_index


def load_graph(filename):
    try:
        nx_graph = nx.read_gpickle(filename)
    except FileNotFoundError:
        nx_graph = read_osm("map-filter.osm")
        nx.write_gpickle(nx_graph, filename)
    return nx_graph


def find_all_closest_nodes(pubs):
    closest_nodes = dict()
    try:
        closest_nodes = pkl.load("all-closest-nodes.pkl")
    except FileNotFoundError:
        for pub in PUBS:
            closest_node = find_closest_node(pub.coordinates, POSITIONS)
            closest_nodes[pub.name] = closest_node
    except TypeError:
        for pub in PUBS:
            closest_node = find_closest_node(pub.coordinates, POSITIONS)
            closest_nodes[pub.name] = closest_node


print(f"I have found {len(PUBS)} pubs.")

PTCL_LOCATION = (-1.2535525, 51.7593620)
GLOUCESTER_GREEN = (-1.26248, 51.75397)
GEODESIC_DISTANCES = [geodesic(PTCL_LOCATION, pub.coordinates).km for pub in PUBS]
print(GEODESIC_DISTANCES)
DISTANCE_CUTOFF = 1.0
EXPENSIVE_PUB_NAMES = {"The Chequers", "BrewDog", "The Anchor", "The Kings Arms"}
MAX_VETOS = {"The Lamb and Flag"}
MATT_VETOS = {"The Half Moon", "The Black Swan"}
MIKE_VETOS = {"The Royal Oak"}

PUBS_WITHIN_DISTANCE = np.array([pub.distance < DISTANCE_CUTOFF for pub in PUBS])
print(
    f"{np.sum(PUBS_WITHIN_DISTANCE)} out of {len(PUBS)} are within {DISTANCE_CUTOFF}km."
)

PUBS_THAT_ARE_NOT_SPOONS = np.array([pub.is_spoons == False for pub in PUBS])
print(f"{np.sum(PUBS_THAT_ARE_NOT_SPOONS)} out of {len(PUBS)} are not Spoons.")
PUBS_WITHOUT_PUB_QUIZ = np.array([pub.has_pub_quiz == False for pub in PUBS])
print(
    f"{np.sum(PUBS_WITHOUT_PUB_QUIZ)} out of {len(PUBS)} do not have a pub quiz tonight."
)
PUBS_WITHOUT_LIVE_MUSIC = np.array([pub.has_live_music == False for pub in PUBS])
print(
    f"{np.sum(PUBS_WITHOUT_LIVE_MUSIC)} out of {len(PUBS)} do not have live music tonight."
)
PUBS_WITH_GOOD_BEER = np.array([pub.has_beer == True for pub in PUBS])
print(
    f"{np.sum(PUBS_WITH_GOOD_BEER)} out of {len(PUBS)} have what Matt considers to be drinkable beer."
)
PUBS_WITHOUT_FUNNY_SMELL = np.array([pub.has_funny_smell == False for pub in PUBS])
print(
    f"{np.sum(PUBS_WITHOUT_FUNNY_SMELL)} out of {len(PUBS)} do not have a funny smell according to Spanish Issy."
)
PUBS_THAT_ARE_NOT_EXPENSIVE = np.array(
    [pub.name not in EXPENSIVE_PUB_NAMES for pub in PUBS]
)
print(
    f"{np.sum(PUBS_THAT_ARE_NOT_EXPENSIVE)} out of {len(PUBS)} have what Matt considers to be reasonable prices."
)
PUB_REQUIREMENTS = np.logical_and.reduce(
    [
        PUBS_WITHIN_DISTANCE,
        PUBS_THAT_ARE_NOT_SPOONS,
        PUBS_WITHOUT_PUB_QUIZ,
        PUBS_WITHOUT_LIVE_MUSIC,
        PUBS_WITH_GOOD_BEER,
        PUBS_THAT_ARE_NOT_EXPENSIVE,
        PUBS_WITHOUT_FUNNY_SMELL,
    ]
)
print(
    f"{np.sum(PUB_REQUIREMENTS)} out of {len(PUBS)} have meet all of these requirements."
)


PUBS_MAX_WILL_GO_TO = np.array([pub.name not in MAX_VETOS for pub in PUBS])
print(f"{np.sum(PUBS_MAX_WILL_GO_TO)} out of {len(PUBS)} have not been veto'd by Max.")
PUBS_MATT_WILL_GO_TO = np.array([pub.name not in MATT_VETOS for pub in PUBS])
print(
    f"{np.sum(PUBS_MATT_WILL_GO_TO)} out of {len(PUBS)} have not been veto'd by Matt."
)
PUBS_MIKE_WILL_GO_TO = np.array([pub.name not in MIKE_VETOS for pub in PUBS])
print(
    f"{np.sum(PUBS_MIKE_WILL_GO_TO)} out of {len(PUBS)} have not been veto'd by Mike."
)

PUB_REQUIREMENTS = np.logical_and.reduce(
    [PUB_REQUIREMENTS, PUBS_MAX_WILL_GO_TO, PUBS_MATT_WILL_GO_TO, PUBS_MIKE_WILL_GO_TO]
)
print(f"We are left with {np.sum(PUB_REQUIREMENTS)} pubs. These are:")
for index, cond in enumerate(PUB_REQUIREMENTS):
    if cond:
        print(PUBS[index].name, end=", ")
print("")
PUB_LIST = [pub for i, pub in enumerate(PUBS) if PUB_REQUIREMENTS[i]]
# RANDOM_CHOICE = random.choice(PUB_LIST).name
RANDOM_CHOICE = "The University Club"
print(f"I have randomly chosen {RANDOM_CHOICE}.")


points = np.array([pub.coordinates for pub in PUBS])


FIG, AX = plt.subplots()

vor = Voronoi(points)
polys = []
for region in vor.regions:
    if -1 in region:
        continue
    coordinates = [
        np.array([vor.vertices[vertex][1], vor.vertices[vertex][0]])
        for vertex in region
    ]
    if coordinates:
        coordinates = np.vstack(coordinates)
        polys.append(mpl.patches.Polygon(coordinates, linewidth=1.0, edgecolor="black"))
region = None
coordinates = None
for pub_id, point_region in enumerate(vor.point_region):
    coordinates = [
        np.array([vor.vertices[vertex][1], vor.vertices[vertex][0]])
        for vertex in vor.regions[point_region]
    ]
    if coordinates:
        coordinates = np.vstack(coordinates)
    centroid = np.mean(coordinates, axis=0)
    text = AX.text(
        centroid[0],
        centroid[1],
        PUBS[pub_id].name,
        horizontalalignment="center",
        color="red",
    )
    text.set_path_effects(
        [path_effects.Stroke(linewidth=3, foreground="black"), path_effects.Normal()]
    )

colors = 100 * np.random.rand(len(polys))
polys = mpl.collections.PatchCollection(
    polys, alpha=0.5, linewidth=2.0, edgecolor="black"
)
polys.set_cmap("Set3")
polys.set_array(np.array(colors))

NX_GRAPH = load_graph("map-filter.gpickle")
POSITIONS = {
    node: np.array([NX_GRAPH.nodes[node]["lon"], NX_GRAPH.nodes[node]["lat"]])
    for node in NX_GRAPH.nodes
}

CLOSEST_NODES = find_all_closest_nodes(PUBS)

for pub in PUBS:
    closest_node = CLOSEST_NODES[pub.name]
    print(
        f"The closest street node to {pub.name} is {closest_node}, which is at {POSITIONS[closest_node]} compared to {pub.coordinates}"
    )

min_x = min(pos[0] for pos in POSITIONS.values())
max_x = max(pos[0] for pos in POSITIONS.values())
min_y = min(pos[1] for pos in POSITIONS.values())
max_y = max(pos[1] for pos in POSITIONS.values())

print("Plotting graph")


voronoi_plot_2d(vor, ax=AX, show_points=True, show_vertices=False, line_colors="red")
networkx.draw_networkx_edges(NX_GRAPH, pos=POSITIONS, ax=AX)
AX.scatter(
    [pub.coordinates[0] for pub in PUBS],
    [pub.coordinates[1] for pub in PUBS],
    s=25,
    c="black",
)
AX.set_xlim(min_x, max_x)
AX.set_ylim(min_y, max_y)
AX.add_collection(polys)
plt.show()
print("plotted")
