import copy

# Specific modules
import xml.sax  # parse osm file

import networkx as nx
from geopy.distance import geodesic


def read_osm(filename_or_stream, only_roads=True):
    """Read graph in OSM format from file specified by name or by stream object.
    Parameters
    ----------
    filename_or_stream : filename or stream object
    Returns
    -------
    G : Graph
    Examples
    --------
    >>> G=nx.read_osm(nx.download_osm(-122.33,47.60,-122.31,47.61))
    >>> import matplotlib.pyplot as plt
    >>> plt.plot([G.node[n]['lat']for n in G], [G.node[n]['lon'] for n in G], 'o', color='k')
    >>> plt.show()
    """
    osm = OSM(filename_or_stream)
    G = nx.DiGraph()

    ## Add ways
    for w in osm.ways.values():
        if only_roads and "highway" not in w.tags:
            continue

        if w.tags.get("oneway", "no") == "yes":
            nx.add_path(G, w.nds, id=w.id)
        else:
            # BOTH DIRECTION
            nx.add_path(G, w.nds, id=w.id)
            nx.add_path(G, w.nds[::-1], id=w.id)

    ## Complete the used nodes' information
    for n_id in G.nodes:
        n = osm.nodes[n_id]
        G.nodes[n_id]["lat"] = n.lat
        G.nodes[n_id]["lon"] = n.lon
        G.nodes[n_id]["id"] = n.id

    ## Estimate the length of each way
    for u, v in G.edges:
        distance = geodesic(
            (G.nodes[u]["lon"], G.nodes[u]["lat"]),
            (G.nodes[v]["lon"], G.nodes[v]["lat"]),
        ).km  # Give a realistic distance estimation (neither EPSG nor projection nor reference system are specified)

        G.edges[u, v]["length"] = distance
    return nx.Graph(G)


class Node:
    def __init__(self, _id, _lon, _lat):
        self.id = _id
        self.lon = _lon
        self.lat = _lat
        self.tags = {}

    def __str__(self):
        return f"Node (id : {self.id}) lon : {self.lon}, lat : {self.lat} "


class Way:
    def __init__(self, _id, _osm):
        self.osm = _osm
        self.id = _id
        self.nds = []
        self.tags = {}

    def split(self, dividers):
        # slice the node-array using this nifty recursive function
        def slice_array(ar, dividers):
            for i in range(1, len(ar) - 1):
                if dividers[ar[i]] > 1:
                    left = ar[: i + 1]
                    right = ar[i:]

                    rightsliced = slice_array(right, dividers)

                    return [left] + rightsliced
            return [ar]

        slices = slice_array(self.nds, dividers)

        # create a way object for each node-array slice
        ret = []
        i = 0
        for slice in slices:
            littleway = copy.copy(self)
            littleway.id += "-%d" % i
            littleway.nds = slice
            ret.append(littleway)
            i += 1

        return ret


class OSM:
    def __init__(self, filename_or_stream):
        """ File can be either a filename or stream/file object."""
        nodes = {}
        ways = {}

        superself = self

        class OSMHandler(xml.sax.ContentHandler):
            @classmethod
            def setDocumentLocator(self, loc):
                pass

            @classmethod
            def startDocument(self):
                pass

            @classmethod
            def endDocument(self):
                pass

            @classmethod
            def startElement(self, name, attrs):
                if name == "node":
                    self.currElem = Node(
                        attrs["id"], float(attrs["lon"]), float(attrs["lat"])
                    )
                elif name == "way":
                    self.currElem = Way(attrs["id"], superself)
                elif name == "tag":
                    self.currElem.tags[attrs["k"]] = attrs["v"]
                elif name == "nd":
                    self.currElem.nds.append(attrs["ref"])

            @classmethod
            def endElement(self, name):
                if name == "node":
                    nodes[self.currElem.id] = self.currElem
                elif name == "way":
                    ways[self.currElem.id] = self.currElem

            @classmethod
            def characters(self, chars):
                pass

        xml.sax.parse(filename_or_stream, OSMHandler)

        self.nodes = nodes
        self.ways = ways

        # count times each node is used
        node_histogram = dict.fromkeys(self.nodes.keys(), 0)
        for way in list(self.ways.values()):
            if (
                len(way.nds) < 2
            ):  # if a way has only one node, delete it out of the osm collection
                del self.ways[way.id]
            else:
                for node in way.nds:
                    node_histogram[node] += 1

        # use that histogram to split all ways, replacing the member set of ways
        new_ways = {}
        for id, way in self.ways.items():
            split_ways = way.split(node_histogram)
            for split_way in split_ways:
                new_ways[split_way.id] = split_way
        self.ways = new_ways
