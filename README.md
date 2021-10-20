# closest-pub-finder

This is a python program that finds the most suitable pub in Oxford.

## Usage

```
python pub_satisfier.py --distance 2.0 --temperature 1.0 Alice Bob Charles
```
The arguments are
* --distance the maximum distance pubgoers are willing to travel
* --temperature a temperature factor for the random choice, a higher T will cause a less optimal pub to be picked.
* --martin Is someone else paying? In which case, invert the price ordering.
* list of pubgoers  A list of the people going to the pub

This program also requires a file called `map.osm` containing OpenStreetMap data. You can get this from OpenStreetMap exporter: [https://www.openstreetmap.org/export#map=13/51.7465/-1.2234](https://www.openstreetmap.org/export#map=13/51.7/-1.2).
It also requires a file called `locations.csv` which will contain rows of `name, lat, lon`. This will be used if you're not all travelling from the default location.

## Requirements

This relies on
* matplotlib
* networkx
* numpy
* geopy
* scipy
* xml


## Pub Data


The file `pub_data.py` contains objects that detail every pub in Oxford with a variety of attributes -- whether they are open, their address, the cheapest pint as of 2021, etc. Most importantly this describes latitude and longitude coordinates for every pub, which will help with navigation.

Current a pub has the following attributes:
| attribute       | Description                                                                            |
|-----------------|----------------------------------------------------------------------------------------|
| name            | The name of the pub (should be unique)                                                 |
| distance        | The distance from your location to the pub, can be overwritten later                   |
| has_beer        | Does this pub have good beer?                                                          |
| has_pub_quiz    | Does this pub have a pub quiz on tonight?                                              |
| is_spoons       | Is this pub a spoons? If yes, never go there.                                          |
| has_live_music  | Does this pub have live music on tonight?                                              |
| address         | Street address of this pub                                                             |
| coordinates     | (lat, lon) coordinates of the pub. These get swapped by various routines so be careful |
| has_funny_smell | Some pubs have a funny smell.                                                          |
| cheapest_pint   | Float containing the cheapest pint, assumed to be infinity otherwise.                  |
| is_open         | Is this pub currently open?                                                            |
| is_college      | Is this a college bar?                                                                 |

## Pub Satisfier

The algorithm will go through two stages: first, applying hard constraints to rule out pubs.
The hard constraint satisfier will go through a list of (attribute, predicate) pairs, and call `predicate(pub.attribute)` for each pub in the collection. Then this will compute a logical `and` of every predicate, and select only pubs where all predicates are met. For example, this will select all pubs that are within 2km, do not have a funny smell, and are not Wetherspoons.

Second, the algorithm will assign a weight based on soft factors. The soft factors currently taken into account are net distance to all pubgoers, and the cheapest pint. This step uses the `osm.py` module to read in a map of Oxford, and read `locations.csv` to find the coordinates of each pubgoer specified at the command line. Then we find the shortest path from the person to each pub, and assign that a distance.
The distances are ranked with ties having the same value -- for example, `[1.2, 6.3, 2.5, 2.5, 0.8]` would be given ranks `[1, 3, 2, 2, 0]`.
The prices are ranked similarly, again with ties having the same value -- for example, `[inf, 5.3, inf, 4.1, 4.3]` would be given price ranks `[3, 2, 3, 0, 1]`.
The ranks are then added (so the total rank for the example pubs would be `[4, 5, 5, 2, 1]`. The ranks are weighted using a Boltzmann factor and normalised to one, so the new rank is `exp(-R/T)/sum_i exp(-R_i / T)` where `R_i` is the rank of pub `i` and `T` is a temperature parameter specified at the command line.
The ranking is printed and a best pub is randomly chosen.

## Caching

Finding the closest graph nodes corresponding to real coordinates is extremely slow for large maps, so this program will generate one pickle file. This is `map.osm.pkl` which is a `gpickle` of the loaded map to save double processing.
The real coordinates of the pubs are snapped to the graph by using a KDTree, which finds the closest node to each part of the graph. This KDTree is cached, and can be slow to build for large graphs.

## Pub Drawing

The pub map will also produce a Voronoi diagram of the map.
This will plot the roads specified in the `.osm` file, and plot the path from each pubgoer to the pub of choice.
A polygon is then drawn per pub finding the regions of the map that are closest as the crow flies to that pub -- note that this is not necessarily the same as the route picked by the pathing algorithm (for example, rivers and uncrossable roads aren't factored in). The Voronoi diagram is given a minimal colouring such that no two adjacent polygons are the same colour.
