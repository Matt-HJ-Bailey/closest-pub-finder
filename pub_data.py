#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to load pub data

@author: Matt Bailey
"""
from typing import List, Tuple

import datetime
import numpy as np
import pandas as pd


class Pub:
    def __init__(
        self,
        name: str,
        distance: float = float("inf"),
        has_beer: bool = True,
        has_pub_quiz: bool = False,
        is_spoons: bool = False,
        has_live_music: bool = False,
        address: str = None,
        coordinates: Tuple[float, float] = None,
        lat: float = None,
        lon: float = None,
        has_funny_smell: bool = False,
        cheapest_pint: float = float("inf"),
        is_open: bool = True,
        is_college: bool = False,
        last_visited: datetime.datetime = datetime.datetime(year=2020, month=1, day=1),
        **kwargs,
    ):
        """
        Plain Ol' Data class for a pub.
        :param name: The name of the pub (although in a dict we may end up storing this twice)
        :param distance: The walking distance in km as measured by Google Maps from the office.
        :param has_beer: Does this pub have real ale worth drinking?
        :param has_pub_quiz: Does this pub have a pub quiz on tonight?
        """
        self.name = name.replace(",", ";")
        self.distance = distance
        self.has_beer = has_beer
        self.has_pub_quiz = has_pub_quiz
        self.is_spoons = is_spoons
        self.has_live_music = has_live_music
        # Remove any commas from the address as it will screw up
        # csv reading
        self.address = address.replace(",", ";")
        if coordinates is not None:
            self.lat = coordinates[0]
            self.lon = coordinates[1]
        elif lat is not None and lon is not None:
            self.lat = lat
            self.lon = lon
        else:
            raise AttributeError("Must provide either coordinates or lat=,lon=")
        self.has_funny_smell = has_funny_smell
        self.cheapest_pint = cheapest_pint
        self.is_open = is_open
        self.is_college = is_college
        self.seconds_since_visit = (
            datetime.datetime.now() - last_visited
        ).total_seconds()
        if np.isnan(self.seconds_since_visit):
            self.seconds_since_visit = np.inf

        for key, val in kwargs.items():
            self.__setattr__(key, val)

    def __repr__(self):
        """
        Get a string representation of this pub
        """
        return ", ".join(str(value) for attr, value in self.__dict__.items())

    def to_csv_str(self) -> str:
        """
        Get this pub as a single string suitable for a .csv
        """
        return repr(self) + "\n"

    @property
    def coordinates(self) -> np.ndarray:
        """
        Get the coordinates of this as a lon, lat pair.

        Used for distance calculations later.
        """
        return np.array([self.lon, self.lat], dtype=float)

    def to_csv_str(self) -> str:
        """
        Get this pub as a single string suitable for a .csv
        """
        return repr(self) + "\n"

    @property
    def coordinates(self) -> np.ndarray:
        """
        Get the coordinates of this as a lon, lat pair.

        Used for distance calculations later.
        """
        return np.array([self.lon, self.lat], dtype=float)


def pub_from_dict(dictionary: dict) -> Pub:
    """
    Convert a pub from a dict (or dict-like).

    Parameters
    ----------
    dictionary
        Some key-val accessible thing with pub stats.
    """
    return Pub(**dictionary)


def import_pubs(filename: str) -> List[Pub]:
    """
    Import pubs from a csv file.

    The csv file should have a header of attributes
    that roughly match what is specified for a pub object,
    but this should cope with extras.

    Parameters
    ----------
    filename
        The name of the data file to read from.

    Returns
    -------
        a list of pubs that have been read from the file
    """
    df = pd.read_csv(filename, skipinitialspace=True, parse_dates=["last_visited"])
    pubs = [pub_from_dict(row) for idx, row in df.iterrows()]
    return pubs


if __name__ == "__main__":
    import_pubs("./oxford-pub-data.csv")
