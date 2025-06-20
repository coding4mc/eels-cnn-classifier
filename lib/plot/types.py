from typing import TypedDict, List


class Coordinate(TypedDict):
    x: int
    y: int


class GraphClickData:
    points: List[Coordinate]