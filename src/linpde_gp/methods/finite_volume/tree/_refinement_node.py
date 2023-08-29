import numpy as np
from linpde_gp.domains import Box


class RefinementNode:
    def __init__(
        self, domain: Box, parent: "RefinementNode" = None, condition: bool = True
    ):
        self._domain = domain
        self._level = 0 if parent is None else parent.level + 1
        self._children = []
        self._parent = parent
        self._condition = condition
        self._expanded = False

    @property
    def domain(self) -> Box:
        return self._domain

    @property
    def level(self) -> int:
        return self._level

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def parent(self) -> "RefinementNode":
        return self._parent

    @property
    def children(self) -> list["RefinementNode"]:
        return self._children

    @property
    def total_num_predecessors(self) -> int:
        return self._total_num_predecessors

    @property
    def condition(self) -> bool:
        return self._condition

    @property
    def expanded(self) -> bool:
        return self._expanded

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.domain})"

    def expand_octree(self, subdivision_axes=None, condition: bool = True) -> list["RefinementNode"]:
        self._children = [
            RefinementNode(subdomain, self, condition=condition)
            for subdomain in self.domain.subdivide_binary(subdivision_axes=subdivision_axes)
        ]
        self._expanded = True
        return self.children

    def expand_axis(
        self, axis: int, num_nodes: int = 2, condition: bool = True
    ) -> list["RefinementNode"]:
        self._children = []
        endpoints = np.linspace(
            self.domain[axis][0], self.domain[axis][1], num_nodes + 1
        )
        start_points = endpoints[:-1]
        end_points = endpoints[1:]

        for start_point, end_point in zip(start_points, end_points):
            bounds = self.domain.bounds.copy()
            bounds[axis] = [start_point, end_point]
            self._children.append(
                RefinementNode(Box(bounds), self, condition=condition)
            )
        self._expanded = True
        return self.children
    
    def remove_child(self, child: "RefinementNode"):
        self._children.remove(child)

    def __contains__(self, point) -> bool:
        return point in self.domain
