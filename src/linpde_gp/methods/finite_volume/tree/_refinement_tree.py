import numpy as np
from linpde_gp.domains import Box

from ._refinement_node import RefinementNode


class RefinementTree:
    def __init__(self, domain: Box):
        self._domain = domain
        self._root = RefinementNode(domain, None)
        self._num_items = 1
        self._depth = 0
        self._leaves = [self._root]

    @classmethod
    def octree_from_depth(cls, domain: Box, depth: int, condition: bool = True):
        tree = cls(domain)
        while tree.depth < depth:
            tree.expand_leaves_octree(condition=condition)
        return tree

    @property
    def domain(self) -> Box:
        return self._domain

    @property
    def leaves(self) -> list[RefinementNode]:
        return self._leaves

    def __len__(self) -> int:
        return self._num_items

    def __iter__(self):
        # Iterate over entire tree in breadth-first order
        queue = [self._root]
        while queue:
            node = queue.pop(0)
            yield node
            queue.extend(node.children)

    @property
    def depth(self) -> int:
        return self._depth

    def _expand_leaf(self, leaf_idx, expand_fn):
        leaf = self._leaves.pop(leaf_idx)
        new_leaves = expand_fn(leaf)
        self._leaves.extend(new_leaves)
        self._num_items += len(leaf.children)
        if leaf.level == self._depth:
            self._depth += 1
        return new_leaves

    def expand_leaf_octree(self, leaf_idx, subdivision_axes=None, condition=True):
        return self._expand_leaf(
        leaf_idx, lambda leaf: leaf.expand_octree(subdivision_axes=subdivision_axes, condition=condition)
        )

    def expand_leaf_axis(
        self, leaf_idx, axis: int, num_nodes: int = 2, condition: bool = True
    ):
        return self._expand_leaf(
            leaf_idx,
            lambda leaf: leaf.expand_axis(axis, num_nodes, condition=condition),
        )

    def expand_leaves_axis(self, axis: int, num_nodes: int = 2, condition: bool = True, idcs = None):
        new_leaves = []
        if idcs is None:
            for _ in range(len(self.leaves)):
                new_leaves += self.expand_leaf_axis(0, axis, num_nodes, condition=condition)
        else:
            idcs = np.sort(idcs) - np.arange(len(idcs))
            for idx in idcs:
                new_leaves += self.expand_leaf_axis(idx, axis, num_nodes, condition=condition)
        return new_leaves

    def expand_leaves_octree(self, subdivision_axes=None, condition: bool = True, idcs = None):
        new_leaves = []
        if idcs is None:
            for _ in range(len(self.leaves)):
                new_leaves += self.expand_leaf_octree(0, subdivision_axes=subdivision_axes, condition=condition)
        else:
            idcs = np.sort(idcs) - np.arange(len(idcs))
            for idx in idcs:
                new_leaves += self.expand_leaf_octree(idx, subdivision_axes=subdivision_axes, condition=condition)
        return new_leaves

    def get_unexpanded_leaves(self) -> list[int]:
        return [idx for idx, leaf in enumerate(self.leaves) if not leaf.expanded]
    
    def expand_unexpanded_leaves_octree(self, subdivision_axes=None):
        self.expand_leaves_octree(subdivision_axes=subdivision_axes, idcs=self.get_unexpanded_leaves())

    def find_point(self, point) -> RefinementNode:
        # Find the leaf node that contains the given point
        node = self._root
        while not node.is_leaf:
            for child in node.children:
                if point in child:
                    node = child
                    break
        return node
    
    def remove_leaf(self, leaf: RefinementNode):
        parent = leaf.parent
        if parent is None:
            raise ValueError("Cannot remove root node")
        parent.remove_child(leaf)
        self._leaves.remove(leaf)
        self._num_items -= 1
        if len(parent.children) == 0:
            self._leaves.append(parent)
    
    def refine(self, score_fn: callable, score_threshold: float, subdivision_axes=None):
        self.expand_unexpanded_leaves_octree(subdivision_axes=subdivision_axes)
        new_leaves_idcs = self.get_unexpanded_leaves()
        new_leaves = np.take(self.leaves, new_leaves_idcs)

        scores = score_fn(new_leaves)

        accepted_leaves = []
        for leaf, score in zip(new_leaves, scores):
            if score < score_threshold:
                self.remove_leaf(leaf)
            else:
                accepted_leaves.append(leaf)
        return accepted_leaves
        

    def get_conditioning_nodes(self) -> list[RefinementNode]:
        return [node for node in self if node.condition]
    
    def get_conditioning_domains(self) -> list[RefinementNode]:
        return [node.domain for node in self if node.condition]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.domain})"
