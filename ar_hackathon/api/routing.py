"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the drive_unit_next_move function in this module.

*****IMPORTANT*****
Team name: SHIJ
Email address: matsunagai84@gmail.com, haolun7788@gmail.com, jeremyhhj@gmail.com", "thanhnhantran1501@gmail.com
*******************
"""

from typing import Optional
import heapq
from ar_hackathon.models.graph_state import GraphState


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    This is the function that students will implement. The game engine will
    call this function for each idle drive unit at each time step to
    determine where it should go next.

    Pickups and deliveries are automatic: a drive unit with free capacity
    that stops at (or passes through) a node with a waiting pod picks it up,
    and a drive unit that reaches a carried pod's destination station drops
    it off.

    Args:
        drive_unit_id: ID of the drive unit being routed
        state: GraphState object containing the current state of the floor

    Returns:
        next_node_id: ID of an adjacent node to move to, or None to wait
                      at the current node
    """
    # If the unit can't be found or is currently in transit, wait
    unit = state.get_drive_unit(drive_unit_id)
    if unit is None:
        return None
    if unit.in_transit:
        return None

    start = unit.current_node

    # Determine targets: if the unit has capacity, target the nearest available
    # pod (one that is not currently being carried). If the unit is full,
    # target the delivery station of the first carried pod.
    targets = set()
    if unit.has_capacity:
        for pod in state.active_pods:
            if pod.current_node is not None and pod.carried_by is None:
                targets.add(pod.current_node)
    else:
        for pod in state.active_pods:
            if pod.carried_by == unit.id:
                targets.add(pod.destination_station)

    if not targets:
        return None

    # Use Dijkstra's algorithm to find the shortest-weighted path to the
    # nearest target node (weights are taken from edges' `weight` field).
    dist = {start: 0.0}
    prev = {}
    heap = [(0.0, start)]

    while heap:
        d, node = heapq.heappop(heap)
        if d > dist.get(node, float('inf')):
            continue

        if node in targets:
            # Reconstruct path from start -> node
            path = [node]
            cur = node
            while cur != start:
                cur = prev[cur]
                path.append(cur)
            path.reverse()
            if len(path) == 1:
                return None
            return path[1]

        for nbr in state.neighbors(node):
            edge = state.get_edge(node, nbr)
            if edge is None:
                continue
            w = edge.weight
            nd = d + w
            if nd < dist.get(nbr, float('inf')):
                dist[nbr] = nd
                prev[nbr] = node
                heapq.heappush(heap, (nd, nbr))

    # No reachable target
    return None
