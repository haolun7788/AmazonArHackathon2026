"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.

*****IMPORTANT*****
Team name: SHIJ
Email address: matsunagai84@gmail.com, haolun7788@gmail.com, jeremyhhj@gmail.com, thanhnhantran1501@gmail.com
*******************
"""

from typing import Optional, Dict, List, Tuple, FrozenSet
import heapq

from ar_hackathon.models.graph_state import GraphState
from ar_hackathon.utils.routing_utils import is_valid_move

# ---------------------------------------------------------------------------
# The engine keeps this module loaded for the whole simulation and calls
# drive_unit_next_move once per idle unit per time step, so module-level
# state here persists between calls. We use that for two things:
#
#   1. The floor layout (nodes/edges) is fixed for a whole run, so we build
#      adjacency lists once and cache shortest-path results instead of
#      recomputing them from scratch on every single call.
#   2. Which pod each drive unit is currently chasing, so idle units don't
#      all converge on the same waiting pod.
#
# The strategy itself is simple and works the same way at every level:
#   - A unit carrying pod(s) heads for the nearest destination among them.
#   - An idle unit with spare capacity claims the nearest unclaimed waiting
#     pod and heads for it.
#   - Whichever of those two targets is closer wins, so a unit with room to
#     spare will grab a pod on its way to a delivery rather than passing it
#     up needlessly.
#   - Every step, the unit re-picks its *next hop* (not just once per
#     target) as whichever open neighbor makes the most progress toward the
#     target. Recomputing this every call - rather than committing to a
#     fixed path - lets a unit route around a jammed aisle or a full dock
#     the moment an alternative opens up, which matters once capacity
#     limits show up (Level 2+).
# ---------------------------------------------------------------------------

_graph_signature: Optional[Tuple[FrozenSet[int], int]] = None
_forward_adj: Dict[int, List[Tuple[int, float]]] = {}
_reverse_adj: Dict[int, List[Tuple[int, float]]] = {}
_dist_to_cache: Dict[int, Dict[int, float]] = {}

_claimed_by: Dict[str, int] = {}   # pod_id -> unit_id currently pursuing it
_unit_claim: Dict[int, str] = {}   # unit_id -> pod_id it is currently pursuing


def _graph_sig(state: GraphState) -> Tuple[FrozenSet[int], int]:
    """Cheap fingerprint of the floor layout, used to detect a new test
    case (e.g. if this process ever gets reused across runs) and rebuild
    cached adjacency/claims instead of using stale ones."""
    return (frozenset(n.id for n in state.nodes), len(state.edges))


def _rebuild_graph(state: GraphState) -> None:
    global _graph_signature, _forward_adj, _reverse_adj, _dist_to_cache
    forward: Dict[int, List[Tuple[int, float]]] = {n.id: [] for n in state.nodes}
    reverse: Dict[int, List[Tuple[int, float]]] = {n.id: [] for n in state.nodes}

    for edge in state.edges:
        forward.setdefault(edge.from_node, []).append((edge.to_node, edge.weight))
        reverse.setdefault(edge.to_node, []).append((edge.from_node, edge.weight))
        if edge.bidirectional:
            forward.setdefault(edge.to_node, []).append((edge.from_node, edge.weight))
            reverse.setdefault(edge.from_node, []).append((edge.to_node, edge.weight))

    _forward_adj = forward
    _reverse_adj = reverse
    _dist_to_cache = {}
    _graph_signature = _graph_sig(state)
    _claimed_by.clear()
    _unit_claim.clear()


def _ensure_graph(state: GraphState) -> None:
    if _graph_signature != _graph_sig(state):
        _rebuild_graph(state)


def _dijkstra(adjacency: Dict[int, List[Tuple[int, float]]], source: int) -> Dict[int, float]:
    dist: Dict[int, float] = {source: 0.0}
    heap = [(0.0, source)]
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, float('inf')):
            continue
        for v, w in adjacency.get(u, []):
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return dist


def _dist_to(target: int) -> Dict[int, float]:
    """Shortest-path distance from every node to `target`. Computed via
    Dijkstra on the reverse graph (one run gives distances *from* every
    node *to* the target) and cached, since the floor doesn't change
    during a run and many units will aim at the same handful of stations."""
    cached = _dist_to_cache.get(target)
    if cached is None:
        cached = _dijkstra(_reverse_adj, target)
        _dist_to_cache[target] = cached
    return cached


def _clean_claim(unit_id: int, state: GraphState) -> None:
    """Drop this unit's claim if the pod it was chasing was already picked
    up by someone else, delivered, or otherwise no longer waiting."""
    pod_id = _unit_claim.get(unit_id)
    if pod_id is None:
        return
    pod = state.get_pod(pod_id)
    if pod is None or pod.carried_by is not None or pod not in state.active_pods:
        _unit_claim.pop(unit_id, None)
        _claimed_by.pop(pod_id, None)


def _find_new_target_pod(unit_id: int, state: GraphState,
                          dist_from_unit: Dict[int, float]) -> Optional[str]:
    """Claim the closest currently-unclaimed waiting pod. Ties go to
    whichever pod has been waiting longest, so old pods don't get
    perpetually skipped in favor of new, slightly-closer ones."""
    best_pod = None
    best_key = None
    for pod in state.active_pods:
        if pod.carried_by is not None or pod.current_node is None:
            continue
        claimant = _claimed_by.get(pod.id)
        if claimant is not None and claimant != unit_id:
            continue
        d = dist_from_unit.get(pod.current_node)
        if d is None:
            continue
        key = (d, pod.entry_time, pod.id)
        if best_key is None or key < best_key:
            best_key = key
            best_pod = pod.id

    if best_pod is not None:
        _claimed_by[best_pod] = unit_id
        _unit_claim[unit_id] = best_pod
    return best_pod


def _pick_route_target(unit_id: int, state: GraphState) -> Optional[int]:
    """Decide which node this drive unit should be heading toward: the
    nearest destination among pods it's carrying, or the nearest pod it can
    still pick up, whichever is closer."""
    unit = state.get_drive_unit(unit_id)
    dist_from_unit = _dijkstra(_forward_adj, unit.current_node)

    delivery_target = None
    delivery_dist = float('inf')
    for pod_id in unit.carrying:
        pod = state.get_pod(pod_id)
        if pod is None:
            continue
        d = dist_from_unit.get(pod.destination_station, float('inf'))
        if d < delivery_dist:
            delivery_dist = d
            delivery_target = pod.destination_station

    pickup_target = None
    pickup_dist = float('inf')
    if unit.has_capacity:
        _clean_claim(unit_id, state)
        pod_id = _unit_claim.get(unit_id)
        if pod_id is None:
            pod_id = _find_new_target_pod(unit_id, state, dist_from_unit)
        if pod_id is not None:
            pod = state.get_pod(pod_id)
            if pod is not None and pod.current_node is not None:
                d = dist_from_unit.get(pod.current_node)
                if d is not None:
                    pickup_target = pod.current_node
                    pickup_dist = d

    if delivery_target is None:
        return pickup_target
    if pickup_target is None:
        return delivery_target
    # Ties favor delivering: an idle dock slot or free capacity is more
    # useful released sooner than gained sooner.
    return delivery_target if delivery_dist <= pickup_dist else pickup_target


def _vacate_if_blocking(unit, state: GraphState) -> Optional[int]:
    """
    An idle unit with nothing to do that happens to be parked on a
    capacity-limited node (a dock, a narrow parking spot) would otherwise
    sit there indefinitely, permanently occupying a slot other units need
    to deliver or pick up through. If that's the situation, step off onto
    any open, ideally unconstrained, neighbor instead of waiting in place.
    """
    node = state.get_node(unit.current_node)
    if node is None or node.capacity is None:
        return None  # not a contended resource - fine to just wait here

    best_neighbor = None
    best_constrained = True
    for neighbor, _weight in _forward_adj.get(unit.current_node, []):
        if not is_valid_move(state, unit, neighbor):
            continue
        neighbor_node = state.get_node(neighbor)
        constrained = bool(neighbor_node and neighbor_node.capacity is not None)
        if best_neighbor is None or constrained < best_constrained:
            best_neighbor = neighbor
            best_constrained = constrained
            if not constrained:
                break
    return best_neighbor


def _estimated_wait(state: GraphState, from_node: int, to_node: int) -> Optional[float]:
    """
    Rough lower-bound estimate of how many time steps until moving from
    from_node to to_node becomes valid, based on who is currently occupying
    the edge and the destination node.

    Returns 0 if the move is already open, a positive number of steps if
    we can see an ETA for it clearing, or None if there's no visible ETA
    (e.g. the destination is held by a unit that's parked rather than
    in transit - we can't know when a parked unit will move, so we treat
    that as "unknown" rather than guessing).
    """
    edge = state.get_edge(from_node, to_node)
    if edge is None:
        return None

    wait = 0.0

    if edge.capacity is not None and state.edge_occupancy(from_node, to_node) >= edge.capacity:
        remaining = [u.transit_remaining_time for u in state.drive_units
                     if u.in_transit and edge.connects(u.current_node, u.transit_destination)]
        if not remaining:
            return None
        wait = max(wait, min(remaining))

    node = state.get_node(to_node)
    if node is not None and node.capacity is not None and state.node_occupancy(to_node) >= node.capacity:
        inbound = [u.transit_remaining_time for u in state.drive_units
                   if u.in_transit and u.transit_destination == to_node]
        parked = any(not u.in_transit and u.current_node == to_node for u in state.drive_units)
        if not inbound and parked:
            return None  # blocked by a parked unit with no visible ETA
        if not inbound:
            return None
        wait = max(wait, min(inbound))

    return wait


def _next_hop(unit, target: int, state: GraphState) -> Optional[int]:
    """
    Pick the next node to move to, weighing open moves against waiting for
    a currently-blocked one. edge_occupancy/node_occupancy (via
    is_valid_move) only say whether a move is possible *this instant* -
    they don't say for how long a full aisle or dock stays full. To decide
    whether waiting beats detouring, we additionally look at the actual
    units causing the congestion and read their transit_remaining_time to
    estimate an ETA, then compare total costs:

        cost(open neighbor)    = edge.weight + dist_to_target[neighbor]
        cost(wait on neighbor) = estimated_wait + edge.weight + dist_to_target[neighbor]

    and take whichever is cheapest overall. If waiting for a blocked
    neighbor beats every currently-open option, we deliberately return
    None (wait) this step instead of settling for a worse detour.
    """
    dist_to_target = _dist_to(target)
    current = unit.current_node
    if current not in dist_to_target:
        return None

    best_open: Optional[Tuple[float, int]] = None
    best_wait: Optional[float] = None

    for neighbor, weight in _forward_adj.get(current, []):
        remaining = dist_to_target.get(neighbor)
        if remaining is None:
            continue
        cost = weight + remaining

        if is_valid_move(state, unit, neighbor):
            if best_open is None or cost < best_open[0]:
                best_open = (cost, neighbor)
        else:
            wait = _estimated_wait(state, current, neighbor)
            if wait is not None:
                total = wait + cost
                if best_wait is None or total < best_wait:
                    best_wait = total

    if best_open is not None and (best_wait is None or best_open[0] <= best_wait):
        return best_open[1]

    # Nothing open is as good as waiting for something blocked would be
    # (or nothing is open at all) - hold position and re-evaluate next step.
    return None


def drive_unit_next_move(drive_unit_id: int, state: GraphState) -> Optional[int]:
    """
    Determine the next node for a drive unit to move to.

    Pickups and deliveries are automatic: a drive unit with free capacity
    that stops at (or passes through) a node with a waiting pod picks it
    up, and a drive unit that reaches a carried pod's destination station
    drops it off.

    Args:
        drive_unit_id: ID of the drive unit being routed
        state: GraphState object containing the current state of the floor

    Returns:
        next_node_id: ID of an adjacent node to move to, or None to wait
                      at the current node
    """
    _ensure_graph(state)

    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    target = _pick_route_target(drive_unit_id, state)
    if target is None:
        return _vacate_if_blocking(unit, state)
    if target == unit.current_node:
        return None

    return _next_hop(unit, target, state)