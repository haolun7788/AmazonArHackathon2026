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
import math

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
_last_assign_time: int = -1
_reservations: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
_last_res_cleanup: int = -1


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


def _cleanup_reservations(state: GraphState) -> None:
    """Remove expired reservations up to the current time step."""
    global _reservations, _last_res_cleanup
    t = state.current_time_step
    if t == _last_res_cleanup:
        return
    for edge, slots in list(_reservations.items()):
        new_slots = [(s, e) for (s, e) in slots if e > t]
        if new_slots:
            _reservations[edge] = new_slots
        else:
            _reservations.pop(edge, None)
    _last_res_cleanup = t


def _is_reserved(from_node: int, to_node: int, start: int, end: int) -> bool:
    """Check if edge (from_node,to_node) has any reservation overlapping [start,end)."""
    key = (from_node, to_node)
    slots = _reservations.get(key, [])
    for s, e in slots:
        if not (e <= start or s >= end):
            return True
    # also check reverse direction if edge is bidirectional
    rev = (to_node, from_node)
    slots = _reservations.get(rev, [])
    for s, e in slots:
        if not (e <= start or s >= end):
            return True
    return False


def _reserve_edge(from_node: int, to_node: int, start: int, end: int) -> None:
    key = (from_node, to_node)
    _reservations.setdefault(key, []).append((start, end))


def _a_star(start: int, target: int) -> float:
    """A* search from start to target using cached reverse distances as
    an admissible heuristic (if available). Returns the path cost or
    float('inf') if target is unreachable.
    """
    if start == target:
        return 0.0
    # heuristic: shortest known distance from node -> target
    dist_to_target = _dist_to(target)

    g_score: Dict[int, float] = {start: 0.0}
    heap = [(g_score[start] + dist_to_target.get(start, 0.0), start)]

    while heap:
        f, u = heapq.heappop(heap)
        if u == target:
            return g_score[u]
        gu = g_score.get(u, float('inf'))
        for v, w in _forward_adj.get(u, []):
            ng = gu + w
            if ng < g_score.get(v, float('inf')):
                g_score[v] = ng
                heapq.heappush(heap, (ng + dist_to_target.get(v, 0.0), v))

    return float('inf')


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


def _global_greedy_assign(state: GraphState) -> None:
    """Globally assign waiting pods to available units using a greedy
    nearest-first strategy. Runs once per simulation timestep to avoid
    conflicting per-unit claims and improve global efficiency.
    """
    global _claimed_by, _unit_claim
    # collect candidate pods and units
    pods = [p for p in state.active_pods if p.current_node is not None and p.carried_by is None]
    units = [u for u in state.drive_units if not u.in_transit and u.has_capacity]
    if not pods or not units:
        _claimed_by.clear()
        _unit_claim.clear()
        return

    # compute per-unit shortest-path distances
    dist_map: Dict[int, Dict[int, float]] = {}
    for u in units:
        dist_map[u.id] = _dijkstra(_forward_adj, u.current_node)

    # build all (distance, entry_time, unit_id, pod_id) tuples so older
    # pods (smaller entry_time) are preferred in ties and near ties.
    triples: List[Tuple[float, int, int, str]] = []
    for u in units:
        dmap = dist_map.get(u.id, {})
        for p in pods:
            d = dmap.get(p.current_node)
            if d is None:
                continue
            triples.append((d, p.entry_time, u.id, p.id))

    triples.sort()

    assigned_units = set()
    assigned_pods = set()
    _claimed_by.clear()
    _unit_claim.clear()

    for d, entry_time, uid, pid in triples:
        if uid in assigned_units or pid in assigned_pods:
            continue
        assigned_units.add(uid)
        assigned_pods.add(pid)
        _claimed_by[pid] = uid
        _unit_claim[uid] = pid



def _pick_route_target(unit_id: int, state: GraphState) -> Optional[int]:
    """Decide which node this drive unit should be heading toward: the
    nearest destination among pods it's carrying, or the nearest pod it can
    still pick up, whichever is closer."""
    unit = state.get_drive_unit(unit_id)

    # Prefer delivering the first-picked pod (FIFO) to avoid target
    # oscillation; use A* to estimate source->target distance.
    delivery_target = None
    delivery_dist = float('inf')
    if unit.carrying:
        first_pod = state.get_pod(unit.carrying[0])
        if first_pod is not None:
            delivery_target = first_pod.destination_station
            delivery_dist = _a_star(unit.current_node, delivery_target)

    pickup_target = None
    pickup_dist = float('inf')
    if unit.has_capacity:
        _clean_claim(unit_id, state)
        pod_id = _unit_claim.get(unit_id)
        if pod_id is None:
            # no claim from global assign; compute a full distance map and
            # claim the nearest pod using the existing helper.
            dist_from_unit = _dijkstra(_forward_adj, unit.current_node)
            pod_id = _find_new_target_pod(unit_id, state, dist_from_unit)
        if pod_id is not None:
            pod = state.get_pod(pod_id)
            if pod is not None and pod.current_node is not None:
                pickup_target = pod.current_node
                pickup_dist = _a_star(unit.current_node, pickup_target)

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

    def _reservation_wait(from_node: int, to_node: int, start_time: int, weight: float) -> Optional[float]:
        """Return how many steps to wait until edge [from_node->to_node]
        is free for a window of length `ceil(weight)`. Returns 0 if free,
        positive number of steps if there's a visible ETA, or None if
        unknown.
        """
        st = start_time
        dur = math.ceil(weight)
        end = st + dur
        key = (from_node, to_node)
        slots = _reservations.get(key, []) + _reservations.get((to_node, from_node), [])
        overlapping = [ (s,e) for (s,e) in slots if not (e <= st or s >= end) ]
        if not overlapping:
            return 0.0
        # earliest time when none of the overlapping reservations remain
        latest_end = max(e for (_s,e) in overlapping)
        return float(max(0, latest_end - st))

    t = state.current_time_step
    for neighbor, weight in _forward_adj.get(current, []):
        remaining = dist_to_target.get(neighbor)
        if remaining is None:
            continue
        cost = weight + remaining

        # Check reservation and validity
        reserved_wait = _reservation_wait(current, neighbor, t, weight)
        move_open = is_valid_move(state, unit, neighbor) and (reserved_wait == 0.0)

        if move_open:
            if best_open is None or cost < best_open[0]:
                best_open = (cost, neighbor)
        else:
            wait1 = _estimated_wait(state, current, neighbor)
            wait2 = reserved_wait
            # if either estimate is None, we treat overall wait as unknown
            if wait1 is None and wait2 is None:
                continue
            waits = [w for w in (wait1, wait2) if w is not None]
            if not waits:
                continue
            total = min(waits) + cost
            if best_wait is None or total < best_wait:
                best_wait = total

    if best_open is not None and (best_wait is None or best_open[0] <= best_wait):
        # Reserve the chosen edge for the unit for the upcoming time window
        neighbor = best_open[1]
        edge = state.get_edge(current, neighbor)
        if edge is not None:
            dur = math.ceil(edge.weight)
            _reserve_edge(current, neighbor, state.current_time_step, state.current_time_step + dur)
        return neighbor

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

    global _last_assign_time
    # Run a single global greedy assignment once per timestep to claim pods
    if state.current_time_step != _last_assign_time:
        _last_assign_time = state.current_time_step
        for uid in (u.id for u in state.drive_units):
            _clean_claim(uid, state)
        _global_greedy_assign(state)
        _cleanup_reservations(state)

    unit = state.get_drive_unit(drive_unit_id)
    if unit is None or unit.in_transit:
        return None

    target = _pick_route_target(drive_unit_id, state)
    if target is None:
        return _vacate_if_blocking(unit, state)
    if target == unit.current_node:
        return None

    return _next_hop(unit, target, state)