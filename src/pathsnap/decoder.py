from collections import deque
import numpy as np


def _shortest_arc_distance(arcs: np.ndarray, source: int, target: int, max_hops=30):
    if source == target:
        return 0
    adj = {}
    for u, v in arcs:
        adj.setdefault(int(u), []).append(int(v))
    q = deque([(int(source), 0)])
    seen = {int(source)}
    while q:
        node, depth = q.popleft()
        if depth >= max_hops:
            continue
        for nxt in adj.get(node, []):
            if nxt == target:
                return depth + 1
            if nxt not in seen:
                seen.add(nxt)
                q.append((nxt, depth + 1))
    return None


def viterbi(candidate_ids, emission_scores, arcs, transition_scale=5.0, gap_break_s=None, timestamps=None):
    """Decode candidate arc sequences. Scores are positive emission weights."""
    if not candidate_ids:
        return []
    paths = [np.asarray(x, dtype=int) for x in candidate_ids]
    scores = [np.asarray(x, dtype=float) for x in emission_scores]
    if any(len(p) != len(s) or not len(p) for p, s in zip(paths, scores)):
        raise ValueError("Each observation needs equally sized, nonempty candidate and score lists")
    result = []
    start = 0
    for t in range(1, len(paths) + 1):
        split = t == len(paths) or (gap_break_s is not None and timestamps is not None and timestamps[t] - timestamps[t-1] > gap_break_s)
        if not split:
            continue
        result.extend(_decode_segment(paths[start:t], scores[start:t], arcs, transition_scale))
        start = t
    return result


def _decode_segment(paths, emissions, arcs, scale):
    dp = np.log(np.maximum(emissions[0], 1e-12))
    backs = []
    for i in range(1, len(paths)):
        cur = np.full(len(paths[i]), -np.inf)
        back = np.zeros(len(paths[i]), dtype=int)
        for j, target in enumerate(paths[i]):
            for k, source in enumerate(paths[i-1]):
                hops = _shortest_arc_distance(arcs, source, target)
                # Unreachable transitions retain a finite but strong penalty to avoid decoder failure.
                trans = -scale * (hops if hops is not None else 20)
                value = dp[k] + trans + np.log(max(emissions[i][j], 1e-12))
                if value > cur[j]:
                    cur[j], back[j] = value, k
        dp = cur
        backs.append(back)
    idx = int(np.argmax(dp))
    chosen = [idx]
    for back in reversed(backs):
        idx = int(back[idx]); chosen.append(idx)
    chosen.reverse()
    return [int(paths[i][j]) for i, j in enumerate(chosen)]
