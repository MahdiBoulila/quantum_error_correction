"""Error sampling for a single noise site (one qubit, one location in a circuit)."""


def sample_error(p, kind, rng):
    """Return one of 'I', 'X', 'Y', 'Z' for a single noise site.

    kind:
      'X'            -- bit-flip channel: X w.p. p, else I
      'Z'            -- phase-flip channel: Z w.p. p, else I
      'depolarizing' -- X/Y/Z each w.p. p/3, else I
    """
    r = rng.random()
    if kind == "X":
        return "X" if r < p else "I"
    if kind == "Z":
        return "Z" if r < p else "I"
    if kind == "depolarizing":
        if r < p / 3:
            return "X"
        if r < 2 * p / 3:
            return "Y"
        if r < p:
            return "Z"
        return "I"
    raise ValueError(f"unknown noise kind: {kind}")
