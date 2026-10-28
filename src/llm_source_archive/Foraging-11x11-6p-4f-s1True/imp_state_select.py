import numpy as np

def select_important_state():
    """
    Returns the indices of the most important state dimensions for coordinated
    multi-agent food collection in LBF, for the Foraging-11x11-6p-4f-s1True config
    (field_size=11, sight=1, grid_observation=True).

    Generalized from the working Foraging-8x8-2p-2f-s1True version of this file.
    That version assumed an (8+2*sight) x (8+2*sight) = 10x10 padded grid per layer
    (agent, food, access), with a 1-cell border of padding on each side (border
    width = sight). This version applies the same formula with field_size=11,
    giving a (11+2*1) x (11+2*1) = 13x13 = 169-cell padded grid per layer instead
    of the 8x8 file's 100-cell layer, and layer offsets of 0 / 169 / 338 instead
    of 0 / 100 / 200.

    - Agent layer (field): valid cells only (border padding excluded)
    - Food layer (field): offset by one full layer (169)
    - Access layer (field): offset by two full layers (338)

    IMPORTANT: this is a reconstruction based on pattern-matching the 8x8 file's
    formula, not verified against the actual env wrapper source. Verify the
    computed indices are all within the real observation vector length before
    trusting this for a full training run (see accompanying verification snippet).
    """
    field_size = 11
    sight = 1
    padded_size = field_size + 2 * sight  # 13
    layer_size = padded_size * padded_size  # 169

    important_dims = []

    # Agent layer: valid (non-border) cells, rows/cols 1..field_size
    for row in range(1, field_size + 1):
        important_dims.extend(
            range(row * padded_size + 1, row * padded_size + field_size + 1)
        )

    # Food layer: same pattern, offset by one full layer
    for row in range(1, field_size + 1):
        important_dims.extend(
            range(layer_size + row * padded_size + 1,
                  layer_size + row * padded_size + field_size + 1)
        )

    # Access layer: same pattern, offset by two full layers
    for row in range(1, field_size + 1):
        important_dims.extend(
            range(2 * layer_size + row * padded_size + 1,
                  2 * layer_size + row * padded_size + field_size + 1)
        )

    return important_dims
