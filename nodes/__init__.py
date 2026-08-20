import logging

# 2D node: legacy API
from .minimax_h3_latent_upscaler_2d import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_2D,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_2D,
)

NODE_CLASS_MAPPINGS = {}
NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_2D)

NODE_DISPLAY_NAME_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_2D)

# 3D node: legacy API (same registration mechanism as 2D)
try:
    from .minimax_h3_latent_upscaler_3d import (
        NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_3D,
        NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_3D,
    )
    NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_3D)
    NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_3D)
except Exception as e:
    logging.error(f"[MinimaxH3] Failed to import 3D node: {e}")

# H3-aware refinement is part of this package, so unexpected import or
# initialization failures must propagate instead of silently hiding the node.
from .minimax_h3_refine import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_REFINE,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_REFINE,
)
NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_REFINE)
NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_REFINE)

# Continuum list outputs must be consumed as one ordered sequence so sampler 2
# can carry the actual post-refine tail of chunk N into chunk N+1's protected
# Native-Masked prefix.  Register this wrapper last under the same stable node
# key so existing workflows gain seam-safe behavior without node replacement.
from .minimax_h3_refine_sequence import (
    NODE_CLASS_MAPPINGS as NODE_CLASS_MAPPINGS_REFINE_SEQUENCE,
    NODE_DISPLAY_NAME_MAPPINGS as NODE_DISPLAY_NAME_MAPPINGS_REFINE_SEQUENCE,
)
NODE_CLASS_MAPPINGS.update(NODE_CLASS_MAPPINGS_REFINE_SEQUENCE)
NODE_DISPLAY_NAME_MAPPINGS.update(NODE_DISPLAY_NAME_MAPPINGS_REFINE_SEQUENCE)

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
