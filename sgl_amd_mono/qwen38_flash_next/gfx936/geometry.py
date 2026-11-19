"""Row geometry shared by the mono kernels, host wrappers, and the MTP contract.

``MAX_ROWS`` is the fused-kernel row capacity (kernel.cpp ``MaxRows``): plain
decode uses one row per request, and a TARGET_VERIFY pass uses ``DRAFT_TOKENS``
rows per request (the draft chain, evaluated front-packed exactly like the
native verify path). Decode therefore serves at most ``MAX_REQUESTS`` requests
per fused launch, the same bound the decode graph captures.
"""

MAX_ROWS = 8
DRAFT_TOKENS = 4
MAX_REQUESTS = MAX_ROWS // DRAFT_TOKENS
