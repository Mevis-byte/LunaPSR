"""AI enhancement model stubs.

These classes define the *interface* for future learned models.  We do NOT
download or ship model weights in Version 1.  Selecting a stub model in the UI
reports that it is not yet available, which keeps the app honest.

Planned models:

* Zero-DCE / Zero-DCE++  (MAPLE-Lab)
* Retinexformer          (low-light enhancement transformer)
* PSR-physics-aware model (future, LunaPSR-specific)
"""

from __future__ import annotations

__all__ = ["RetinexformerModel", "ZeroDCEModel"]