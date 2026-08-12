"""Compatibility entry point for drift-detector evaluation.

The canonical evaluation implementation lives in
:mod:`agent.drift_detection.evaluate`.
"""

from agent.drift_detection.evaluate import *  # noqa: F401,F403
from agent.drift_detection.evaluate import main


if __name__ == "__main__":
    main()
