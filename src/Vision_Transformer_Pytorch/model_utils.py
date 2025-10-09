"""Utility functions for model handling."""

import os
import glob
import logging

logger = logging.getLogger(__name__)


def resolve_model_path(path_pattern):
    """Resolve a model path pattern to an actual file path.

    Args:
        path_pattern: Path or glob pattern (e.g., "artifacts/results/gtsrb_*_best.pth")

    Returns:
        Resolved file path (most recent if multiple matches)

    Raises:
        FileNotFoundError: If no matching files found
    """
    # If pattern contains wildcards, use glob
    if '*' in path_pattern or '?' in path_pattern:
        matches = glob.glob(path_pattern)
        if not matches:
            raise FileNotFoundError(f"No files found matching pattern: {path_pattern}")
        # Return most recent file if multiple matches
        matches.sort(key=os.path.getmtime, reverse=True)
        resolved_path = matches[0]
        logger.info(f"Resolved pattern '{path_pattern}' to '{resolved_path}' ({len(matches)} matches found)")
        return resolved_path
    else:
        # Direct path - verify it exists
        if not os.path.exists(path_pattern):
            raise FileNotFoundError(f"Model file not found: {path_pattern}")
        return path_pattern
