"""
Tear/eye-wetness observation -- deliberately NOT a "is this person crying"
classifier. Facial landmarks alone cannot reliably determine tear presence
(reflections, lighting, and eye moisture are visually similar to many other
things), and no verified tear/wetness dataset exists anywhere in this
repository to train a real detector on.

This module is the documented, optional interface the spec asks for. It
currently always returns "not_trained" -- it does not run any model, and it
must never be wired to output a crying/emotion claim.
"""
from typing import Any, Dict


def analyze_tear_presence(frames) -> Dict[str, Any]:
    """
    Placeholder interface for a future tear/eye-wetness detector. Returns
    "not_trained" honestly rather than a fabricated always-false value,
    since "not trained" and "confidently detected no tears" are different
    claims and only the former is true right now.

    A real implementation, if one is ever built, should output soft,
    literal observational terms -- e.g. "tear_like_visual_presence",
    "eye_wetness", "tear_streak_observation" -- never "person_is_crying".
    """
    return {
        "status": "not_trained",
        "tear_like_visual_presence": "not_available",
        "eye_wetness": "not_available",
        "tear_streak_observation": "not_available",
        "note": "No verified tear/wetness dataset exists in this repository; this is an unimplemented, optional module.",
    }
