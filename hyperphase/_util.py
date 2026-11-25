"""Loose planet-name matching, shared by every module that looks a planet up by name."""


def _normalise(name):
    """Key for loose name matching ("K2-141b" == "K2-141 b" == "k2_141_b"): lower case without spaces, hyphens and underscores, so a planet is
    found whichever spelling a table, a PlanetParams.name or the archive's pl_name carries. Any object is accepted (str() first): an archive
    column may hold a NaN name."""
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")
