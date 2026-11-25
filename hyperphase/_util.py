"""The loose planet-name matching, shared by every module that looks a planet up by name (bands, targets, noise, the bias tables and maps)."""


# Loose name matching ("K2-141b" == "K2-141 b" == "k2_141_b"), so a planet is found whichever spelling a table, a PlanetParams.name or the archive's
# pl_name carries. Any object is accepted (str() first): an archive column may hold a NaN name.
def _normalise(name):
    return str(name).lower().replace(" ", "").replace("-", "").replace("_", "")
