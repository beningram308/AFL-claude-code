from afl_bot.data.venues import VENUE_METADATA, is_roofed, venue_info


def test_venue_info_known_and_unknown():
    mcg = venue_info("M.C.G.")
    assert mcg is not None and mcg["city"] == "Melbourne"
    assert venue_info("Nonexistent Ground") is None


def test_is_roofed():
    assert is_roofed("Marvel Stadium") is True
    assert is_roofed("Docklands") is True       # same ground, alternate name
    assert is_roofed("M.C.G.") is False
    assert is_roofed("Unknown Ground") is False  # unknown -> treated open-air


def test_all_metadata_well_formed():
    for venue, info in VENUE_METADATA.items():
        assert {"city", "lat", "lon", "roofed"} <= info.keys()
        assert -45 <= info["lat"] <= 35      # Australia .. Shanghai
        assert 110 <= info["lon"] <= 155 or 120 <= info["lon"] <= 122  # AU or Shanghai
        assert isinstance(info["roofed"], bool)
