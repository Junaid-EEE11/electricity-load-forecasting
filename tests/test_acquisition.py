from __future__ import annotations

import re

from prepare_victoria_data import SOURCES, UPSTREAM_COMMIT


def test_public_data_sources_are_immutable_and_checksum_pinned() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}", UPSTREAM_COMMIT)
    for source in SOURCES.values():
        assert UPSTREAM_COMMIT in str(source["url"])
        assert re.fullmatch(r"[0-9a-f]{64}", str(source["sha256"]))
