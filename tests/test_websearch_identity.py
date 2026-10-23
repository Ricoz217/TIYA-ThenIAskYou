from pathlib import Path

import pytest

from TIYA.agent_tools.websearch.tools import (
    BrowseRequest,
    FetchRequest,
    IdentityRef,
    IdentityStore,
    MapRequest,
)


def test_identity_store_resolves_profiles_below_its_root(tmp_path: Path) -> None:
    store = IdentityStore(tmp_path / "profiles")
    identity = IdentityRef("github-rico")

    path = store.resolve(identity, create=True)

    assert path == (tmp_path / "profiles" / "github-rico").resolve()
    assert path.is_dir()


@pytest.mark.parametrize("profile_id", ["", "../escape", "a/b", "a\\b", "."])
def test_identity_ref_rejects_unsafe_profile_ids(profile_id: str) -> None:
    with pytest.raises(ValueError):
        IdentityRef(profile_id)


def test_web_requests_carry_an_opaque_identity_reference() -> None:
    identity = IdentityRef("signed-in")

    assert FetchRequest("https://example.com", identity=identity).identity is identity
    assert MapRequest("https://example.com", identity=identity).identity is identity
    request = BrowseRequest(
        "https://example.com", identity=identity, visible=True
    )
    assert request.identity is identity
    assert request.visible is True
