# tests/unit/test_model_service.py

from model_service.main import create_signature


def test_create_signature_matches_known_hmac_sha256_vector() -> None:
    """
    Verify signature generation against a known HMAC-SHA256 value.

    Test vector:
        secret:  key
        message: The quick brown fox jumps over the lazy dog
    """
    payload = b"The quick brown fox jumps over the lazy dog"
    secret = "key"

    signature = create_signature(
        payload_bytes=payload,
        secret=secret,
    )

    assert signature == (
        "sha256="
        "f7bc83f430538424b13298e6aa6fb143"
        "ef4d59a14946175997479dbc2d1a3cd8"
    )


def test_create_signature_changes_when_payload_changes() -> None:
    """
    A modified request body must produce a different signature.
    """
    secret = "development-shared-secret"

    original_signature = create_signature(
        payload_bytes=b'{"status":"original"}',
        secret=secret,
    )

    modified_signature = create_signature(
        payload_bytes=b'{"status":"modified"}',
        secret=secret,
    )

    assert original_signature != modified_signature


def test_create_signature_changes_when_secret_changes() -> None:
    """
    The same payload signed with different secrets must not produce
    the same signature.
    """
    payload = b'{"report_id":"drift_report_001"}'

    first_signature = create_signature(
        payload_bytes=payload,
        secret="first-secret",
    )

    second_signature = create_signature(
        payload_bytes=payload,
        secret="second-secret",
    )

    assert first_signature != second_signature


def test_create_signature_uses_expected_prefix() -> None:
    """
    The wire contract requires signatures to use the sha256=<digest> form.
    """
    signature = create_signature(
        payload_bytes=b"payload",
        secret="secret",
    )

    assert signature.startswith("sha256=")
    assert len(signature) == len("sha256=") + 64
