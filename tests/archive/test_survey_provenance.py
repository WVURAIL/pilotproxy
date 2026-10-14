"""Public certificate identity is distinct from its private-key container."""
import datetime as dt
import hashlib
import json

from pilot_proxy.archive.survey_provenance import runtime_provenance


def test_public_certificate_provenance_excludes_private_key(tmp_path, monkeypatch):
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "local-survey-test")])
    expiry = dt.datetime(2030, 1, 2, tzinfo=dt.timezone.utc)
    cert = (
        x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(1)
        .not_valid_before(dt.datetime(2029, 1, 1, tzinfo=dt.timezone.utc))
        .not_valid_after(expiry).sign(key, hashes.SHA256())
    )
    pem = cert.public_bytes(serialization.Encoding.PEM) + key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption())
    path = tmp_path / "proxy.pem"
    path.write_bytes(pem)
    monkeypatch.setenv("CADC_CERT", str(path))
    result = runtime_provenance()
    assert result["certificate_status"] == "read"
    assert result["certificate_not_after"] == expiry.isoformat()
    assert result["certificate_sha256"] == cert.fingerprint(hashes.SHA256()).hex()
    assert result["certificate_sha256"] != hashlib.sha256(pem).hexdigest()
    assert "PRIVATE KEY" not in json.dumps(result)


def test_unavailable_certificate_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("CADC_CERT", str(tmp_path / "absent.pem"))
    result = runtime_provenance()
    assert result["certificate_status"] == "unavailable"
    assert result["certificate_not_after"] is None
    assert result["certificate_sha256"] is None
