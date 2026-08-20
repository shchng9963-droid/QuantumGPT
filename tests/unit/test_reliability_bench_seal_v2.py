"""Tests for encrypted method-validation set sealing."""

from __future__ import annotations

import io
import json
import tarfile

import pytest

from eval.reliability_bench.generator_v2 import DEVELOPMENT_CONFIG
from eval.reliability_bench.seal_v2 import (
    _decrypt_archive_for_verification,
    _read_secret,
    seal_method_validation_set,
)


def test_sealer_writes_only_ciphertext_and_manager_manifest(tmp_path):
    output = tmp_path / "sealed"
    key_path = tmp_path / "custodian" / "method-validation.key"
    manifest = seal_method_validation_set(
        output_dir=output,
        key_path=key_path,
        generator_commit="2c19a60",
        seal_commit="abc1234",
        authorized_custodians=["study_manager"],
        unseal_condition="judge v2 and ActionGuard are independently frozen",
        config=DEVELOPMENT_CONFIG,
        generated_at="2026-08-20T00:00:00+00:00",
    )
    assert manifest["episode_count"] == 24
    assert manifest["unseal_allowed"] is False
    assert manifest["status"] == "generated_sealed_not_unsealed"
    assert not (output / "episodes.jsonl").exists()
    assert {path.name for path in output.iterdir()} == {
        "method_validation_v2.tar.aesgcm",
        "sealed_manifest.json",
    }
    encrypted = (output / "method_validation_v2.tar.aesgcm").read_bytes()
    plaintext = _decrypt_archive_for_verification(encrypted, _read_secret(key_path))
    with tarfile.open(fileobj=io.BytesIO(plaintext), mode="r") as archive:
        assert set(archive.getnames()) == {
            "episodes.jsonl",
            "private_generation_manifest.json",
        }
        episodes = archive.extractfile("episodes.jsonl").read().splitlines()
        private = json.load(archive.extractfile("private_generation_manifest.json"))
    assert len(episodes) == 24
    assert private["config_sha256"] == manifest["config_sha256"]
    assert private["plaintext_dataset_sha256"] == manifest[
        "plaintext_dataset_sha256"
    ]


def test_sealer_refuses_overwrite_and_requires_custody_protocol(tmp_path):
    output = tmp_path / "sealed"
    key_path = tmp_path / "key"
    kwargs = {
        "output_dir": output,
        "key_path": key_path,
        "generator_commit": "2c19a60",
        "seal_commit": "abc1234",
        "authorized_custodians": ["study_manager"],
        "unseal_condition": "judge and method are frozen",
        "config": DEVELOPMENT_CONFIG,
    }
    seal_method_validation_set(**kwargs)
    with pytest.raises(FileExistsError):
        seal_method_validation_set(**kwargs)

    with pytest.raises(ValueError, match="custodian"):
        seal_method_validation_set(
            output_dir=tmp_path / "other",
            key_path=tmp_path / "other.key",
            generator_commit="2c19a60",
            seal_commit="abc1234",
            authorized_custodians=[],
            unseal_condition="judge and method are frozen",
            config=DEVELOPMENT_CONFIG,
        )
