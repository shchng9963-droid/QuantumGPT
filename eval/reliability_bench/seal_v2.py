"""Generate and seal the held-out method-validation set without plaintext output."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from . import generator_v2, predicates_v2, schema
from .generator_v2 import (
    GENERATOR_VERSION,
    GeneratorV2Config,
    canonical_config,
    config_sha256,
    generate_method_validation_candidates,
)


SEAL_VERSION = "reliabilitybench-q/method-validation-seal-1.1"
VALIDATION_SET_VERSION = "reliabilitybench-q/method-validation-v2.1"
ARCHIVE_MAGIC = b"RBQ-AES256-GCM-V1\x00"
ARCHIVE_AAD = VALIDATION_SET_VERSION.encode("utf-8")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_hash(module: Any) -> str:
    return _sha256_bytes(Path(module.__file__).read_bytes())


def _dataset_bytes(config: GeneratorV2Config) -> tuple[bytes, int]:
    episodes = generate_method_validation_candidates(config)
    payload = (
        "\n".join(_canonical_json(episode.to_dict()) for episode in episodes) + "\n"
    ).encode("utf-8")
    return payload, len(episodes)


def _tar_bytes(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, payload in sorted(files.items()):
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            info.mtime = 0
            info.mode = 0o600
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    return output.getvalue()


def _write_new_secret(path: Path, secret: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, base64.urlsafe_b64encode(secret) + b"\n")
    finally:
        os.close(descriptor)


def _read_secret(path: Path) -> bytes:
    return base64.urlsafe_b64decode(path.read_bytes().strip())


def _decrypt_archive_for_verification(archive: bytes, key: bytes) -> bytes:
    if not archive.startswith(ARCHIVE_MAGIC):
        raise ValueError("unknown sealed archive format")
    offset = len(ARCHIVE_MAGIC)
    nonce = archive[offset : offset + 12]
    ciphertext = archive[offset + 12 :]
    return AESGCM(key).decrypt(nonce, ciphertext, ARCHIVE_AAD)


def seal_method_validation_set(
    *,
    output_dir: Path,
    key_path: Path,
    generator_commit: str,
    seal_commit: str,
    authorized_custodians: list[str],
    unseal_condition: str,
    config: GeneratorV2Config,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if not generator_commit.strip():
        raise ValueError("generator_commit is required")
    if not seal_commit.strip():
        raise ValueError("seal_commit is required")
    if not authorized_custodians or any(
        not item.strip() for item in authorized_custodians
    ):
        raise ValueError("at least one authorized custodian is required")
    if not unseal_condition.strip():
        raise ValueError("unseal_condition is required")
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)

    dataset, episode_count = _dataset_bytes(config)
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()
    source_hashes = {
        "generator_v2_sha256": _source_hash(generator_v2),
        "schema_sha256": _source_hash(schema),
        "seal_v2_sha256": _sha256_bytes(Path(__file__).read_bytes()),
        "task_predicate_sha256": _source_hash(predicates_v2),
    }
    private_manifest = {
        "validation_set_version": VALIDATION_SET_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_commit": generator_commit,
        "seal_commit": seal_commit,
        "schema_version": schema.LATEST_SCHEMA_VERSION,
        "config": canonical_config(config),
        "config_sha256": config_sha256(config),
        "random_seed": config.random_seed,
        "episode_count": episode_count,
        "plaintext_dataset_sha256": _sha256_bytes(dataset),
        "generation_timestamp": timestamp,
        **source_hashes,
    }
    plaintext_archive = _tar_bytes(
        {
            "episodes.jsonl": dataset,
            "private_generation_manifest.json": (
                json.dumps(
                    private_manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8"),
        }
    )
    key = AESGCM.generate_key(bit_length=256)
    nonce = os.urandom(12)
    encrypted = ARCHIVE_MAGIC + nonce + AESGCM(key).encrypt(
        nonce, plaintext_archive, ARCHIVE_AAD
    )
    archive_path = output_dir / "method_validation_v2.tar.aesgcm"
    archive_path.write_bytes(encrypted)
    os.chmod(archive_path, 0o600)
    _write_new_secret(key_path, key)

    manifest = {
        "seal_version": SEAL_VERSION,
        "validation_set_version": VALIDATION_SET_VERSION,
        "status": "generated_sealed_not_unsealed",
        "access_class": "manager_only",
        "generator_version": GENERATOR_VERSION,
        "generator_commit": generator_commit,
        "seal_commit": seal_commit,
        "schema_version": schema.LATEST_SCHEMA_VERSION,
        "config_sha256": config_sha256(config),
        "random_seed": config.random_seed,
        "task_predicate_sha256": source_hashes["task_predicate_sha256"],
        "generator_source_sha256": source_hashes["generator_v2_sha256"],
        "schema_source_sha256": source_hashes["schema_sha256"],
        "seal_source_sha256": source_hashes["seal_v2_sha256"],
        "episode_count": episode_count,
        "encrypted_archive_sha256": _sha256_bytes(encrypted),
        "plaintext_dataset_sha256": _sha256_bytes(dataset),
        "generation_timestamp": timestamp,
        "authorized_custodians": authorized_custodians,
        "unseal_condition": unseal_condition,
        "unseal_allowed": False,
        "encryption": {
            "algorithm": "AES-256-GCM",
            "archive_format": "deterministic-tar-before-encryption",
            "associated_data": VALIDATION_SET_VERSION,
            "key_separated_from_archive": True,
        },
        "preregistered_constraints": {
            "action_guard_hidden_fields": [
                "judge",
                "task_predicate",
                "acceptable_action_set",
                "ground_truth",
            ],
            "substantive_bug_policy": (
                "upgrade generator/schema/predicate version and regenerate the entire "
                "sealed validation set; never patch selected episodes"
            ),
            "judge_audit_retry_policy": (
                "after a failed independent audit, use a newly sampled audit set and "
                "do not change preregistered thresholds after viewing results"
            ),
            "legacy_b1_status": "development-only / invalidated judge v1",
        },
    }
    manifest_path = output_dir / "sealed_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(manifest_path, 0o600)
    return manifest


__all__ = [
    "ARCHIVE_AAD",
    "ARCHIVE_MAGIC",
    "SEAL_VERSION",
    "VALIDATION_SET_VERSION",
    "seal_method_validation_set",
]
