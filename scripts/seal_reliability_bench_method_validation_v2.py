#!/usr/bin/env python3
"""Generate and encrypt the frozen ReliabilityBench-Q method-validation set."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.reliability_bench.generator_v2 import GeneratorV2Config  # noqa: E402
from eval.reliability_bench.seal_v2 import seal_method_validation_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--key-out", type=Path, required=True)
    parser.add_argument("--generator-commit", required=True)
    parser.add_argument("--seal-commit", required=True)
    parser.add_argument("--custodian", action="append", required=True)
    parser.add_argument("--unseal-condition", required=True)
    parser.add_argument("--generated-at")
    args = parser.parse_args()
    config = GeneratorV2Config(random_seed=secrets.randbits(63))
    manifest = seal_method_validation_set(
        output_dir=args.out,
        key_path=args.key_out,
        generator_commit=args.generator_commit,
        seal_commit=args.seal_commit,
        authorized_custodians=args.custodian,
        unseal_condition=args.unseal_condition,
        config=config,
        generated_at=args.generated_at,
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "validation_set_version": manifest["validation_set_version"],
                "episode_count": manifest["episode_count"],
                "config_sha256": manifest["config_sha256"],
                "encrypted_archive_sha256": manifest["encrypted_archive_sha256"],
                "unseal_allowed": manifest["unseal_allowed"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
