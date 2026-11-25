#!/usr/bin/env python3
"""Run the decomposition agent on the 56 cohort topics.

Decomposition is the first agent, so it takes no gold prior. The shared
runner still treats it as an isolated step: later agents load gold from
``evaluation/gold/<prior_step>/<topic>.json`` and write beside this output.

Reads ``text2query/evaluation/raw/queries.tsv`` (the format-B topic text)
and writes one strict decomposition tree per topic:

    text2query/evaluation/agent_outputs/<model_name>/01_decomposition/<topic>.json

Topic ids are zero-padded to two digits (``01.json`` through ``56.json``).
Pipeline fields such as ``clause_uuid`` are omitted.

Usage (from the repository root)::

    python text2query/experiments/01_run_decomposition.py --config config.json

``config.json`` is the implementation model config (``MODEL_NAME``,
``MODEL_PROVIDER``, and that provider's credentials). The concept catalog
is not loaded.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

_TEXT2QUERY = Path(__file__).resolve().parent.parent
_IMPLEMENTATION = _TEXT2QUERY / "implementation"
if str(_IMPLEMENTATION) not in sys.path:
    sys.path.insert(0, str(_IMPLEMENTATION))

_EXPERIMENTS = Path(__file__).resolve().parent
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from pydantic import TypeAdapter

from common import (  # noqa: E402
    DECOMPOSITION,
    GOLD_ROOT,
    OUTPUT_ROOT,
    QUERIES_PATH,
    load_config,
    load_topics,
    run_isolated_agent,
)
from phoenix_ehr.core.querying.generation.generator_structs import (  # noqa: E402
    StrictDecompositionNode,
)
from phoenix_ehr.core.querying.generation.query_generator import QueryGenerator  # noqa: E402

_STRICT = TypeAdapter(StrictDecompositionNode)


def decomposition_document(root: Any) -> dict[str, Any]:
    """Strict annotation tree, without pipeline enrichment fields."""
    payload = root.model_dump(mode="python") if hasattr(root, "model_dump") else root
    document = _STRICT.validate_python(payload).model_dump(mode="json", exclude_none=True)
    return _whole_numbers(document)


def _whole_numbers(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _whole_numbers(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_whole_numbers(value) for value in node]
    if isinstance(node, float) and node.is_integer():
        return int(node)
    return node


def _execute(generator: QueryGenerator, topic_id: str, text: str, priors: dict[str, Any]) -> dict[str, Any]:
    if priors:
        raise RuntimeError(
            f"decomposition topic {topic_id} received prior gold {sorted(priors)}; "
            "this step has no prior agent"
        )
    return decomposition_document(generator.decompose(text))


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("experiments.01_run_decomposition")

    parser = argparse.ArgumentParser(
        description=(
            "Run decomposition in isolation on the 56 cohort topics and write "
            "evaluation/agent_outputs/<model_name>/01_decomposition/<topic>.json."
        )
    )
    parser.add_argument("--config", required=True, type=Path, help="Path to config.json.")
    parser.add_argument(
        "--queries",
        type=Path,
        default=QUERIES_PATH,
        help=f"Topic TSV (default: {QUERIES_PATH}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output root (default: {OUTPUT_ROOT}).",
    )
    parser.add_argument(
        "--gold-root",
        type=Path,
        default=GOLD_ROOT,
        help=(
            "Gold root used by later agents "
            f"(default: {GOLD_ROOT}). Decomposition does not read it."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of topics to run.",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    model_name = str(cfg.get("MODEL_NAME") or "").strip()
    if not model_name:
        raise SystemExit(f"{args.config} is missing MODEL_NAME")

    topics = load_topics(args.queries)
    log.info("Loaded %d topics from %s", len(topics), args.queries)
    generator = QueryGenerator.for_decomposition(**cfg)
    return run_isolated_agent(
        DECOMPOSITION,
        model_name=model_name,
        topics=topics,
        limit=args.limit,
        output_root=args.output_root,
        gold_root=args.gold_root,
        generator=generator,
        execute=_execute,
        log=log,
    )


if __name__ == "__main__":
    sys.exit(main())
