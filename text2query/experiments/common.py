"""Shared runner for testing one query-generation agent at a time.

Each agent is an ``AgentStep``. A step reads the cohort text plus the gold
standard for every step in ``prior_step_ids``, and writes only its own
artifact:

    evaluation/gold/<prior_step_id>/<topic>.json
    evaluation/agent_outputs/<model_name>/<step_id>/<topic>.json

Prior files come from the gold tree. A run does not read another model's
``agent_outputs``. Decomposition is first and has an empty prior list.
Later scripts should use the same ``run_isolated_agent`` loop and pass the
parsed gold documents into that agent.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

TEXT2QUERY = Path(__file__).resolve().parent.parent
QUERIES_PATH = TEXT2QUERY / "evaluation" / "raw" / "queries.tsv"
OUTPUT_ROOT = TEXT2QUERY / "evaluation" / "agent_outputs"
GOLD_ROOT = TEXT2QUERY / "evaluation" / "gold"

_MODEL_DIR_UNSAFE = re.compile(r'[<>:"/\\|?*]')


@dataclass(frozen=True)
class AgentStep:
    """One agent, and the gold steps that must already exist before it runs."""

    step_id: str
    prior_step_ids: tuple[str, ...] = ()


# Order matches ReviewStep. Each entry's priors are the gold inputs required
# to score that agent without running the agents before it.
AGENT_STEPS: tuple[AgentStep, ...] = (
    AgentStep("01_decomposition"),
    AgentStep("02_queriability", ("01_decomposition",)),
    AgentStep("03_named_entities", ("02_queriability",)),
    AgentStep("04_entity_relations", ("03_named_entities",)),
    AgentStep("05_entity_linkages", ("04_entity_relations",)),
    AgentStep("06_compositional_traversal", ("05_entity_linkages",)),
    AgentStep("07_non_entity_constraints", ("06_compositional_traversal",)),
    AgentStep("08_structured_sql", ("07_non_entity_constraints",)),
    AgentStep("09_symbolic_nlp", ("07_non_entity_constraints",)),
    AgentStep("10_aggregate", ("08_structured_sql", "09_symbolic_nlp")),
)

DECOMPOSITION = AGENT_STEPS[0]


def step_by_id(step_id: str) -> AgentStep:
    for step in AGENT_STEPS:
        if step.step_id == step_id:
            return step
    known = ", ".join(step.step_id for step in AGENT_STEPS)
    raise KeyError(f"unknown agent step {step_id!r}; expected one of: {known}")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must be a JSON object")
    return {key: value for key, value in payload.items() if not str(key).startswith("_")}


def model_dir_name(model_name: str) -> str:
    """Single path segment for ``MODEL_NAME``."""
    cleaned = _MODEL_DIR_UNSAFE.sub("_", model_name.strip()).rstrip(" .")
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"MODEL_NAME {model_name!r} is not a usable directory name")
    return cleaned


def load_topics(path: Path) -> list[tuple[str, str]]:
    """Return ``(topic_id, query_text)`` for every row in the topics TSV."""
    topics: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        header = handle.readline()
        columns = header.rstrip("\r\n").split("\t")
        if columns[:2] != ["id", "query"]:
            raise ValueError(f"{path} must start with columns id and query")
        for lineno, raw in enumerate(handle, start=2):
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            topic_id, separator, text = line.partition("\t")
            if not separator or not topic_id.strip() or not text.strip():
                raise ValueError(f"{path}:{lineno} is missing an id or query")
            topics.append((f"{int(topic_id):02d}", text.strip()))
    return topics


def select_topics(
    topics: list[tuple[str, str]],
    *,
    limit: int | None,
) -> Iterator[tuple[str, str]]:
    produced = 0
    for topic_id, text in topics:
        if limit is not None and produced >= limit:
            return
        produced += 1
        yield topic_id, text


def gold_topic_path(gold_root: Path, step_id: str, topic_id: str) -> Path:
    return gold_root / step_id / f"{topic_id}.json"


def output_topic_path(output_root: Path, model_name: str, step: AgentStep, topic_id: str) -> Path:
    return output_root / model_dir_name(model_name) / step.step_id / f"{topic_id}.json"


def load_gold_priors(step: AgentStep, gold_root: Path, topic_id: str) -> dict[str, Any]:
    """Load this step's gold priors for one topic.

    Returns an empty dict when the step has no priors. Raises ``FileNotFoundError``
    if a required gold file is absent.
    """
    priors: dict[str, Any] = {}
    for prior_step_id in step.prior_step_ids:
        path = gold_topic_path(gold_root, prior_step_id, topic_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"{step.step_id} requires gold {prior_step_id} for topic {topic_id} at {path}"
            )
        with path.open(encoding="utf-8") as handle:
            priors[prior_step_id] = json.load(handle)
    return priors


def write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def run_isolated_agent(
    step: AgentStep,
    *,
    model_name: str,
    topics: list[tuple[str, str]],
    limit: int | None,
    output_root: Path,
    gold_root: Path,
    generator: Any,
    execute: Callable[[Any, str, str, dict[str, Any]], Any],
    log: logging.Logger,
) -> int:
    """Run ``execute`` once per topic with that topic's gold priors.

    ``execute(generator, topic_id, query_text, priors)`` returns the JSON
    document to write. ``priors`` is keyed by prior step id.
    """
    if step.prior_step_ids:
        log.info(
            "Isolated %s; gold priors: %s from %s",
            step.step_id,
            ", ".join(step.prior_step_ids),
            gold_root,
        )
    else:
        log.info("Isolated %s; this step has no prior agent", step.step_id)

    succeeded = 0
    failed = 0
    for index, (topic_id, text) in enumerate(select_topics(topics, limit=limit), start=1):
        destination = output_topic_path(output_root, model_name, step, topic_id)
        log.info("[%d] %s topic=%s", index, step.step_id, topic_id)
        try:
            priors = load_gold_priors(step, gold_root, topic_id)
            document = execute(generator, topic_id, text, priors)
            write_json(destination, document)
        except Exception:
            failed += 1
            log.exception("%s failed for topic=%s", step.step_id, topic_id)
            continue
        succeeded += 1

    log.info(
        "Done. succeeded=%d failed=%d. Outputs in %s",
        succeeded,
        failed,
        output_root / model_dir_name(model_name) / step.step_id,
    )
    return 1 if failed else 0
