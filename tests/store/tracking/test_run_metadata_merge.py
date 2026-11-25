from __future__ import annotations

import glob
import json
import math
import multiprocessing
import os
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from mlflow.entities import Metric, Param, RunTag, ViewType
from mlflow.environment_variables import MLFLOW_ALLOW_FILE_STORE
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.store.tracking.file_store import FileStore
from mlflow.store.tracking.run_metadata_merge import (
    FileStoreRunMerger,
    MergeConflict,
    RunMetadataDelta,
    compute_delta_digest,
)
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_MODELS, MLFLOW_RUN_NAME


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FileStore:
    monkeypatch.setenv(MLFLOW_ALLOW_FILE_STORE.name, "true")
    return FileStore(str(tmp_path / "mlruns"))


@pytest.fixture
def merger(store: FileStore) -> FileStoreRunMerger:
    return FileStoreRunMerger(store, lock_timeout_seconds=10.0)


def _create_run(store: FileStore, name: str = "run") -> str:
    existing = store.get_experiment_by_name("exp")
    experiment_id = (
        existing.experiment_id if existing is not None else store.create_experiment("exp")
    )
    return store.create_run(
        experiment_id, user_id="user", start_time=0, tags=[], run_name=name
    ).info.run_id


def _delta(
    source_id: str,
    produced_at_ms: int,
    *,
    params: tuple[Param, ...] = (),
    tags: tuple[RunTag, ...] = (),
    metrics: tuple[Metric, ...] = (),
    status: str | None = None,
    end_time: int | None = None,
) -> RunMetadataDelta:
    return RunMetadataDelta(
        source_id=source_id,
        produced_at_ms=produced_at_ms,
        params=params,
        tags=tags,
        metrics=metrics,
        status=status,
        end_time=end_time,
    )


# ---------------------------------------------------------------------------
# Basic merge, replay and revision semantics
# ---------------------------------------------------------------------------


def test_merge_applies_delta_and_creates_journal_and_lock(store, merger, tmp_path):
    run_id = _create_run(store)
    delta = _delta(
        "s1",
        100,
        params=(Param("p", "1"),),
        tags=(RunTag("t", "a"),),
        metrics=(Metric("m", 1.0, 1, 0),),
    )

    outcome = merger.merge(run_id, [delta])

    assert outcome.revision == 1
    assert outcome.applied_sources == ("s1",)
    assert outcome.replayed_sources == ()
    assert outcome.rejected_sources == ()
    assert outcome.conflicts == ()
    assert outcome.param_writes == 1
    assert outcome.tag_writes == 1
    assert outcome.metric_writes == 1

    run_dir = Path(store.resolve_run_dir(run_id))
    assert (run_dir / "merge_state.json").exists()
    assert (run_dir / ".merge.lock").exists()
    # New files never pollute the artifacts directory.
    assert not (run_dir / "artifacts" / "merge_state.json").exists()

    run = store.get_run(run_id)
    assert run.data.params == {"p": "1"}
    assert run.data.tags["t"] == "a"
    assert run.data.metrics == {"m": 1.0}


def test_replay_is_idempotent_without_writes_or_revision_bump(store, merger):
    run_id = _create_run(store)
    delta = _delta(
        "s1",
        100,
        params=(Param("p", "1"),),
        tags=(RunTag("t", "a"),),
        metrics=(Metric("m", 1.0, 1, 0),),
    )
    first = merger.merge(run_id, [delta])
    assert first.revision == 1

    replay = merger.merge(run_id, [delta])

    assert replay.revision == 1
    assert replay.applied_sources == ()
    assert replay.replayed_sources == ("s1",)
    assert replay.rejected_sources == ()
    assert replay.metric_writes == 0
    assert replay.tag_writes == 0
    assert replay.param_writes == 0
    assert replay.conflicts == ()


def test_journal_is_pretty_formed_sorted_keys_and_trailing_newline(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("s1", 100, tags=(RunTag("t", "a"),))])

    raw = (Path(store.resolve_run_dir(run_id)) / "merge_state.json").read_bytes()
    assert raw.endswith(b"\n")
    document = json.loads(raw)
    assert list(document.keys()) == sorted(document.keys())
    assert set(document) == {
        "schema_version",
        "run_id",
        "revision",
        "applied_sources",
        "status",
        "end_time",
        "updated_at_ms",
    }
    assert document["schema_version"] == 1
    assert document["run_id"] == run_id
    assert document["revision"] == 1


def test_inspect_without_journal_creates_nothing(store, merger):
    run_id = _create_run(store)

    state = merger.inspect(run_id)

    assert state.revision == 0
    assert dict(state.applied_sources) == {}
    assert state.status == "RUNNING"
    assert state.end_time is None
    assert not (Path(store.resolve_run_dir(run_id)) / "merge_state.json").exists()


def test_inspect_mixes_live_run_state_with_journal(store, merger):
    run_id = _create_run(store)
    merger.merge(
        run_id,
        [_delta("s1", 100, tags=(RunTag("t", "a"),), status="FINISHED", end_time=42)],
    )

    state = merger.inspect(run_id)

    assert state.revision == 1
    assert set(state.applied_sources) == {"s1"}
    assert state.status == "FINISHED"
    assert state.end_time == 42


# ---------------------------------------------------------------------------
# Order independence
# ---------------------------------------------------------------------------


def test_input_permutation_produces_identical_results(store, merger):
    run_a = _create_run(store, "a")
    run_b = _create_run(store, "b")
    delta_a = _delta("a", 100, tags=(RunTag("t", "a"),), metrics=(Metric("m", 1.0, 1, 0),))
    delta_b = _delta("b", 200, tags=(RunTag("t", "b"),), metrics=(Metric("m", 2.0, 2, 0),))

    first = merger.merge(run_a, [delta_a, delta_b])
    second = merger.merge(run_b, [delta_b, delta_a])

    for field in ("status", "end_time", "param_writes", "tag_writes", "metric_writes"):
        assert getattr(first, field) == getattr(second, field)
    assert first.applied_sources == second.applied_sources == ("a", "b")

    run_first = store.get_run(run_a)
    run_second = store.get_run(run_b)
    assert run_first.data.tags["t"] == run_second.data.tags["t"] == "b"
    assert run_first.data.metrics == run_second.data.metrics == {"m": 2.0}
    assert len(store.get_metric_history(run_a, "m")) == 2
    assert len(store.get_metric_history(run_b, "m")) == 2


# ---------------------------------------------------------------------------
# Param immutability and conflict isolation
# ---------------------------------------------------------------------------


def test_param_value_conflict_does_not_block_tags_or_metrics(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("s0", 10, params=(Param("p", "1"),))])

    outcome = merger.merge(
        run_id,
        [
            _delta(
                "s1",
                20,
                params=(Param("p", "2"),),
                tags=(RunTag("t", "v"),),
                metrics=(Metric("m", 9.0, 1, 0),),
            )
        ],
    )

    assert outcome.param_writes == 0
    assert outcome.tag_writes == 1
    assert outcome.metric_writes == 1
    assert outcome.applied_sources == ()
    assert outcome.rejected_sources == ("s1",)
    conflict = next(c for c in outcome.conflicts if c.kind == "param_value")
    assert conflict == MergeConflict(kind="param_value", key="p", existing="1", incoming="2")
    run = store.get_run(run_id)
    assert run.data.params == {"p": "1"}
    assert run.data.tags["t"] == "v"
    assert run.data.metrics == {"m": 9.0}


def test_same_value_param_is_idempotent(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("s0", 10, params=(Param("p", "1"),))])

    outcome = merger.merge(run_id, [_delta("s1", 20, params=(Param("p", "1"),))])

    assert outcome.param_writes == 0
    assert outcome.conflicts == ()
    assert outcome.applied_sources == ("s1",)


def test_conflicting_param_values_within_batch_write_nothing(store, merger):
    run_id = _create_run(store)
    outcome = merger.merge(
        run_id,
        [
            _delta("a", 10, params=(Param("p", "1"),)),
            _delta("b", 20, params=(Param("p", "2"),)),
        ],
    )

    assert outcome.param_writes == 0
    assert any(c.kind == "param_value" for c in outcome.conflicts)
    assert "p" not in store.get_run(run_id).data.params
    assert set(outcome.rejected_sources) == {"a", "b"}


# ---------------------------------------------------------------------------
# Tags and metrics
# ---------------------------------------------------------------------------


def test_tags_only_write_when_value_changes_and_last_write_wins(store, merger):
    run_id = _create_run(store)
    first = merger.merge(
        run_id,
        [
            _delta("a", 10, tags=(RunTag("t", "1"),)),
            _delta("b", 20, tags=(RunTag("t", "2"),)),
        ],
    )
    assert first.tag_writes == 1
    assert store.get_run(run_id).data.tags["t"] == "2"

    second = merger.merge(
        run_id,
        [
            _delta(
                "c",
                30,
                tags=(
                    RunTag("t", "2"),
                    RunTag("u", "x"),
                ),
            )
        ],
    )
    assert second.tag_writes == 1
    tags = store.get_run(run_id).data.tags
    assert tags["t"] == "2"
    assert tags["u"] == "x"


def test_metric_identity_dedup_including_nan(store, merger):
    run_id = _create_run(store)
    nan_metric = Metric("m", float("nan"), 1, 0)
    first = merger.merge(run_id, [_delta("a", 10, metrics=(nan_metric,))])
    assert first.metric_writes == 1

    second = merger.merge(
        run_id,
        [
            _delta(
                "b",
                20,
                metrics=(
                    nan_metric,
                    Metric("m", float("nan"), 2, 0),
                    Metric("m", 1.0, 1, 0),
                ),
            )
        ],
    )
    assert second.metric_writes == 2
    assert len(store.get_metric_history(run_id, "m")) == 3
    assert math.isnan(store.get_metric_history(run_id, "m")[0].value)


def test_metrics_are_appended_in_one_log_batch_and_latest_wins_in_run_data(store, merger):
    run_id = _create_run(store)
    merger.merge(
        run_id,
        [
            _delta("a", 10, metrics=(Metric("m", 1.0, 1, 0),)),
            _delta("b", 20, metrics=(Metric("m", 5.0, 2, 0), Metric("m", 9.0, 2, 0))),
        ],
    )

    history = store.get_metric_history(run_id, "m")
    assert [(m.timestamp, m.value) for m in history] == [
        (1, 1.0),
        (2, 5.0),
        (2, 9.0),
    ]
    assert store.get_run(run_id).data.metrics == {"m": 9.0}


def test_duplicate_metric_identity_within_delta_is_collapsed(store, merger):
    run_id = _create_run(store)
    outcome = merger.merge(
        run_id,
        [
            _delta(
                "a",
                10,
                metrics=(Metric("m", 1.0, 1, 0), Metric("m", 1.0, 1, 0)),
            )
        ],
    )
    assert outcome.metric_writes == 1
    assert len(store.get_metric_history(run_id, "m")) == 1


# ---------------------------------------------------------------------------
# Status and end time
# ---------------------------------------------------------------------------


def test_terminal_state_requires_end_time(store, merger):
    run_id = _create_run(store)
    outcome = merger.merge(run_id, [_delta("a", 10, status="FINISHED", end_time=None)])

    assert any(c.kind == "terminal_state" for c in outcome.conflicts)
    assert "a" in outcome.rejected_sources
    run = store.get_run(run_id)
    assert run.info.status == "RUNNING"
    assert run.info.end_time is None


def test_finished_then_failed_conflicts_and_freezes_end_time(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("a", 10, status="FINISHED", end_time=100)])

    outcome = merger.merge(run_id, [_delta("b", 20, status="FAILED", end_time=200)])

    assert any(c.kind == "terminal_state" for c in outcome.conflicts)
    assert "b" in outcome.rejected_sources
    run = store.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.info.end_time == 100


def test_repeated_identical_terminal_state_is_idempotent(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("a", 10, status="FINISHED", end_time=100)])
    revision_before = merger.inspect(run_id).revision

    outcome = merger.merge(run_id, [_delta("b", 20, status="FINISHED", end_time=100)])

    assert outcome.conflicts == ()
    assert outcome.applied_sources == ("b",)
    assert store.get_run(run_id).info.status == "FINISHED"
    assert merger.inspect(run_id).revision == revision_before + 1


def test_multiple_end_times_for_one_terminal_target_take_minimum(store, merger):
    run_id = _create_run(store)
    outcome = merger.merge(
        run_id,
        [
            _delta("a", 10, status="FINISHED", end_time=300),
            _delta("b", 20, status="FINISHED", end_time=100),
        ],
    )

    assert outcome.status == "FINISHED"
    assert outcome.end_time == 100
    assert store.get_run(run_id).info.end_time == 100


def test_multiple_status_targets_in_one_call_conflict(store, merger):
    run_id = _create_run(store)
    outcome = merger.merge(
        run_id,
        [
            _delta("a", 10, status="FINISHED", end_time=100),
            _delta("b", 20, status="FAILED", end_time=200),
        ],
    )

    assert any(c.kind == "terminal_state" for c in outcome.conflicts)
    assert store.get_run(run_id).info.status == "RUNNING"


# ---------------------------------------------------------------------------
# Revision mismatch and source reuse
# ---------------------------------------------------------------------------


def test_revision_mismatch_writes_nothing(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("a", 10, tags=(RunTag("t", "1"),))])
    revision_before = merger.inspect(run_id).revision

    outcome = merger.merge(
        run_id,
        [_delta("b", 20, tags=(RunTag("u", "2"),))],
        expected_revision=revision_before + 99,
    )

    assert len(outcome.conflicts) == 1
    assert outcome.conflicts[0].kind == "revision_mismatch"
    assert outcome.revision == revision_before
    assert outcome.applied_sources == ()
    assert "u" not in store.get_run(run_id).data.tags
    assert merger.inspect(run_id).revision == revision_before


def test_source_reuse_with_different_digest_skips_whole_source(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("a", 10, tags=(RunTag("t", "1"),))])
    digest_before = merger.inspect(run_id).applied_sources["a"]

    outcome = merger.merge(run_id, [_delta("a", 20, tags=(RunTag("t", "2"),))])

    conflict = next(c for c in outcome.conflicts if c.kind == "source_reuse")
    assert conflict.key == "a"
    assert conflict.existing == digest_before
    assert conflict.incoming == compute_delta_digest(_delta("a", 20, tags=(RunTag("t", "2"),)))
    assert outcome.rejected_sources == ("a",)
    assert store.get_run(run_id).data.tags["t"] == "1"


def test_duplicate_source_id_within_call_first_wins_others_conflict(store, merger):
    run_id = _create_run(store)
    outcome = merger.merge(
        run_id,
        [
            _delta("dup", 20, tags=(RunTag("t", "late"),)),
            _delta("dup", 10, tags=(RunTag("t", "early"),)),
        ],
    )

    assert any(c.kind == "source_reuse" for c in outcome.conflicts)
    assert outcome.rejected_sources == ("dup",)
    # The conflict-free field of the first (earliest produced_at_ms) delta was
    # still written and converges on retry.
    assert store.get_run(run_id).data.tags["t"] == "early"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bad_delta", "expected_message"),
    [
        (_delta("", 10), "non-empty"),
        (_delta("x" * 129, 10), "at most 128"),
        (_delta("a", -1), "produced_at_ms"),
        (_delta("a", 1, end_time=-1), "end_time"),
        (_delta("a", 1, status="NOT_A_STATUS"), "Invalid status"),
    ],
)
def test_invalid_deltas_fail_the_whole_call(store, merger, bad_delta, expected_message):
    run_id = _create_run(store)
    with pytest.raises(MlflowException, match=expected_message) as exc:
        merger.merge(run_id, [bad_delta, _delta("good", 10, tags=(RunTag("t", "v"),))])
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert not (Path(store.resolve_run_dir(run_id)) / "merge_state.json").exists()
    assert "t" not in store.get_run(run_id).data.tags


def test_metric_model_id_must_be_none(store, merger):
    run_id = _create_run(store)
    with pytest.raises(MlflowException, match="model_id") as exc:
        merger.merge(
            run_id,
            [_delta("a", 1, metrics=(Metric("m", 1.0, 1, 0, model_id="model"),))],
        )
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_batch_limits_match_log_batch(store, merger):
    run_id = _create_run(store)
    too_many_metrics = tuple(Metric(f"m{i}", 1.0, i, 0) for i in range(1001))
    with pytest.raises(MlflowException, match="at most 1000 metrics") as exc:
        merger.merge(run_id, [_delta("a", 1, metrics=too_many_metrics)])
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"
    assert not (Path(store.resolve_run_dir(run_id)) / "merge_state.json").exists()

    run_id_2 = _create_run(store)
    too_many_params = tuple(Param(f"p{i}", "v") for i in range(101))
    with pytest.raises(MlflowException, match="at most 100 params") as exc:
        merger.merge(run_id_2, [_delta("a", 1, params=too_many_params)])
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


def test_run_not_found_raises_resource_does_not_exist(merger):
    with pytest.raises(MlflowException, match="not found") as exc:
        merger.merge("0" * 32, [])
    assert exc.value.error_code == "RESOURCE_DOES_NOT_EXIST"


def test_merge_on_deleted_run_raises_invalid_parameter(store, merger):
    run_id = _create_run(store)
    store.delete_run(run_id)
    with pytest.raises(MlflowException, match="lifecycle_stage") as exc:
        merger.merge(run_id, [_delta("a", 1, tags=(RunTag("t", "v"),))])
    assert exc.value.error_code == "INVALID_PARAMETER_VALUE"


# ---------------------------------------------------------------------------
# Journal corruption recovery
# ---------------------------------------------------------------------------


def test_corrupt_journal_is_quarantined_and_merge_rebuilds(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("a", 10, metrics=(Metric("m", 1.0, 1, 0),))])
    run_dir = Path(store.resolve_run_dir(run_id))
    (run_dir / "merge_state.json").write_text("{broken json")

    outcome = merger.merge(
        run_id, [_delta("b", 20, metrics=(Metric("m", 1.0, 1, 0), Metric("n", 2.0, 2, 0)))]
    )

    quarantined = glob.glob(str(run_dir / "merge_state.corrupt.*.json"))
    assert len(quarantined) == 1
    assert outcome.metric_writes == 1
    assert outcome.applied_sources == ("b",)
    # The pre-existing metric must not be duplicated after rebuilding from 0.
    assert len(store.get_metric_history(run_id, "m")) == 1
    state = merger.inspect(run_id)
    assert state.revision == 1
    assert set(state.applied_sources) == {"b"}


def test_journal_with_bad_schema_version_is_quarantined(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("a", 10, tags=(RunTag("t", "v"),))])
    run_dir = Path(store.resolve_run_dir(run_id))
    journal = json.loads((run_dir / "merge_state.json").read_text())
    journal["schema_version"] = 999
    (run_dir / "merge_state.json").write_text(json.dumps(journal))

    outcome = merger.merge(run_id, [_delta("b", 20, tags=(RunTag("u", "w"),))])

    assert outcome.revision == 1
    assert glob.glob(str(run_dir / "merge_state.corrupt.*.json"))
    assert set(outcome.applied_sources) == {"b"}


def test_unknown_journal_fields_are_ignored(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("a", 10, tags=(RunTag("t", "v"),))])
    run_dir = Path(store.resolve_run_dir(run_id))
    journal_path = run_dir / "merge_state.json"
    journal = json.loads(journal_path.read_text())
    journal["future_field"] = {"nested": True}
    journal_path.write_text(json.dumps(journal))

    state = merger.inspect(run_id)
    assert state.revision == 1
    # A subsequent write round-trips only the fixed field set.
    merger.merge(run_id, [_delta("b", 20, tags=(RunTag("u", "w"),))])
    persisted = json.loads(journal_path.read_text())
    assert "future_field" not in persisted


# ---------------------------------------------------------------------------
# mlflow.runName synchronization
# ---------------------------------------------------------------------------


def test_merge_run_name_tag_syncs_meta_yaml(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("a", 10, tags=(RunTag(MLFLOW_RUN_NAME, "new-name"),))])

    run = store.get_run(run_id)
    assert run.info.run_name == "new-name"
    assert run.data.tags[MLFLOW_RUN_NAME] == "new-name"
    meta = yaml.safe_load((Path(store.resolve_run_dir(run_id)) / "meta.yaml").read_text())
    assert meta["run_name"] == "new-name"


def test_log_batch_run_name_does_not_deadlock_under_reentrant_lock(store):
    run_id = _create_run(store)
    store.log_batch(
        run_id,
        metrics=[],
        params=[],
        tags=[RunTag(MLFLOW_RUN_NAME, "batched-name")],
    )

    run = store.get_run(run_id)
    assert run.data.tags[MLFLOW_RUN_NAME] == "batched-name"
    assert run.info.run_name == "batched-name"
    meta = yaml.safe_load((Path(store.resolve_run_dir(run_id)) / "meta.yaml").read_text())
    assert meta["run_name"] == "batched-name"


# ---------------------------------------------------------------------------
# Concurrency: threads
# ---------------------------------------------------------------------------


def test_concurrent_thread_merges(store, merger):
    run_id = _create_run(store)
    start = threading.Event()
    errors: list[Exception] = []

    def worker(index: int) -> None:
        start.wait()
        try:
            merger.merge(
                run_id,
                [
                    _delta(
                        f"src-{index}",
                        index,
                        metrics=(Metric(f"m{index}", float(index), index, 0),),
                        tags=(RunTag(f"t{index}", "v"),),
                    )
                ],
            )
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout=20)

    assert errors == []
    state = merger.inspect(run_id)
    assert len(state.applied_sources) == 16
    assert state.revision == 16
    run = store.get_run(run_id)
    assert len(run.data.metrics) == 16
    for index in range(16):
        assert len(store.get_metric_history(run_id, f"m{index}")) == 1


def test_concurrent_record_logged_models(store):
    run_id = _create_run(store)
    start = threading.Event()
    errors: list[Exception] = []

    def worker(index: int) -> None:
        start.wait()
        try:
            model = Model(name=f"model-{index}")
            store.record_logged_model(run_id, model)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout=20)

    assert errors == []
    logged_models_tag = next(
        tag for tag in store.get_all_tags(run_id) if tag.key == MLFLOW_LOGGED_MODELS
    )
    assert len(json.loads(logged_models_tag.value)) == 16


def test_concurrent_update_and_read_never_observes_partial_meta_yaml(store):
    run_id = _create_run(store)
    stop = threading.Event()
    errors: list[Exception] = []

    def writer() -> None:
        status = 0
        while not stop.is_set():
            store.update_run_info(
                run_id,
                1,  # RUNNING
                None,
                run_name=f"name-{status}",
            )
            status += 1

    def reader() -> None:
        for _ in range(500):
            try:
                run = store.get_run(run_id)
                assert run.info.run_id == run_id
                assert run.info.experiment_id is not None
            except yaml.YAMLError as exc:  # pragma: no cover - failure path
                errors.append(exc)

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    writer_thread.start()
    reader_thread.start()
    reader_thread.join(timeout=20)
    stop.set()
    writer_thread.join(timeout=20)

    assert errors == []


# ---------------------------------------------------------------------------
# Concurrency: processes
# ---------------------------------------------------------------------------


def _mp_merge_worker(root: str, run_id: str, index: int, ready: Any, go: Any) -> None:
    os.environ[MLFLOW_ALLOW_FILE_STORE.name] = "true"
    store = FileStore(root)
    merger = FileStoreRunMerger(store, lock_timeout_seconds=15.0)
    ready.set()
    go.wait()
    merger.merge(
        run_id,
        [
            _delta(
                f"proc-{index}",
                index,
                metrics=(Metric(f"pm{index}", float(index), index, 0),),
            )
        ],
    )


def test_concurrent_process_merges(store, tmp_path):
    run_id = _create_run(store)
    root = store.root_directory
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Event()
    go = ctx.Event()
    processes = [
        ctx.Process(
            target=_mp_merge_worker,
            args=(root, run_id, i, ready, go),
        )
        for i in range(4)
    ]
    for process in processes:
        process.start()
    for _ in range(4):
        ready.wait(timeout=10)
    go.set()
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0

    merger = FileStoreRunMerger(store)
    state = merger.inspect(run_id)
    assert len(state.applied_sources) == 4
    assert state.revision == 4


def _mp_lock_holder(root: str, run_id: str, acquired: Any, release: Any) -> None:
    os.environ[MLFLOW_ALLOW_FILE_STORE.name] = "true"
    store = FileStore(root)
    with store._run_write_lock(run_id):
        acquired.set()
        release.wait()


def test_lock_timeout_raises_temporarily_unavailable(store, tmp_path):
    run_id = _create_run(store)
    root = store.root_directory
    ctx = multiprocessing.get_context("spawn")
    acquired = ctx.Event()
    release = ctx.Event()
    holder = ctx.Process(target=_mp_lock_holder, args=(root, run_id, acquired, release))
    holder.start()
    try:
        assert acquired.wait(timeout=10)
        slow_merger = FileStoreRunMerger(store, lock_timeout_seconds=0.5)
        with pytest.raises(MlflowException, match="Timed out") as exc:
            slow_merger.merge(run_id, [_delta("blocked", 1, tags=(RunTag("t", "v"),))])
        assert exc.value.error_code == "TEMPORARILY_UNAVAILABLE"
    finally:
        release.set()
        holder.join(timeout=10)

    # Once the holder exits, the lock is immediately reusable.
    merger = FileStoreRunMerger(store, lock_timeout_seconds=5.0)
    outcome = merger.merge(run_id, [_delta("after", 1, tags=(RunTag("t", "v"),))])
    assert outcome.applied_sources == ("after",)


def test_lock_is_reentrant_within_one_thread(store):
    run_id = _create_run(store)
    with store._run_write_lock(run_id):
        with store._run_write_lock(run_id):
            store.set_tag(run_id, RunTag("nested", "v"))
    assert store.get_run(run_id).data.tags["nested"] == "v"


# ---------------------------------------------------------------------------
# Read paths stay lock-free and journal stays invisible
# ---------------------------------------------------------------------------


def test_journal_files_never_appear_in_tags_or_search(store, merger):
    run_id = _create_run(store)
    merger.merge(run_id, [_delta("a", 10, tags=(RunTag("t", "v"),))])

    tag_keys = {tag.key for tag in store.get_all_tags(run_id)}
    assert "merge_state.json" not in tag_keys
    assert ".merge.lock" not in tag_keys

    experiment_id = store.get_run(run_id).info.experiment_id
    runs = store.search_runs([experiment_id], "", ViewType.ALL, max_results=10)
    assert [run.info.run_id for run in runs] == [run_id]
    assert all("merge_state.json" not in run.data.tags for run in runs)
    assert all(".merge.lock" not in run.data.tags for run in runs)


def test_get_metric_history_excludes_journal_and_remains_complete(store, merger):
    run_id = _create_run(store)
    merger.merge(
        run_id, [_delta("a", 10, metrics=(Metric("m", 1.0, 1, 0), Metric("m", 2.0, 2, 0)))]
    )
    history = store.get_metric_history(run_id, "m")
    assert [m.value for m in history] == [1.0, 2.0]


def test_file_uri_tracking_uri_still_works(tmp_path, monkeypatch):
    monkeypatch.setenv(MLFLOW_ALLOW_FILE_STORE.name, "true")
    uri = tmp_path.absolute().as_uri()
    store = FileStore(uri)
    run_id = _create_run(store)
    merger = FileStoreRunMerger(store)
    outcome = merger.merge(run_id, [_delta("a", 10, tags=(RunTag("t", "v"),))])
    assert outcome.applied_sources == ("a",)
    assert store.get_run(run_id).data.tags["t"] == "v"
