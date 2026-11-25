"""Idempotent, journaled merging of run metadata deltas into the FileStore.

The merger gives the filesystem tracking backend an apply/replay semantics for
batched run metadata: each delta is named by a ``source_id`` and content-addressed
by a SHA-256 digest. A per-run JSON journal (``merge_state.json``) records which
sources have already been applied, and a per-run lock file (``.merge.lock``)
serializes every mutating FileStore operation, both within a process (reentrant)
and across processes (``fcntl``/``msvcrt``).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from mlflow.entities import Metric, Param, RunInfo, RunStatus, RunTag
from mlflow.entities.run_info import check_run_is_active
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import (
    INTERNAL_ERROR,
    INVALID_PARAMETER_VALUE,
    TEMPORARILY_UNAVAILABLE,
)
from mlflow.utils.file_utils import ENCODING
from mlflow.utils.time import get_current_time_millis
from mlflow.utils.validation import (
    _validate_batch_log_limits,
    _validate_metric,
    _validate_param,
    _validate_run_id,
    _validate_tag,
)

if TYPE_CHECKING:
    from mlflow.store.tracking.file_store import FileStore

if os.name == "nt":
    import msvcrt  # clint: disable=lazy-import
else:
    import fcntl  # clint: disable=lazy-import

_logger = logging.getLogger(__name__)

MergeConflictKind = Literal["param_value", "terminal_state", "source_reuse", "revision_mismatch"]

_SCHEMA_VERSION = 1
_JOURNAL_FILE_NAME = "merge_state.json"
_LOCK_FILE_NAME = ".merge.lock"
_CORRUPT_FILE_PREFIX = "merge_state.corrupt."
_CORRUPT_FILE_SUFFIX = ".json"
_MAX_SOURCE_ID_LENGTH = 128

# Shared by every FileStore in this interpreter; keyed by the resolved run
# directory so distinct store instances pointing at the same root serialize on
# one in-process lock, while fcntl/msvcrt handles cross-process exclusion.
_LOCK_REGISTRY: dict[str, RunWriteLock] = {}
_LOCK_REGISTRY_GUARD = threading.Lock()


# ---------------------------------------------------------------------------
# Locking and atomic writes
# ---------------------------------------------------------------------------


def _journal_path(run_dir: str) -> str:
    return os.path.join(run_dir, _JOURNAL_FILE_NAME)


def _lock_path(run_dir: str) -> str:
    return os.path.join(run_dir, _LOCK_FILE_NAME)


def _fsync_directory(directory: str) -> None:
    if os.name == "nt":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _replace_with_retry(src: str, dst: str) -> None:
    if os.name != "nt":
        os.replace(src, dst)
        return
    # Windows can briefly reject ReplaceFile with a sharing violation when an
    # antivirus/indexer handle is open on the destination. Retry boundedly
    # instead of falling back to a non-atomic direct write.
    deadline = time.monotonic() + 2.0
    while True:
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def atomic_replace_write(path: str, payload: str) -> None:
    """Write ``payload`` to ``path`` via a same-directory temp file + os.replace.

    Readers can only observe the complete old or complete new version. The
    temp file is fsynced before the replace, and the parent directory is
    fsynced afterwards (POSIX) so the replacement survives a crash.
    """
    directory = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        # mkstemp creates the temp file at 0600; match the target's existing
        # mode (or the process umask default for a brand-new file) so replacing
        # does not tighten permissions.
        try:
            mode = os.stat(path).st_mode
        except FileNotFoundError:
            mode = 0o666 & ~get_umask()
        os.fchmod(fd, mode)
        with os.fdopen(fd, "w", encoding=ENCODING, newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        _replace_with_retry(tmp_path, path)
        _fsync_directory(directory)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def get_umask() -> int:
    current = os.umask(0)
    os.umask(current)
    return current


class RunWriteLock:
    """Reentrant, per-run exclusive lock backed by an OS file lock.

    The on-disk lock file holds one byte and carries no PID or state: when a
    process exits holding the lock the kernel releases it automatically, so a
    leftover ``.merge.lock`` never means the lock is still held.
    """

    def __init__(self, run_dir: str) -> None:
        self._run_dir = run_dir
        self._path = _lock_path(run_dir)
        self._depth = threading.local()

    @contextlib.contextmanager
    def acquire(self, timeout_seconds: float) -> Iterator[None]:
        if self._current_depth() > 0:
            self._depth.depth = self._current_depth() + 1
            try:
                yield
            finally:
                self._depth.depth = self._current_depth() - 1
            return

        os.makedirs(self._run_dir, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            self._acquire(fd, timeout_seconds)
            self._depth.depth = 1
            try:
                yield
            finally:
                self._depth.depth = 0
                self._release(fd)
        finally:
            os.close(fd)

    def _current_depth(self) -> int:
        return getattr(self._depth, "depth", 0)

    def _acquire(self, fd: int, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        backoff = 0.01
        while True:
            if self._try_lock(fd):
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\x00")
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise MlflowException(
                    f"Timed out after {timeout_seconds}s waiting for the write lock of "
                    f"run directory '{self._run_dir}'.",
                    TEMPORARILY_UNAVAILABLE,
                )
            time.sleep(min(backoff + random.uniform(0, backoff), remaining))
            backoff = min(backoff * 2, 0.25)

    def _try_lock(self, fd: int) -> bool:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
                return True
            except OSError:
                return False
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False

    def _release(self, fd: int) -> None:
        if os.name == "nt":
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
        else:
            fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def run_write_lock(store: FileStore, run_id: str, timeout_seconds: float = 10.0) -> Iterator[None]:
    run_dir = store.resolve_run_dir(run_id)
    with _LOCK_REGISTRY_GUARD:
        lock = _LOCK_REGISTRY.get(run_dir)
        if lock is None:
            lock = RunWriteLock(run_dir)
            _LOCK_REGISTRY[run_dir] = lock
    with lock.acquire(timeout_seconds):
        yield


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MergeConflict:
    kind: MergeConflictKind
    key: str
    existing: str | None
    incoming: str | None


@dataclass(frozen=True)
class RunMetadataDelta:
    source_id: str
    produced_at_ms: int
    params: tuple[Param, ...]
    tags: tuple[RunTag, ...]
    metrics: tuple[Metric, ...]
    status: str | None
    end_time: int | None


@dataclass(frozen=True)
class MergeState:
    revision: int
    applied_sources: Mapping[str, str]
    status: str | None
    end_time: int | None


@dataclass(frozen=True)
class MergeOutcome:
    run_id: str
    revision: int
    applied_sources: tuple[str, ...]
    replayed_sources: tuple[str, ...]
    rejected_sources: tuple[str, ...]
    param_writes: int
    tag_writes: int
    metric_writes: int
    conflicts: tuple[MergeConflict, ...]
    status: str | None
    end_time: int | None


@dataclass(frozen=True)
class _StatusPlan:
    applied: bool
    status: str | None = None
    end_time: int | None = None
    sources: frozenset[str] = frozenset()


# ---------------------------------------------------------------------------
# Digests, identities and validation
# ---------------------------------------------------------------------------


def _is_nan(value: float) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _canonical_entity_dict(delta: RunMetadataDelta) -> dict[str, Any]:
    def param_dict(param: Param) -> dict[str, Any]:
        return {"key": param.key, "value": param.value}

    def tag_dict(tag: RunTag) -> dict[str, Any]:
        return {"key": tag.key, "value": tag.value}

    def metric_dict(metric: Metric) -> dict[str, Any]:
        return {
            "key": metric.key,
            "value": "NaN" if _is_nan(metric.value) else metric.value,
            "timestamp": metric.timestamp,
            "step": metric.step,
            "dataset_name": metric.dataset_name,
            "dataset_digest": metric.dataset_digest,
        }

    return {
        "source_id": delta.source_id,
        "produced_at_ms": delta.produced_at_ms,
        "params": [param_dict(param) for param in delta.params],
        "tags": [tag_dict(tag) for tag in delta.tags],
        "metrics": [metric_dict(metric) for metric in delta.metrics],
        "status": delta.status,
        "end_time": delta.end_time,
    }


def compute_delta_digest(delta: RunMetadataDelta) -> str:
    """SHA-256 (lowercase hex) of the normalized delta payload."""
    canonical = json.dumps(
        _canonical_entity_dict(delta),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _metric_identity(metric: Metric) -> tuple[Any, ...]:
    value = "NaN" if _is_nan(metric.value) else metric.value
    return (
        metric.key,
        metric.timestamp,
        metric.step,
        value,
        metric.dataset_name,
        metric.dataset_digest,
    )


def _validate_delta(delta: RunMetadataDelta) -> None:
    if not isinstance(delta, RunMetadataDelta):
        raise MlflowException(
            f"Each delta must be a RunMetadataDelta, got {type(delta)!r}.",
            INVALID_PARAMETER_VALUE,
        )
    if not isinstance(delta.source_id, str) or not delta.source_id:
        raise MlflowException(
            "delta.source_id must be a non-empty string.",
            INVALID_PARAMETER_VALUE,
        )
    if len(delta.source_id) > _MAX_SOURCE_ID_LENGTH:
        raise MlflowException(
            f"delta.source_id must be at most {_MAX_SOURCE_ID_LENGTH} characters, got "
            f"{len(delta.source_id)}.",
            INVALID_PARAMETER_VALUE,
        )
    if isinstance(delta.produced_at_ms, bool) or not isinstance(delta.produced_at_ms, int):
        raise MlflowException(
            "delta.produced_at_ms must be a nonnegative integer.",
            INVALID_PARAMETER_VALUE,
        )
    if delta.produced_at_ms < 0:
        raise MlflowException(
            "delta.produced_at_ms must be a nonnegative integer.",
            INVALID_PARAMETER_VALUE,
        )
    if delta.end_time is not None and (
        isinstance(delta.end_time, bool)
        or not isinstance(delta.end_time, int)
        or delta.end_time < 0
    ):
        raise MlflowException(
            "delta.end_time must be a nonnegative integer or None.",
            INVALID_PARAMETER_VALUE,
        )
    if delta.status is not None and delta.status not in RunStatus._STRING_TO_STATUS:
        raise MlflowException(
            f"Invalid status '{delta.status}'. Valid statuses: "
            f"{sorted(RunStatus._STRING_TO_STATUS)}.",
            INVALID_PARAMETER_VALUE,
        )
    for param in delta.params:
        if not isinstance(param, Param):
            raise MlflowException(
                f"delta.params entries must be Param, got {type(param)!r}.",
                INVALID_PARAMETER_VALUE,
            )
        _validate_param(param.key, param.value)
    for tag in delta.tags:
        if not isinstance(tag, RunTag):
            raise MlflowException(
                f"delta.tags entries must be RunTag, got {type(tag)!r}.",
                INVALID_PARAMETER_VALUE,
            )
        _validate_tag(tag.key, tag.value)
    for metric in delta.metrics:
        if not isinstance(metric, Metric):
            raise MlflowException(
                f"delta.metrics entries must be Metric, got {type(metric)!r}.",
                INVALID_PARAMETER_VALUE,
            )
        if metric.model_id is not None:
            raise MlflowException(
                "metric.model_id must be None for run metadata deltas.",
                INVALID_PARAMETER_VALUE,
            )
        _validate_metric(metric.key, metric.value, metric.timestamp, metric.step)


# ---------------------------------------------------------------------------
# Delta normalization
# ---------------------------------------------------------------------------


def _dedupe_delta_entities(delta: RunMetadataDelta) -> RunMetadataDelta:
    # Tags: last write wins within the delta.
    tag_by_key: dict[str, RunTag] = {}
    for tag in delta.tags:
        tag_by_key[tag.key] = tag

    seen_metric_ids: set[tuple[Any, ...]] = set()
    metrics: list[Metric] = []
    for metric in delta.metrics:
        identity = _metric_identity(metric)
        if identity not in seen_metric_ids:
            seen_metric_ids.add(identity)
            metrics.append(metric)

    return RunMetadataDelta(
        source_id=delta.source_id,
        produced_at_ms=delta.produced_at_ms,
        params=tuple(delta.params),
        tags=tuple(tag_by_key[key] for key in sorted(tag_by_key)),
        metrics=tuple(metrics),
        status=delta.status,
        end_time=delta.end_time,
    )


def _dedupe_params_within_delta(
    delta: RunMetadataDelta,
) -> tuple[list[Param], list[MergeConflict]]:
    by_key: dict[str, str] = {}
    conflicts: list[MergeConflict] = []
    for param in delta.params:
        previous = by_key.get(param.key)
        if previous is None:
            by_key[param.key] = param.value
        elif previous != param.value:
            conflicts.append(
                MergeConflict(
                    kind="param_value",
                    key=param.key,
                    existing=previous,
                    incoming=param.value,
                )
            )
    return [Param(key, value) for key, value in sorted(by_key.items())], conflicts


def _normalize_deltas(
    deltas: list[RunMetadataDelta],
) -> tuple[list[RunMetadataDelta], list[MergeConflict], set[str]]:
    """Sort deltas, collapse repeated source_id submissions, dedupe entities.

    Returns one normalized delta per source_id (ordered by
    ``(produced_at_ms, source_id)``), the ``source_reuse`` conflicts for
    duplicate source_id submissions within this call, and their rejected ids.
    """
    enriched = sorted(
        ((delta, compute_delta_digest(delta)) for delta in deltas),
        key=lambda item: (item[0].produced_at_ms, item[0].source_id, item[1]),
    )
    conflicts: list[MergeConflict] = []
    rejected: set[str] = set()
    chosen: dict[str, tuple[RunMetadataDelta, str]] = {}
    for delta, digest in enriched:
        existing = chosen.get(delta.source_id)
        if existing is None:
            chosen[delta.source_id] = (delta, digest)
        else:
            conflicts.append(
                MergeConflict(
                    kind="source_reuse",
                    key=delta.source_id,
                    existing=existing[1],
                    incoming=digest,
                )
            )
            rejected.add(delta.source_id)

    normalized = [
        _dedupe_delta_entities(chosen[source_id][0])
        for source_id in sorted(chosen, key=lambda sid: (chosen[sid][0].produced_at_ms, sid))
    ]
    return normalized, conflicts, rejected


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


class _Journal:
    def __init__(
        self,
        revision: int,
        applied_sources: dict[str, str],
        status: str | None,
        end_time: int | None,
        updated_at_ms: int,
    ) -> None:
        self.revision = revision
        self.applied_sources = applied_sources
        self.status = status
        self.end_time = end_time
        self.updated_at_ms = updated_at_ms

    def to_payload(self, run_id: str) -> str:
        document = {
            "schema_version": _SCHEMA_VERSION,
            "run_id": run_id,
            "revision": self.revision,
            "applied_sources": dict(sorted(self.applied_sources.items())),
            "status": self.status,
            "end_time": self.end_time,
            "updated_at_ms": self.updated_at_ms,
        }
        return json.dumps(document, sort_keys=True, ensure_ascii=False) + "\n"


def _empty_journal(status: str | None, end_time: int | None) -> _Journal:
    return _Journal(0, {}, status, end_time, get_current_time_millis())


def _load_journal(
    run_dir: str, run_id: str, current_status: str | None, current_end_time: int | None
) -> _Journal:
    path = _journal_path(run_dir)
    if not os.path.exists(path):
        return _empty_journal(current_status, current_end_time)
    try:
        with open(path, encoding=ENCODING) as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            raise ValueError("journal root must be a JSON object")
        if document.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {document.get('schema_version')!r}")
        if document.get("run_id") != run_id:
            raise ValueError(f"journal run_id {document.get('run_id')!r} does not match {run_id!r}")
        revision = document["revision"]
        applied = document["applied_sources"]
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ValueError(f"invalid revision: {revision!r}")
        if not isinstance(applied, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in applied.items()
        ):
            raise ValueError("invalid applied_sources mapping")
        return _Journal(
            revision=revision,
            applied_sources=dict(applied),
            # Unknown fields are ignored on read; only the fixed field set is
            # written back.
            status=document.get("status"),
            end_time=document.get("end_time"),
            updated_at_ms=document.get("updated_at_ms", get_current_time_millis()),
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return _quarantine_and_rebuild(run_dir, path, current_status, current_end_time, exc)


def _quarantine_and_rebuild(
    run_dir: str,
    corrupt_path: str,
    current_status: str | None,
    current_end_time: int | None,
    exc: Exception,
) -> _Journal:
    corrupt_name = f"{_CORRUPT_FILE_PREFIX}{get_current_time_millis()}{_CORRUPT_FILE_SUFFIX}"
    os.replace(corrupt_path, os.path.join(run_dir, corrupt_name))
    _fsync_directory(run_dir)
    _logger.warning(
        "Quarantined corrupt merge journal '%s' as '%s': %s",
        corrupt_path,
        corrupt_name,
        exc,
    )
    return _empty_journal(current_status, current_end_time)


def _conflict_sort_key(conflict: MergeConflict) -> tuple[str, str, str, str]:
    return (
        conflict.kind,
        conflict.key,
        conflict.existing if conflict.existing is not None else "",
        conflict.incoming if conflict.incoming is not None else "",
    )


# ---------------------------------------------------------------------------
# Merger
# ---------------------------------------------------------------------------


class FileStoreRunMerger:
    """Merge ordered run-metadata deltas into a :class:`FileStore`-backed run."""

    def __init__(self, store: FileStore, *, lock_timeout_seconds: float = 10.0) -> None:
        self._store = store
        self._lock_timeout_seconds = lock_timeout_seconds

    def inspect(self, run_id: str) -> MergeState:
        _validate_run_id(run_id)
        # Read-only: no exclusive lock, may observe an old or new journal
        # snapshot. Never creates merge_state.json.
        run_info = self._store._get_run_info(run_id)
        run_dir = self._store.resolve_run_dir(run_id)
        journal = _load_journal(run_dir, run_id, run_info.status, run_info.end_time)
        return MergeState(
            revision=journal.revision,
            applied_sources=dict(sorted(journal.applied_sources.items())),
            status=run_info.status,
            end_time=run_info.end_time,
        )

    def merge(
        self,
        run_id: str,
        deltas: list[RunMetadataDelta],
        *,
        expected_revision: int | None = None,
    ) -> MergeOutcome:
        _validate_run_id(run_id)
        if not isinstance(deltas, list | tuple):
            raise MlflowException(
                "deltas must be a list of RunMetadataDelta.",
                INVALID_PARAMETER_VALUE,
            )
        # All validation happens before any lock is taken or byte written;
        # one invalid delta fails the whole call with no side effects.
        for delta in deltas:
            _validate_delta(delta)

        with run_write_lock(self._store, run_id, self._lock_timeout_seconds):
            return self._merge_locked(run_id, list(deltas), expected_revision)

    def _merge_locked(
        self,
        run_id: str,
        deltas: list[RunMetadataDelta],
        expected_revision: int | None,
    ) -> MergeOutcome:
        run_info = self._store._get_run_info(run_id)
        check_run_is_active(run_info)
        run_dir = self._store.resolve_run_dir(run_id)
        journal = _load_journal(run_dir, run_id, run_info.status, run_info.end_time)

        if expected_revision is not None and expected_revision != journal.revision:
            return self._revision_mismatch_outcome(run_id, run_info, journal, expected_revision)

        normalized, intra_call_conflicts, intra_call_rejected = _normalize_deltas(deltas)

        eligible: list[tuple[RunMetadataDelta, str]] = []
        replayed: list[str] = []
        rejected: set[str] = set(intra_call_rejected)
        conflicts: list[MergeConflict] = list(intra_call_conflicts)
        for delta in normalized:
            digest = compute_delta_digest(delta)
            recorded = journal.applied_sources.get(delta.source_id)
            if recorded is None:
                eligible.append((delta, digest))
            elif recorded == digest:
                replayed.append(delta.source_id)
            else:
                conflicts.append(
                    MergeConflict(
                        kind="source_reuse",
                        key=delta.source_id,
                        existing=recorded,
                        incoming=digest,
                    )
                )
                rejected.add(delta.source_id)

        param_plan, param_conflicts, param_conflict_sources = self._plan_params(run_info, eligible)
        tag_plan = self._plan_tags(run_info, eligible)
        metric_plan = self._plan_metrics(run_info, eligible)
        status_plan, status_conflicts, status_conflict_sources = self._plan_status(
            run_info, eligible
        )
        conflicts.extend(param_conflicts)
        conflicts.extend(status_conflicts)

        conflicted_sources = rejected | param_conflict_sources | status_conflict_sources

        # Aggregate limits match log_batch's and are enforced before any write.
        _validate_batch_log_limits(
            [metric for metric, _sources in metric_plan],
            [param for param, _sources in param_plan],
            [tag for tag, _sources in tag_plan],
        )

        try:
            applied, writes = self._apply_plans(
                run_info=run_info,
                eligible=eligible,
                conflicted_sources=conflicted_sources,
                param_plan=param_plan,
                tag_plan=tag_plan,
                metric_plan=metric_plan,
                status_plan=status_plan,
            )
        except MlflowException:
            raise
        except Exception as exc:
            # The journal is not advanced on a failed write; callers retry the
            # same deltas and converge through identity/idempotency rules.
            raise MlflowException(str(exc), INTERNAL_ERROR) from exc
        param_writes, tag_writes, metric_writes, final_info = writes

        added_sources = {
            delta.source_id: digest for delta, digest in eligible if delta.source_id in applied
        }
        wrote_store = param_writes + tag_writes + metric_writes > 0 or status_plan.applied
        added_any = bool(added_sources)
        if added_any:
            journal.applied_sources.update(added_sources)
        if status_plan.applied:
            journal.status = final_info.status
            journal.end_time = final_info.end_time

        if wrote_store or added_any:
            journal.revision += 1
            journal.updated_at_ms = get_current_time_millis()
            self._write_journal(run_dir, run_id, journal)

        return MergeOutcome(
            run_id=run_id,
            revision=journal.revision,
            applied_sources=tuple(sorted(applied)),
            replayed_sources=tuple(sorted(replayed)),
            rejected_sources=tuple(sorted(conflicted_sources)),
            param_writes=param_writes,
            tag_writes=tag_writes,
            metric_writes=metric_writes,
            conflicts=tuple(sorted(conflicts, key=_conflict_sort_key)),
            status=final_info.status,
            end_time=final_info.end_time,
        )

    def _revision_mismatch_outcome(
        self,
        run_id: str,
        run_info: RunInfo,
        journal: _Journal,
        expected_revision: int,
    ) -> MergeOutcome:
        conflict = MergeConflict(
            kind="revision_mismatch",
            key="revision",
            existing=str(journal.revision),
            incoming=str(expected_revision),
        )
        return MergeOutcome(
            run_id=run_id,
            revision=journal.revision,
            applied_sources=(),
            replayed_sources=(),
            rejected_sources=(),
            param_writes=0,
            tag_writes=0,
            metric_writes=0,
            conflicts=(conflict,),
            status=run_info.status,
            end_time=run_info.end_time,
        )

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_params(
        self,
        run_info: RunInfo,
        eligible: list[tuple[RunMetadataDelta, str]],
    ) -> tuple[list[tuple[Param, set[str]]], list[MergeConflict], set[str]]:
        existing_params = {
            param.key: param.value for param in self._store._get_all_params(run_info)
        }
        conflicts: list[MergeConflict] = []
        conflict_sources: set[str] = set()
        # key -> (value, contributing source ids)
        planned: dict[str, tuple[str, set[str]]] = {}
        conflicting_keys: set[str] = set()

        for delta, _digest in eligible:
            params, intra_conflicts = _dedupe_params_within_delta(delta)
            if intra_conflicts:
                conflict_sources.add(delta.source_id)
                conflicts.extend(intra_conflicts)
                conflicting_keys.update(conflict.key for conflict in intra_conflicts)
            for param in params:
                if param.key in conflicting_keys:
                    conflict_sources.add(delta.source_id)
                    continue
                entry = planned.get(param.key)
                if entry is None:
                    planned[param.key] = (param.value, {delta.source_id})
                elif entry[0] != param.value:
                    conflicting_keys.add(param.key)
                    conflict_sources.update(entry[1])
                    conflict_sources.add(delta.source_id)
                    conflicts.append(
                        MergeConflict(
                            kind="param_value",
                            key=param.key,
                            existing=entry[0],
                            incoming=param.value,
                        )
                    )
                else:
                    entry[1].add(delta.source_id)

        plan: list[tuple[Param, set[str]]] = []
        for key in sorted(planned):
            if key in conflicting_keys:
                continue
            value, sources = planned[key]
            if key in existing_params:
                if existing_params[key] != value:
                    conflicts.append(
                        MergeConflict(
                            kind="param_value",
                            key=key,
                            existing=existing_params[key],
                            incoming=value,
                        )
                    )
                    conflict_sources.update(sources)
                # Same value already stored: idempotent skip, not a write.
                continue
            plan.append((Param(key, value), sources))
        return plan, conflicts, conflict_sources

    def _plan_tags(
        self,
        run_info: RunInfo,
        eligible: list[tuple[RunMetadataDelta, str]],
    ) -> list[tuple[RunTag, set[str]]]:
        existing_tags = {tag.key: tag.value for tag in self._store._get_all_tags(run_info)}
        # Later deltas overwrite earlier ones; eligible is already ordered.
        planned: dict[str, tuple[str, set[str]]] = {}
        for delta, _digest in eligible:
            for tag in delta.tags:
                entry = planned.get(tag.key)
                if entry is None:
                    planned[tag.key] = (tag.value, {delta.source_id})
                else:
                    entry[1].add(delta.source_id)
                    planned[tag.key] = (tag.value, entry[1])
        return [
            (RunTag(key, value), sources)
            for key, (value, sources) in sorted(planned.items())
            if existing_tags.get(key) != value
        ]

    def _plan_metrics(
        self,
        run_info: RunInfo,
        eligible: list[tuple[RunMetadataDelta, str]],
    ) -> list[tuple[Metric, set[str]]]:
        existing_by_key: dict[str, set[tuple[Any, ...]]] = {}
        for metric in self._store._get_all_metrics(run_info):
            existing_by_key.setdefault(metric.key, set()).add(_metric_identity(metric))
        planned: dict[tuple[Any, ...], tuple[Metric, set[str]]] = {}
        for delta, _digest in eligible:
            for metric in delta.metrics:
                identity = _metric_identity(metric)
                if identity in existing_by_key.get(metric.key, set()):
                    continue
                entry = planned.get(identity)
                if entry is None:
                    planned[identity] = (metric, {delta.source_id})
                else:
                    entry[1].add(delta.source_id)
        return list(planned.values())

    def _plan_status(
        self,
        run_info: RunInfo,
        eligible: list[tuple[RunMetadataDelta, str]],
    ) -> tuple[_StatusPlan, list[MergeConflict], set[str]]:
        requests = [(delta, digest) for delta, digest in eligible if delta.status is not None]
        conflicts: list[MergeConflict] = []
        conflict_sources: set[str] = set()
        if not requests:
            return _StatusPlan(applied=False), conflicts, conflict_sources

        terminal_strings = {
            RunStatus.to_string(status) for status in RunStatus._TERMINATED_STATUSES
        }
        current_terminal = run_info.status in terminal_strings

        valid_requests: list[tuple[RunMetadataDelta, str]] = []
        for delta, digest in requests:
            if delta.status in terminal_strings and delta.end_time is None:
                conflicts.append(
                    MergeConflict(
                        kind="terminal_state",
                        key="status",
                        existing=run_info.status,
                        incoming=delta.status,
                    )
                )
                conflict_sources.add(delta.source_id)
            else:
                valid_requests.append((delta, digest))

        if current_terminal:
            for delta, _digest in valid_requests:
                if delta.status != run_info.status:
                    conflicts.append(
                        MergeConflict(
                            kind="terminal_state",
                            key="status",
                            existing=run_info.status,
                            incoming=delta.status,
                        )
                    )
                    conflict_sources.add(delta.source_id)
            # Repeated identical terminal states are idempotent.
            return _StatusPlan(applied=False), conflicts, conflict_sources

        if not valid_requests:
            return _StatusPlan(applied=False), conflicts, conflict_sources

        unique_targets = {delta.status for delta, _digest in valid_requests}
        if len(unique_targets) > 1:
            conflict_sources.update(delta.source_id for delta, _digest in valid_requests)
            conflicts.append(
                MergeConflict(
                    kind="terminal_state",
                    key="status",
                    existing=run_info.status,
                    incoming=",".join(sorted(target for target in unique_targets if target)),
                )
            )
            return _StatusPlan(applied=False), conflicts, conflict_sources

        target = next(iter(unique_targets))
        target_deltas = [delta for delta, _digest in valid_requests]
        if target in terminal_strings:
            end_time = min(delta.end_time for delta in target_deltas if delta.end_time is not None)
        else:
            end_time = run_info.end_time
        return (
            _StatusPlan(
                applied=True,
                status=target,
                end_time=end_time,
                sources=frozenset(delta.source_id for delta in target_deltas),
            ),
            conflicts,
            conflict_sources,
        )

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def _apply_plans(
        self,
        run_info: RunInfo,
        eligible: list[tuple[RunMetadataDelta, str]],
        conflicted_sources: set[str],
        param_plan: list[tuple[Param, set[str]]],
        tag_plan: list[tuple[RunTag, set[str]]],
        metric_plan: list[tuple[Metric, set[str]]],
        status_plan: _StatusPlan,
    ) -> tuple[set[str], tuple[int, int, int, RunInfo]]:
        store = self._store

        for param, _sources in param_plan:
            store._log_run_param(run_info, param)
        param_writes = len(param_plan)

        for tag, _sources in tag_plan:
            # Preserve the mlflow.runName <-> meta.yaml synchronization.
            if tag.key == "mlflow.runName":
                store._set_run_name(run_info, tag.value)
            else:
                store._set_run_tag(run_info, tag)
        tag_writes = len(tag_plan)

        if metric_plan:
            store.log_batch(
                run_info.run_id,
                [metric for metric, _sources in metric_plan],
                [],
                [],
            )
        metric_writes = len(metric_plan)

        final_info = run_info
        if status_plan.applied and status_plan.status is not None:
            final_info = self._overwrite_run_status(
                run_info, status_plan.status, status_plan.end_time
            )

        applied = {
            delta.source_id
            for delta, _digest in eligible
            if delta.source_id not in conflicted_sources
        }
        return applied, (param_writes, tag_writes, metric_writes, final_info)

    def _overwrite_run_status(
        self, run_info: RunInfo, status: str, end_time: int | None
    ) -> RunInfo:
        new_info = RunInfo(
            run_id=run_info.run_id,
            run_name=run_info.run_name,
            experiment_id=run_info.experiment_id,
            user_id=run_info.user_id,
            status=status,
            start_time=run_info.start_time,
            end_time=end_time,
            lifecycle_stage=run_info.lifecycle_stage,
            artifact_uri=run_info.artifact_uri,
        )
        self._store._overwrite_run_info_atomic(new_info)
        return new_info

    # ------------------------------------------------------------------
    # Journal helpers
    # ------------------------------------------------------------------

    def _write_journal(self, run_dir: str, run_id: str, journal: _Journal) -> None:
        try:
            atomic_replace_write(_journal_path(run_dir), journal.to_payload(run_id))
        except MlflowException:
            raise
        except OSError as exc:
            raise MlflowException(
                f"Failed to write merge journal for run '{run_id}': {exc}",
                INTERNAL_ERROR,
            ) from exc
