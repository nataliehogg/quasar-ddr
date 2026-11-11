"""Helpers for saving mock data, chains and figures."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return value.__module__ + "." + value.__qualname__
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


def configuration_id(configuration):
    payload = json.dumps(configuration, sort_keys=True, allow_nan=False,
                         separators=(",", ":"), default=_json_default)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bundle_paths(prefix):
    prefix = str(prefix)
    return {"npz": Path(prefix + "_paired_dataset.npz"),
            "metadata": Path(prefix + "_metadata.json"),
            "diagnostic": Path(prefix + "_diagnostic.png")}


@contextmanager
def mock_output_bundle(prefix, *, overwrite=False):
    """Stage all products before publishing; refuse replacement by default.

    Writers using this helper coordinate through an exclusive lock. Metadata
    is published last and binds the NPZ checksum. A crash may leave a lock or
    checksum mismatch: readers refuse these rather than accepting a mixed pair.
    This is not a filesystem-wide atomic transaction for arbitrary readers.
    """
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    destinations = _bundle_paths(prefix)
    lock = Path(str(prefix) + ".write.lock")
    with lock.open("x"):
        pass
    try:
        if not overwrite and any(p.exists() for p in destinations.values()):
            raise FileExistsError("Mock output already exists; choose another run or explicitly overwrite")
        with TemporaryDirectory(prefix=".mock-stage-", dir=prefix.parent) as tmp:
            staged = {key: Path(tmp) / path.name for key, path in destinations.items()}
            yield staged
            if not all(p.is_file() for p in staged.values()):
                raise ValueError("All NPZ, metadata and diagnostic products must be staged")
            metadata = json.loads(staged["metadata"].read_text())
            metadata["dataset_sha256"] = file_sha256(staged["npz"])
            staged["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False))
            for key in ("npz", "diagnostic", "metadata"):
                os.replace(staged[key], destinations[key])
    finally:
        lock.unlink()


def load_mock_metadata(prefix):
    """Read a companion JSON, checking NPZ identity when recorded.

    Older metadata without a checksum remains readable. Run-resume protection
    still hashes the actual NPZ and metadata files independently.
    """
    prefix = Path(prefix)
    lock = Path(str(prefix) + ".write.lock")
    if lock.exists():
        raise RuntimeError("Mock bundle is being written or has an unfinished write lock")
    paths = _bundle_paths(prefix)
    metadata = json.loads(paths["metadata"].read_text())
    expected = metadata.get("dataset_sha256")
    if expected is not None and file_sha256(paths["npz"]) != expected:
        raise ValueError("Mock NPZ checksum does not match its metadata")
    if lock.exists():
        raise RuntimeError("Mock bundle changed while reading")
    return metadata


def prepare_sampler_run(info, mock_file, metadata_file, *, resume=False, overwrite=False):
    """Bind a chain prefix to input/configuration identity before Cobaya runs."""
    if resume and overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    result = dict(info)
    result["force"] = bool(overwrite)
    result["resume"] = bool(resume)
    configuration = {k: v for k, v in result.items() if k not in {"force", "resume", "debug"}}
    identity = {"configuration": configuration,
                "dataset_sha256": file_sha256(mock_file),
                "metadata_sha256": file_sha256(metadata_file)}
    fingerprint = configuration_id(identity)
    output = Path(result["output"])
    record = Path(str(output) + ".run.json")
    artifacts = [p for p in output.parent.glob(output.name + ".*") if p != record]
    if resume:
        if not record.is_file() or not artifacts:
            raise FileNotFoundError("Resume requires existing chain output and its run identity record")
        if json.loads(record.read_text()).get("fingerprint") != fingerprint:
            raise ValueError("Cannot resume: dataset, metadata or sampler configuration changed")
        return result
    if not overwrite and (artifacts or record.exists()):
        raise FileExistsError("Chain output exists; resume the same run or explicitly overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "fingerprint": fingerprint, **identity}
    text = json.dumps(payload, indent=2, allow_nan=False, default=_json_default)
    # Exclusive creation also prevents two fresh sessions claiming one prefix.
    if not overwrite:
        with record.open("x") as handle:
            handle.write(text)
    else:
        temporary = record.with_name(record.name + ".tmp")
        with temporary.open("x") as handle:
            handle.write(text)
        os.replace(temporary, record)
    return result


def prepare_figure_output(path, *, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError("Figure output exists; choose another path or explicitly overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
