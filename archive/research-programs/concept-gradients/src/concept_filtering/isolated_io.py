"""Fail-closed reads for receipt-bound local input files."""

from __future__ import annotations

import os
import secrets
import stat
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO


class IsolatedIOError(ValueError):
    """Raised when a local input is aliased or changes during a read."""


def _file_identity(status: os.stat_result, label: str) -> tuple[int, ...]:
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise IsolatedIOError(f"{label} is not an isolated regular file")
    return (
        status.st_dev,
        status.st_ino,
        status.st_nlink,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _parent_identities(path: Path, label: str) -> tuple[tuple[int, int], ...]:
    identities: list[tuple[int, int]] = []
    try:
        for parent in path.absolute().parents:
            status = parent.lstat()
            if not stat.S_ISDIR(status.st_mode):
                raise IsolatedIOError(f"{label} must have plain parent directories")
            identities.append((status.st_dev, status.st_ino))
    except OSError as error:
        raise IsolatedIOError(f"{label} must have plain parent directories") from error
    return tuple(identities)


def _existing_parent_identities(
    path: Path, label: str
) -> tuple[tuple[Path, int, int], ...]:
    identities: list[tuple[Path, int, int]] = []
    for parent in path.absolute().parents:
        try:
            status = parent.lstat()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise IsolatedIOError(
                f"{label} must have plain parent directories"
            ) from error
        if not stat.S_ISDIR(status.st_mode):
            raise IsolatedIOError(f"{label} must have plain parent directories")
        identities.append((parent, status.st_dev, status.st_ino))
    return tuple(identities)


def prepare_plain_parent_directories(
    path: Path, label: str
) -> tuple[tuple[int, int], ...]:
    """Create missing parents without accepting an existing alias."""
    existing = _existing_parent_identities(path, label)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise IsolatedIOError(f"cannot create {label} parent: {error}") from error
    for parent, device, inode in existing:
        status = parent.lstat()
        if not stat.S_ISDIR(status.st_mode) or (status.st_dev, status.st_ino) != (
            device,
            inode,
        ):
            raise IsolatedIOError(f"{label} parent identity changed")
    return _parent_identities(path, label)


def verify_plain_parent_directories(
    path: Path, expected: tuple[tuple[int, int], ...], label: str
) -> None:
    if _parent_identities(path, label) != expected:
        raise IsolatedIOError(f"{label} parent identity changed")


def read_isolated_bytes(path: Path, label: str) -> bytes:
    """Read one single-link regular file while proving path and parent continuity."""
    message = f"{label} is not an isolated regular file"
    parents_before = _parent_identities(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as error:
        raise IsolatedIOError(message) from error
    try:
        with os.fdopen(descriptor, "rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()), label)
            value = handle.read()
            after = _file_identity(os.fstat(handle.fileno()), label)
        path_identity = _file_identity(path.lstat(), label)
        parents_after = _parent_identities(path, label)
    except OSError as error:
        raise IsolatedIOError(message) from error
    if before != after or after != path_identity or parents_before != parents_after:
        raise IsolatedIOError(f"{label} identity changed during read")
    return value


def _stream_sha256(handle: BinaryIO) -> str:
    digest = sha256()
    while chunk := handle.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_isolated_file(path: Path, label: str) -> str:
    """Stream one file hash while proving link, path, and parent continuity."""
    message = f"{label} is not an isolated regular file"
    parents_before = _parent_identities(path, label)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()), label)
            digest = _stream_sha256(handle)
            after = _file_identity(os.fstat(handle.fileno()), label)
        current = _file_identity(path.lstat(), label)
        parents_after = _parent_identities(path, label)
    except OSError as error:
        raise IsolatedIOError(message) from error
    if before != after or after != current or parents_before != parents_after:
        raise IsolatedIOError(f"{label} identity changed during hash")
    return digest


def _open_verified_parent(
    path: Path, expected: tuple[tuple[int, int], ...], label: str
) -> int:
    try:
        descriptor = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as error:
        raise IsolatedIOError(f"cannot open {label} parent: {error}") from error
    status = os.fstat(descriptor)
    if not expected or (status.st_dev, status.st_ino) != expected[0]:
        os.close(descriptor)
        raise IsolatedIOError(f"{label} parent identity changed")
    return descriptor


def _open_stage(parent: int, path: Path, label: str) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    for _ in range(32):
        name = f".{path.name}.{secrets.token_hex(12)}.tmp"
        try:
            return os.open(name, flags, 0o600, dir_fd=parent), name
        except FileExistsError:
            continue
        except OSError as error:
            raise IsolatedIOError(f"cannot stage {label}: {error}") from error
    raise IsolatedIOError(f"cannot stage {label}: temporary name collision")


def _stage_bytes(parent: int, path: Path, value: bytes, label: str) -> str:
    descriptor, name = _open_stage(parent, path, label)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        os.unlink(name, dir_fd=parent)
        raise IsolatedIOError(f"cannot stage {label}: {error}") from error
    return name


def _link_staged(parent: int, temporary: str, path: Path, label: str) -> None:
    try:
        os.link(
            temporary,
            path.name,
            src_dir_fd=parent,
            dst_dir_fd=parent,
            follow_symlinks=False,
        )
    except FileExistsError as error:
        raise IsolatedIOError(f"refusing to overwrite {label}: {path}") from error
    except OSError as error:
        raise IsolatedIOError(f"cannot publish {label}: {error}") from error


def _read_published(parent: int, path: Path, label: str) -> bytes:
    message = f"{label} is not an isolated regular file"
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=parent,
        )
        with os.fdopen(descriptor, "rb") as handle:
            before = _file_identity(os.fstat(handle.fileno()), label)
            value = handle.read()
            after = _file_identity(os.fstat(handle.fileno()), label)
        current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
    except OSError as error:
        raise IsolatedIOError(message) from error
    if before != after or after != _file_identity(current, label):
        raise IsolatedIOError(f"{label} identity changed during read")
    return value


def publish_new_bytes(
    path: Path,
    value: bytes,
    label: str,
    *,
    expected_parents: tuple[tuple[int, int], ...] | None = None,
) -> None:
    """Publish new bytes without following aliases or replacing existing output."""
    parents_before = prepare_plain_parent_directories(path, label)
    if expected_parents is not None and parents_before != expected_parents:
        raise IsolatedIOError(f"{label} parent identity changed")
    parent = _open_verified_parent(path, parents_before, label)
    temporary: str | None = None
    try:
        temporary = _stage_bytes(parent, path, value, label)
        verify_plain_parent_directories(path, parents_before, label)
        _link_staged(parent, temporary, path, label)
        os.unlink(temporary, dir_fd=parent)
        temporary = None
        observed = _read_published(parent, path, label)
        verify_plain_parent_directories(path, parents_before, label)
    finally:
        if temporary is not None:
            os.unlink(temporary, dir_fd=parent)
        os.close(parent)
    if observed != value:
        raise IsolatedIOError(f"{label} changed during publication")
