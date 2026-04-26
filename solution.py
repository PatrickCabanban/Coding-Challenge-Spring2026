from __future__ import annotations

from multiprocessing import shared_memory
from typing import TypeAlias

import numpy as np


__all__ = ["SharedBuffer"]

RingView: TypeAlias = tuple[memoryview, memoryview | None, int, bool]


class SharedBuffer(shared_memory.SharedMemory):
    """
    Applicant template.

    Replace every method body with your own implementation while preserving the
    public API used by the official tests.

    The intended contract is:
    - one writer and one or more readers
    - shared state visible across processes
    - bounded storage with reusable space after readers advance
    - reads and writes report how many bytes are actually available
    """

    _NO_READER = -1

    def __init__(
        self,
        name: str,
        create: bool,
        size: int,
        num_readers: int,
        reader: int,
        cache_align: bool = False,
        cache_size: int = 64,
    ):
        """Open or create the shared buffer."""
        if size <= 0:
            raise ValueError("Buffer size must be positive")
        if num_readers < 1:
            raise ValueError("Must have at least one reader slot")
        if not (reader == self._NO_READER or 0 <= reader < num_readers):
            raise ValueError(f"Invalid reader index: {reader}")
        if cache_align and (cache_size & (cache_size - 1)) != 0:
            raise ValueError(
                f"cache_size must be a power of 2 when cache_align=True, got {cache_size}"
            )

        self.buffer_size = size
        self.num_readers = num_readers
        self.reader = reader

        # Header layout dictated by tests/support.py:
        # [0]=writer_pos, [1..5]=reserved, [6+r*3]=reader r pos,
        # [6+r*3+1]=alive flag, [6+r*3+2]=reserved.
        self._metadata_size = (6 + num_readers * 3) * 8
        total_size = self._metadata_size + size

        super().__init__(name=name, create=create, size=total_size)

        # Both header and buffer must derive from a single np.ndarray over
        # self.buf — using two separate self.buf calls broke wrap-around
        # writes across processes on Windows.
        self._shm_np = np.ndarray(shape=(total_size,), dtype=np.uint8, buffer=self.buf)
        self.header = self._shm_np[: self._metadata_size].view(np.int64)
        self.buffer = memoryview(self._shm_np[self._metadata_size :])

        if create:
            self.header.fill(0)

    def _require_reader(self) -> None:
        # Tests check for RuntimeError specifically (not ValueError).
        if self.reader == self._NO_READER:
            raise RuntimeError("Method not available on a writer-only instance")

    def close(self) -> None:
        """Release local views and close this process's handle to the shared memory."""
        # self.buffer may have already been released by drop_local_views.
        try:
            self.buffer.release()
        except Exception:
            pass
        try:
            super().close()
        except Exception:
            pass

    def __enter__(self) -> "SharedBuffer":
        """Enter the context manager."""
        if self.reader != self._NO_READER:
            self.set_reader_active(True)
        return self

    def __exit__(self, *_):
        """Exit the context manager."""
        if self.reader != self._NO_READER:
            try:
                self.set_reader_active(False)
            except Exception:
                pass
        self.close()

    def calculate_pressure(self) -> int:
        """Return current writer pressure as an integer percentage."""
        writer_pos = int(self.header[0])
        min_rpos = None
        for r in range(self.num_readers):
            slot = 6 + r * 3
            if self.header[slot + 1]:
                rpos = int(self.header[slot])
                if min_rpos is None or rpos < min_rpos:
                    min_rpos = rpos
        if min_rpos is None:
            return 0
        return int((writer_pos - min_rpos) / self.buffer_size * 100)

    def compute_max_amount_writable(self, force_rescan: bool = False) -> int:
        """Return how many bytes the writer can safely expose right now."""
        # force_rescan ignored: header is live-backed by shared memory.
        h = self.header
        writer_pos = int(h[0])
        min_rpos = None
        slot = 6
        for _ in range(self.num_readers):
            if h[slot + 1]:
                rpos = int(h[slot])
                if min_rpos is None or rpos < min_rpos:
                    min_rpos = rpos
            slot += 3
        if min_rpos is None:
            return self.buffer_size
        return max(0, self.buffer_size - (writer_pos - min_rpos))

    def int_to_pos(self, value: int) -> int:
        """Convert an absolute position counter into a position inside the bounded payload area."""
        return value % self.buffer_size

    def get_write_pos(self) -> int:
        """Return the current absolute writer position."""
        return int(self.header[0])

    def update_write_pos(self, new_writer_pos: int) -> None:
        """Store the writer's absolute write position in shared state."""
        self.header[0] = new_writer_pos

    def inc_writer_pos(self, inc_amount: int) -> None:
        """Advance the writer's absolute position by `inc_amount` bytes."""
        self.header[0] += inc_amount

    def update_reader_pos(self, new_reader_pos: int) -> None:
        """Store this reader's absolute read position in shared state."""
        self._require_reader()
        self.header[6 + self.reader * 3] = new_reader_pos

    def inc_reader_pos(self, inc_amount: int) -> None:
        """Advance this reader's absolute position by `inc_amount` bytes."""
        self._require_reader()
        self.header[6 + self.reader * 3] += inc_amount

    def set_reader_active(self, active: bool) -> None:
        """Mark this reader as active or inactive in shared state."""
        self._require_reader()
        self.header[6 + self.reader * 3 + 1] = 1 if active else 0

    def is_reader_active(self) -> bool:
        """Return whether this reader is currently marked active."""
        self._require_reader()
        return bool(self.header[6 + self.reader * 3 + 1])

    def jump_to_writer(self) -> None:
        """Move this reader directly to the current writer position."""
        self._require_reader()
        self.header[6 + self.reader * 3] = int(self.header[0])

    def expose_writer_mem_view(self, size: int) -> RingView:
        """Return a writable view tuple for up to `size` bytes."""
        h = self.header
        writer_pos = int(h[0])
        buffer_size = self.buffer_size

        min_rpos = None
        slot = 6
        for _ in range(self.num_readers):
            if h[slot + 1]:
                rpos = int(h[slot])
                if min_rpos is None or rpos < min_rpos:
                    min_rpos = rpos
            slot += 3
        if min_rpos is None:
            avail = buffer_size
        else:
            avail = buffer_size - (writer_pos - min_rpos)
            if avail < 0:
                avail = 0

        actual = size if size < avail else avail
        start = writer_pos % buffer_size
        buf = self.buffer

        if actual == 0:
            return (buf[start:start], None, 0, False)

        end = start + actual
        if end <= buffer_size:
            return (buf[start:end], None, actual, False)

        return (buf[start:], buf[0 : actual - (buffer_size - start)], actual, True)

    def expose_reader_mem_view(self, size: int) -> RingView:
        """Return a readable view tuple for up to `size` bytes."""
        self._require_reader()
        writer_pos = int(self.header[0])
        slot = 6 + self.reader * 3
        reader_pos = int(self.header[slot])

        # Stale reader: data already overwritten, snap forward to writer.
        if reader_pos < writer_pos - self.buffer_size:
            reader_pos = writer_pos
            self.header[slot] = reader_pos

        avail = min(writer_pos - reader_pos, self.buffer_size)
        actual = min(size, avail)
        start = reader_pos % self.buffer_size

        if actual == 0:
            return (self.buffer[start:start], None, 0, False)

        end = start + actual
        if end <= self.buffer_size:
            return (self.buffer[start:end], None, actual, False)

        return (
            self.buffer[start:],
            self.buffer[0 : actual - (self.buffer_size - start)],
            actual,
            True,
        )

    def simple_write(self, writer_mem_view: RingView, src: object) -> None:
        """Copy bytes from `src` into the exposed writer view(s)."""
        mv1, mv2, actual_size, split = writer_mem_view
        src_mv = memoryview(src)
        if src_mv.format != "B" or src_mv.ndim != 1:
            src_mv = src_mv.cast("B")
        src_n = src_mv.nbytes
        to_copy = src_n if src_n < actual_size else actual_size
        if not split:
            mv1[:to_copy] = src_mv[:to_copy]
        else:
            p1 = mv1.nbytes
            if to_copy <= p1:
                mv1[:to_copy] = src_mv[:to_copy]
            else:
                mv1[:] = src_mv[:p1]
                p2 = to_copy - p1
                if p2 > 0 and mv2 is not None:
                    mv2[:p2] = src_mv[p1 : p1 + p2]

    def simple_read(self, reader_mem_view: RingView, dst: object) -> None:
        """Copy bytes from the exposed reader view(s) into `dst`."""
        mv1, mv2, actual_size, split = reader_mem_view
        try:
            dst_mv = memoryview(dst)
        except TypeError:
            dst_mv = None
        if dst_mv is not None:
            if dst_mv.format != "B" or dst_mv.ndim != 1:
                dst_mv = dst_mv.cast("B")
            dst_n = dst_mv.nbytes
            to_copy = actual_size if actual_size < dst_n else dst_n
            if not split:
                dst_mv[:to_copy] = mv1[:to_copy]
            else:
                p1 = mv1.nbytes
                if to_copy <= p1:
                    dst_mv[:to_copy] = mv1[:to_copy]
                else:
                    dst_mv[:p1] = mv1[:]
                    p2 = to_copy - p1
                    if p2 > 0 and mv2 is not None:
                        dst_mv[p1 : p1 + p2] = mv2[:p2]
        else:
            to_copy = actual_size if actual_size < len(dst) else len(dst)
            if not split:
                dst[:to_copy] = mv1[:to_copy]
            else:
                p1 = mv1.nbytes
                if to_copy <= p1:
                    dst[:to_copy] = mv1[:to_copy]
                else:
                    dst[:p1] = mv1[:]
                    p2 = to_copy - p1
                    if p2 > 0 and mv2 is not None:
                        dst[p1 : p1 + p2] = mv2[:p2]

    def write_array(self, arr: np.ndarray) -> int:
        """Write a NumPy array's raw bytes into the shared buffer."""
        nbytes = arr.nbytes
        if nbytes > self.compute_max_amount_writable():
            return 0
        view = self.expose_writer_mem_view(nbytes)
        self.simple_write(view, arr)
        self.inc_writer_pos(nbytes)
        return nbytes

    def read_array(self, nbytes: int, dtype: np.dtype) -> np.ndarray:
        """Read `nbytes` from the shared buffer and interpret them as `dtype`."""
        self._require_reader()
        h = self.header
        writer_pos = int(h[0])
        slot = 6 + self.reader * 3
        reader_pos = int(h[slot])

        if reader_pos < writer_pos - self.buffer_size:
            reader_pos = writer_pos
            h[slot] = reader_pos

        avail = writer_pos - reader_pos
        if avail > self.buffer_size:
            avail = self.buffer_size
        if avail < nbytes:
            return np.empty(0, dtype=dtype)

        dt = np.dtype(dtype)
        out = np.empty(nbytes // dt.itemsize, dtype=dt)
        view = self.expose_reader_mem_view(nbytes)
        self.simple_read(view, memoryview(out).cast("B"))
        self.inc_reader_pos(nbytes)
        return out
