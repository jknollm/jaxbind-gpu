"""Suballocate temporary CuPy buffers from an XLA-owned FFI scratch output."""

import bisect


class RecordingScratch:
    """Share bounded temporaries between graphs queued on the same XLA stream.

    Capture records work without executing it. Replays use stream order, so a
    temporary can serve many recordings without duplicating its allocation.
    Each graph keeps this owner alive through its CUDA user object.
    """

    def __init__(self, allocator, device, limit):
        self.allocator, self.device, self.limit = allocator, device, limit
        self.blocks = []
        self.free = set()
        self.epoch = 0
        self.nbytes = 0

    def reset(self):
        self.epoch += 1
        self.free = set(range(len(self.blocks)))

    def contains(self, pointer, size):
        return size == 0 or any(
            memory.ptr <= pointer and pointer + size <= memory.ptr + capacity
            for memory, capacity in self.blocks
        )

    def allocate(self, size):
        import cupy as cp

        candidates = [i for i in self.free if self.blocks[i][1] >= size]
        if candidates:
            index = min(candidates, key=lambda i: self.blocks[i][1])
            self.free.remove(index)
        else:
            if self.nbytes + ((size + 511) // 512 * 512) > self.limit:
                return self.allocator(size)
            memory = self.allocator(size)
            index = len(self.blocks)
            self.blocks.append((memory, size))
            self.nbytes += memory.mem.size
        memory, _ = self.blocks[index]
        owner = _RecordingLoan(self, index)
        borrowed = cp.cuda.UnownedMemory(memory.ptr, size, owner, device_id=self.device)
        return cp.cuda.MemoryPointer(borrowed, 0)


class _RecordingLoan:
    def __init__(self, scratch, index):
        self.scratch, self.index, self.epoch = scratch, index, scratch.epoch

    def __del__(self):
        if self.epoch == self.scratch.epoch:
            self.scratch.free.add(self.index)


class _Allocation:
    def __init__(self, arena, offset, size):
        self.arena, self.offset, self.size = arena, offset, size

    def __del__(self):
        self.arena.release(self.offset, self.size)


class Workspace:
    def __init__(self, buffer, device):
        self.buffer, self.device = buffer, device
        self.free = [(0, buffer.nbytes)]
        self.peak_bytes = 0
        self.active_bytes = 0

    def allocate(self, size):
        import cupy as cp

        requested = size
        if not size:
            memory = cp.cuda.UnownedMemory(
                self.buffer.data.ptr, 0, self, device_id=self.device
            )
            return cp.cuda.MemoryPointer(memory, 0)
        size = (size + 511) // 512 * 512
        for i, (offset, available) in enumerate(self.free):
            if available < size:
                continue
            if available == size:
                del self.free[i]
            else:
                self.free[i] = offset + size, available - size
            self.active_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.active_bytes)
            owner = _Allocation(self, offset, size)
            memory = cp.cuda.UnownedMemory(
                self.buffer.data.ptr + offset, requested, owner, device_id=self.device
            )
            return cp.cuda.MemoryPointer(memory, 0)
        raise MemoryError(
            f"XLA workspace exhausted: requested {requested} bytes, capacity {self.buffer.nbytes}; increase workspace_bytes"
        )

    def release(self, offset, size):
        if not size:
            return
        self.active_bytes -= size
        i = bisect.bisect_left(self.free, (offset, size))
        if i and self.free[i - 1][0] + self.free[i - 1][1] == offset:
            i -= 1
            start, length = self.free.pop(i)
            offset, size = start, size + length
        if i < len(self.free) and offset + size == self.free[i][0]:
            _, length = self.free.pop(i)
            size += length
        self.free.insert(i, (offset, size))
