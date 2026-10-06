"""drop_page_cache.py DIR...: posix_fadvise(DONTNEED) every file under DIR so charged page cache stops counting against the
HiCache host budget (sglang counts it as used; see host_memory.available_host_memory_bytes)."""
import os, sys
n = 0
for root in sys.argv[1:]:
    for d, _, fs in os.walk(root):
        for f in fs:
            try:
                fd = os.open(os.path.join(d, f), os.O_RDONLY | os.O_NOFOLLOW); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
            except OSError:
                pass
print(n)
