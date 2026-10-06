"""Repeatedly drop this cgroup's page cache for the checkpoint files (weights live on the GPUs once loaded)."""
import os, sys, time
roots = sys.argv[1:]
end = time.time() + 900
while time.time() < end:
    for root in roots:
        for d, _, fs in os.walk(root):
            for f in fs:
                p = os.path.join(d, f)
                try:
                    fd = os.open(p, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
                except OSError:
                    pass
    time.sleep(5)
