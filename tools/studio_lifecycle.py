"""Own a desktop backend and finish the current export before shutting down."""
import ctypes
import json
import os
import threading
import time
from pathlib import Path

from studio_project import atomic_text


def write_session(server, path):
    value = {'schema': 'ddh_desktop_session/v1', 'pid': os.getpid(), 'parent_pid': os.getppid(),
             'url': f'http://127.0.0.1:{server.server_port}/', 'token': server.token}
    atomic_text(Path(path), json.dumps(value, ensure_ascii=False))


def finish_and_stop(server):
    server.service.cancel_event.set()
    if server.service.collaboration:
        server.service.collaboration.stop()
    while server.service.job['state'] == 'running':
        time.sleep(.1)
    server.shutdown()


def request_shutdown(server):
    thread = threading.Thread(target=finish_and_stop, args=(server,), daemon=True)
    thread.start()
    return thread


def monitor_parent(server, alive, interval=1):
    while not server.stopped.wait(interval):
        if not alive():
            finish_and_stop(server)
            return


def watch_parent(server, pid):
    if os.name != 'nt':
        raise ValueError('桌面进程监护仅支持 Windows')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        raise ValueError('桌面主进程已退出或无法访问')

    def watch():
        try:
            monitor_parent(server, lambda: kernel.WaitForSingleObject(handle, 0) == 258)
        finally:
            kernel.CloseHandle(handle)

    threading.Thread(target=watch, daemon=True).start()
