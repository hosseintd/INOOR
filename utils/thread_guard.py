"""
Guard against re-entrant QThread launches.

Every controller stores its worker as `self.worker = SomeQThread(...)`.
Re-triggering the action while the previous worker is still running rebinds
the attribute, dropping the only Python reference to a live QThread --
garbage collection then destroys it mid-run ("QThread: Destroyed while
thread is still running", a hard crash).

Usage at every launch site:

    if is_worker_busy(self, 'worker', view=self.view,
                      busy_msg="A bulk operation is already running."):
        return
    self.worker = BulkPhotometryWorker(...)
"""
from utils.logging_config import get_logger

log = get_logger(__name__)


def is_worker_busy(owner, attr, view=None, busy_msg=None):
    """Return True if owner.<attr> is a QThread that is still running.

    When True and view/busy_msg are provided, the user is informed why the
    action was ignored; otherwise the refusal is only logged.
    """
    worker = getattr(owner, attr, None)
    if worker is None:
        return False
    try:
        running = worker.isRunning()
    except RuntimeError:
        # Underlying C++ object already deleted -- safe to replace.
        return False
    if not running:
        return False

    log.info("Ignored re-entrant start: %s.%s is still running",
             type(owner).__name__, attr)
    if view is not None and busy_msg:
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(view, "Operation in Progress", busy_msg)
    return True


# Every worker attribute a controller may hold. Used on shutdown; a name that
# is absent on a given controller is simply skipped.
WORKER_ATTRS = ('worker', 'det_worker', 'cat_worker', 'exp_worker',
                'preload_worker', 'load_worker')


def stop_workers(owner, timeout_ms=3000):
    """Ask every live QThread on `owner` to stop, and wait for it.

    Nothing did this. The project had no closeEvent at all, so quitting during
    a long job -- a 900 s plate solve, a bulk photometry run -- tore down the
    Python object that owned a running QThread and Qt aborted the process with
    "QThread: Destroyed while thread is still running".

    Returns the number of threads that were still running when asked.
    """
    stopped = 0
    for attr in WORKER_ATTRS:
        w = getattr(owner, attr, None)
        if w is None:
            continue
        try:
            if not w.isRunning():
                continue
            stopped += 1
            # Both mechanisms: requestAbort is the co-operative flag the
            # long-running workers poll, requestInterruption is Qt's.
            if hasattr(w, 'requestAbort'):
                w.requestAbort()
            w.requestInterruption()
            w.quit()
            if not w.wait(timeout_ms):
                log.warning("%s.%s did not stop within %d ms",
                            type(owner).__name__, attr, timeout_ms)
        except RuntimeError:
            # Underlying C++ object already gone.
            continue
    return stopped
