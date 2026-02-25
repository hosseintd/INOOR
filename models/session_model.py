from PyQt5.QtCore import QObject, pyqtSignal
from typing import List, Optional
from .file_set import FileSet

class SessionModel(QObject):
    """
    Manages the global state of the application sessions,
    specifically the list of FileSets for the Create Master tab.
    """
    sets_changed = pyqtSignal()     # Emitted when the list of sets changes (add/remove)
    data_changed = pyqtSignal()     # Emitted when content of a set changes

    def __init__(self):
        super().__init__()
        self._sets: List[FileSet] = []
        self._current_set_index: int = -1

    def add_set(self, file_set: FileSet):
        self._sets.append(file_set)
        if self._current_set_index == -1:
            self._current_set_index = 0
        self.sets_changed.emit()

    def remove_set(self, index: int):
        if 0 <= index < len(self._sets):
            self._sets.pop(index)
            if self._current_set_index >= len(self._sets):
                self._current_set_index = len(self._sets) - 1
            self.sets_changed.emit()

    def get_sets(self) -> List[FileSet]:
        return self._sets

    def get_set(self, index: int) -> Optional[FileSet]:
        if 0 <= index < len(self._sets):
            return self._sets[index]
        return None

    def current_set_index(self) -> int:
        return self._current_set_index

    def set_current_set_index(self, index: int):
        if -1 <= index < len(self._sets):
            self._current_set_index = index
            # We might want a separate signal for selection change, but for now data_changed covers updates
            self.data_changed.emit()
    
    def get_current_set(self) -> Optional[FileSet]:
        return self.get_set(self._current_set_index)

    def trigger_update(self):
        """Manually trigger update signal (e.g. after modifying a mutable FileSet)"""
        self.data_changed.emit()
