"""Result of screening a flat set: every measurement, and what failed.

Deliberately shows the NUMBERS, not just a verdict. The thresholds are
judgement calls, and an operator who can see that a frame failed at 3560 ADU
against a 3500 ADU limit can decide the frame is fine; one shown only a red
cross cannot.

Nothing is applied until the operator presses Exclude. Screening advises.
"""
from PyQt5 import QtCore, QtWidgets

from views.theme import mpl_colors, set_variant, theme_manager


class FlatScreenDialog(QtWidgets.QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle("Flat Frame Screening")
        self.resize(980, 560)
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        layout = QtWidgets.QVBoxLayout(self)

        frames = self.result.get('frames', [])
        bad = [r for r in frames if not r['ok']]

        head = QtWidgets.QLabel(self.result.get('summary', ''))
        set_variant(head, "heading")
        layout.addWidget(head)

        note = QtWidgets.QLabel(
            "The spike and consensus tests compare the frames with EACH OTHER, so a "
            "set that is uniformly bad will pass them. Level and saturation are "
            "absolute and always apply.")
        note.setWordWrap(True)
        set_variant(note, "muted")
        layout.addWidget(note)

        self.table = QtWidgets.QTableWidget(len(frames), 6)
        self.table.setHorizontalHeaderLabels(
            ["Frame", "Level (ADU)", "Saturated", "Spikes", "Worst region", "Verdict"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for c in range(1, 6):
            hh.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)

        danger = mpl_colors()["danger"]
        normal = theme_manager.color("text")
        for row, r in enumerate(frames):
            blk = r.get('block_deviation')
            cells = [
                r['name'],
                "-" if r['median'] != r['median'] else f"{r['median']:.0f}",
                f"{100 * r['sat_fraction']:.3f}%" if r['sat_fraction'] == r['sat_fraction'] else "-",
                str(r['spikes']) if r['spikes'] >= 0 else "-",
                f"{100 * blk:.2f}%" if blk is not None and blk == blk else "-",
                "OK" if r['ok'] else "EXCLUDE",
            ]
            ink = _colour(danger if not r['ok'] else normal)
            for col, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if col:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                item.setForeground(ink)
                self.table.setItem(row, col, item)
            if r['reasons']:
                self.table.item(row, 0).setToolTip("\n".join(r['reasons']))
        layout.addWidget(self.table, 1)

        if bad:
            why = QtWidgets.QTextEdit()
            why.setReadOnly(True)
            why.setMaximumHeight(150)
            lines = []
            for r in bad:
                lines.append(f"{r['name']}")
                lines.extend(f"    - {w}" for w in r['reasons'])
            why.setPlainText("\n".join(lines))
            layout.addWidget(QtWidgets.QLabel("Why each frame failed:"))
            layout.addWidget(why)

        row = QtWidgets.QHBoxLayout()
        row.addStretch()
        self.btn_close = QtWidgets.QPushButton("Close, change nothing")
        self.btn_apply = QtWidgets.QPushButton(
            f"Exclude the {len(bad)} flagged frame(s)" if bad else "Nothing to exclude")
        self.btn_apply.setEnabled(bool(bad))
        set_variant(self.btn_apply, "danger" if bad else "muted")
        row.addWidget(self.btn_close)
        row.addWidget(self.btn_apply)
        layout.addLayout(row)

        self.btn_close.clicked.connect(self.reject)
        self.btn_apply.clicked.connect(self.accept)

    # ------------------------------------------------------------------
    def failed_indices(self):
        """Positions, within the screened list, of the frames that failed."""
        return {i for i, r in enumerate(self.result.get('frames', [])) if not r['ok']}


def _colour(name):
    from PyQt5 import QtGui
    return QtGui.QBrush(QtGui.QColor(name))
