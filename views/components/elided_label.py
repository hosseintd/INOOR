from PyQt5 import QtCore, QtWidgets


class ElidedLabel(QtWidgets.QLabel):
    """
    One-line label that shortens its text with "..." to fit its width, instead
    of forcing the window wider. The full text stays in the tooltip.

    A plain QLabel's minimum width is its whole text. An 85-character KL4040
    filename made the Multi Photometry window 1397 px wide at minimum, so on a
    1280 px half-screen the right-hand panel was cut off and could not be
    brought back by resizing.
    """

    def __init__(self, text="", parent=None, mode=QtCore.Qt.ElideMiddle):
        super().__init__(parent)
        self._full_text = ""
        self._mode = mode
        # Ignored: the layout decides the width; the text adapts to it.
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored,
                           QtWidgets.QSizePolicy.Preferred)
        self.setText(text)

    def setText(self, text):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._elide()

    def fullText(self):
        return self._full_text

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._elide()

    def _elide(self):
        width = max(0, self.contentsRect().width())
        super().setText(self.fontMetrics().elidedText(
            self._full_text, self._mode, width))
