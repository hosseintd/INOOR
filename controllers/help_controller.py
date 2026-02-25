import os

class HelpController:
    def __init__(self, view):
        self.view = view
        
        # Determine paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.help_doc_path = os.path.abspath(os.path.join(
            current_dir, "..", "utils", "resources", "help", "help_document.html"
        ))

        # Initial Load
        self._load_initial_content()
        
        # Connect signals
        self.view.sidebar.currentRowChanged.connect(self._handle_category_change)

    def _load_initial_content(self):
        # The document is now generated natively in the view from a python template
        self.view.load_help_file(self.help_doc_path)

    def _handle_category_change(self, index):
        anchors = [
            "introduction",
            "master",
            "calibration",
            "photometry",
            "astrometry",
            "about"
        ]
        if 0 <= index < len(anchors):
            self.view.scroll_to_anchor(anchors[index])
