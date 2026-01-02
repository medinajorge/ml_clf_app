#!/usr/bin/env python3
"""
Deep Trajectory Classifier - GUI Application
Main entry point for the graphical user interface.
"""

import sys
from tkinter import messagebox
import ttkbootstrap as ttkb
from pathlib import Path

# Fix for PyInstaller with console=False
if getattr(sys, 'frozen', False):
    import io
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.gui_app import DeepTrajectoryClassifierApp

def main():
    """Launch the GUI application."""
    try:
        # Create main window with modern theme
        root = ttkb.Window(
            themename="cosmo"  # Options: cosmo, flatly, litera, minty, lumen, sandstone, yeti, pulse, united, morph, journal, darkly, superhero, solar, cyborg, vapor, simplex, cerculean
        )

        # Set application icon (if available)
        icon_path = Path("assets/icon.ico")
        if icon_path.exists():
            root.iconbitmap(icon_path)

        # Create application
        app = DeepTrajectoryClassifierApp(root)

        # Handle window close
        def on_closing():
            if app.is_processing:
                if messagebox.askyesno(
                    "Confirm Exit",
                    "Classification is in progress. Are you sure you want to exit?"
                ):
                    root.destroy()
            else:
                root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_closing)

        # Center window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f'{width}x{height}+{x}+{y}')

        # Start the application
        root.mainloop()

    except Exception as e:
        messagebox.showerror(
            "Application Error",
            f"Failed to start application:\n{str(e)}"
        )
        sys.exit(1)

if __name__ == "__main__":
    main()
