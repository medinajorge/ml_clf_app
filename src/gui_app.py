import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
import threading
import yaml
from pathlib import Path
import traceback
from typing import Optional

from .pipeline import SpeciesClassifierPipeline

class DeepTrajectoryClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Trajectory Classifier v1.0")
        self.root.geometry("800x600")
        self.root.resizable(True, True)

        # Set minimum window size
        self.root.minsize(700, 500)

        # Variables
        self.input_file_path = tk.StringVar()
        self.output_file_path = tk.StringVar()
        self.pipeline: Optional[SpeciesClassifierPipeline] = None
        self.config = None
        self.is_processing = False

        # Load configuration
        self._load_config()

        # Setup GUI
        self._setup_styles()
        self._create_widgets()

        # Load models on startup (in background)
        self._load_models_background()

    def _load_config(self):
        """Load configuration file."""
        try:
            config_path = Path('config.yaml')
            if not config_path.exists():
                messagebox.showerror(
                    "Configuration Error",
                    "config.yaml not found. Please ensure the configuration file exists."
                )
                self.root.quit()
                return

            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            messagebox.showerror(
                "Configuration Error",
                f"Failed to load configuration:\n{str(e)}"
            )
            self.root.quit()

    def _setup_styles(self):
        """Configure custom styles."""
        style = ttkb.Style()

        # Configure custom button styles
        style.configure('Action.TButton', font=('Helvetica', 11, 'bold'))
        style.configure('Info.TLabel', font=('Helvetica', 10))
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Status.TLabel', font=('Helvetica', 9))

    def _create_widgets(self):
        """Create all GUI widgets."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights for responsive design
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)

        # --- HEADER SECTION ---
        self._create_header(main_frame)

        # --- MODEL STATUS SECTION ---
        self._create_model_status_section(main_frame)

        # --- INPUT FILE SECTION ---
        self._create_input_section(main_frame)

        # --- PROGRESS SECTION ---
        self._create_progress_section(main_frame)

        # --- ACTION BUTTONS SECTION ---
        self._create_action_buttons(main_frame)

        # --- STATUS BAR ---
        self._create_status_bar(main_frame)

    def _create_header(self, parent):
        """Create header section with title."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

        # Title
        title_label = ttk.Label(
            header_frame,
            text="🚀 Deep Trajectory Classifier",
            style='Title.TLabel',
            bootstyle="primary"
        )
        title_label.grid(row=0, column=0, sticky=tk.W)

        # Version
        version_label = ttk.Label(
            header_frame,
            text="v1.0",
            style='Info.TLabel',
            bootstyle="secondary"
        )
        version_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        # Separator
        separator = ttk.Separator(parent, orient='horizontal')
        separator.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

    def _create_model_status_section(self, parent):
        """Create model loading status section."""
        status_frame = ttk.LabelFrame(
            parent,
            text="Model Status",
            padding="15",
            bootstyle="info"
        )
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        status_frame.columnconfigure(1, weight=1)

        # Status label
        self.model_status_label = ttk.Label(
            status_frame,
            text="⏳ Loading models...",
            style='Info.TLabel'
        )
        self.model_status_label.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        # Model info (initially hidden)
        self.model_info_frame = ttk.Frame(status_frame)
        self.model_info_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        self.model_info_frame.grid_remove()  # Hide initially

    def _create_input_section(self, parent):
        """Create input file selection section."""
        input_frame = ttk.LabelFrame(
            parent,
            text="Input Data",
            padding="15",
            bootstyle="primary"
        )
        input_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        input_frame.columnconfigure(1, weight=1)

        # Input file label
        ttk.Label(
            input_frame,
            text="CSV File:",
            style='Info.TLabel'
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 10))

        # Input file path entry
        self.input_entry = ttk.Entry(
            input_frame,
            textvariable=self.input_file_path,
            state='readonly',
            bootstyle="secondary"
        )
        self.input_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))

        # Add data button
        self.add_data_btn = ttk.Button(
            input_frame,
            text="📁 Add Data",
            command=self._browse_input_file,
            bootstyle="primary",
            width=15
        )
        self.add_data_btn.grid(row=0, column=2, sticky=tk.E)

        # File info label
        self.file_info_label = ttk.Label(
            input_frame,
            text="",
            style='Status.TLabel',
            bootstyle="secondary"
        )
        self.file_info_label.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 0))

    def _create_progress_section(self, parent):
        """Create progress tracking section."""
        progress_frame = ttk.LabelFrame(
            parent,
            text="Processing Status",
            padding="15",
            bootstyle="success"
        )
        progress_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        progress_frame.columnconfigure(0, weight=1)

        # Progress message
        self.progress_label = ttk.Label(
            progress_frame,
            text="Ready to classify trajectories",
            style='Info.TLabel'
        )
        self.progress_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            bootstyle="success-striped",
            length=400
        )
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.progress_bar['value'] = 0

        # Percentage label
        self.progress_percentage = ttk.Label(
            progress_frame,
            text="0%",
            style='Info.TLabel',
            bootstyle="success"
        )
        self.progress_percentage.grid(row=1, column=1, sticky=tk.E, padx=(10, 0))

    def _create_action_buttons(self, parent):
        """Create main action buttons."""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=5, column=0, pady=(0, 20))

        # Classify button (large and prominent)
        self.classify_btn = ttk.Button(
            button_frame,
            text="⚡ Classify Trajectories",
            command=self._classify_data,
            style='Action.TButton',
            bootstyle="success",
            width=30,
            state='disabled'  # Initially disabled
        )
        self.classify_btn.grid(row=0, column=0, padx=5)

        # Clear button
        self.clear_btn = ttk.Button(
            button_frame,
            text="🔄 Clear",
            command=self._clear_data,
            bootstyle="warning-outline",
            width=15
        )
        self.clear_btn.grid(row=0, column=1, padx=5)

    def _create_status_bar(self, parent):
        """Create status bar at bottom."""
        status_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        status_frame.grid(row=6, column=0, sticky=(tk.W, tk.E))

        self.status_bar_label = ttk.Label(
            status_frame,
            text="Ready",
            style='Status.TLabel',
            anchor=tk.W
        )
        self.status_bar_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)

    def _load_models_background(self):
        """Load ML models in background thread."""
        def load_models():
            try:
                self.pipeline = SpeciesClassifierPipeline(
                    self.config,
                    progress_callback=self._update_model_loading_progress
                )

                # Update UI on main thread
                self.root.after(0, self._on_models_loaded)

            except Exception as e:
                error_msg = f"Failed to load models:\n{str(e)}"
                self.root.after(0, lambda: self._on_models_failed(error_msg))

        # Start loading in background
        thread = threading.Thread(target=load_models, daemon=True)
        thread.start()

    def _update_model_loading_progress(self, message: str, percentage: int):
        """Update model loading progress (called from background thread)."""
        self.root.after(0, lambda: self._update_model_status(message, percentage))

    def _update_model_status(self, message: str, percentage: int):
        """Update model status label (main thread only)."""
        self.model_status_label.config(text=f"⏳ {message} ({percentage}%)")

    def _on_models_loaded(self):
        """Called when models are successfully loaded."""
        # Update status
        self.model_status_label.config(text="✅ Models loaded successfully")

        # Get model info
        info = self.pipeline.get_model_info()

        # Show model details
        ttk.Label(
            self.model_info_frame,
            text=f"Classes: {info['num_classes']} | Features: {info['num_features']}",
            style='Status.TLabel',
            bootstyle="info"
        ).pack(side=tk.LEFT)

        self.model_info_frame.grid()

        # Update status bar
        self._set_status("Ready to classify trajectories")

        # Enable add data button
        self.add_data_btn.config(state='normal')

    def _on_models_failed(self, error_msg: str):
        """Called when model loading fails."""
        self.model_status_label.config(text="❌ Failed to load models")
        self._set_status("Error: Models failed to load")

        messagebox.showerror("Model Loading Error", error_msg)

    def _browse_input_file(self):
        """Open file dialog to select input CSV."""
        filename = filedialog.askopenfilename(
            title="Select Input CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            parent=self.root
        )

        if filename:
            self.input_file_path.set(filename)
            self._update_file_info(filename)
            self._set_status(f"Loaded: {Path(filename).name}")

            # Enable classify button if models are loaded
            if self.pipeline is not None:
                self.classify_btn.config(state='normal')

    def _update_file_info(self, filepath: str):
        """Display information about selected file."""
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            num_rows = len(df)
            num_cols = len(df.columns)

            self.file_info_label.config(
                text=f"📊 {num_rows:,} rows × {num_cols} columns"
            )
        except Exception as e:
            self.file_info_label.config(text=f"⚠️ Could not read file: {str(e)}")

    def _classify_data(self):
        """Start classification process."""
        if not self.input_file_path.get():
            messagebox.showwarning("No Input File", "Please select an input CSV file first.")
            return

        if self.pipeline is None:
            messagebox.showerror("Models Not Loaded", "Models are still loading. Please wait.")
            return

        # Ask user where to save output
        output_file = filedialog.asksaveasfilename(
            title="Save Classification Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"{Path(self.input_file_path.get()).stem}_classified.csv",
            parent=self.root
        )

        if not output_file:
            return  # User cancelled

        self.output_file_path.set(output_file)

        # Disable buttons during processing
        self._set_processing_state(True)

        # Start classification in background thread
        thread = threading.Thread(
            target=self._run_classification,
            args=(self.input_file_path.get(), output_file),
            daemon=True
        )
        thread.start()

    def _run_classification(self, input_path: str):
        """Run classification in background thread."""
        try:
            # Process with progress updates
            result_df = self.pipeline.process_csv(input_path)
            output_path = input_path.replace('.csv', '_output.csv')

            # Success callback on main thread
            self.root.after(0, lambda: self._on_classification_complete(output_path, len(result_df)))

        except Exception as e:
            error_msg = f"Classification failed:\n{str(e)}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: self._on_classification_failed(error_msg))

    def _update_classification_progress(self, message: str, percentage: int):
        """Update progress during classification (called from background thread)."""
        self.root.after(0, lambda: self._update_progress_ui(message, percentage))

    def _update_progress_ui(self, message: str, percentage: int):
        """Update progress UI elements (main thread only)."""
        self.progress_label.config(text=message)
        self.progress_bar['value'] = percentage
        self.progress_percentage.config(text=f"{percentage}%")
        self._set_status(message)

    def _on_classification_complete(self, output_path: str, num_rows: int):
        """Called when classification completes successfully."""
        self._set_processing_state(False)

        # Reset progress
        self.progress_bar['value'] = 100
        self.progress_percentage.config(text="100%")
        self.progress_label.config(text="✅ Classification complete!")

        # Show success message
        messagebox.showinfo(
            "Success",
            f"Classification completed successfully!\n\n"
            f"Processed: {num_rows:,} rows\n"
            f"Output saved to:\n{output_path}"
        )

        self._set_status(f"Completed: {num_rows:,} rows classified")

    def _on_classification_failed(self, error_msg: str):
        """Called when classification fails."""
        self._set_processing_state(False)

        # Reset progress
        self.progress_bar['value'] = 0
        self.progress_percentage.config(text="0%")
        self.progress_label.config(text="❌ Classification failed")

        # Show error
        messagebox.showerror("Classification Error", error_msg)

        self._set_status("Error: Classification failed")

    def _clear_data(self):
        """Clear all input data and reset UI."""
        self.input_file_path.set("")
        self.output_file_path.set("")
        self.file_info_label.config(text="")
        self.progress_bar['value'] = 0
        self.progress_percentage.config(text="0%")
        self.progress_label.config(text="Ready to classify trajectories")
        self.classify_btn.config(state='disabled')
        self._set_status("Ready")

    def _set_processing_state(self, is_processing: bool):
        """Enable/disable UI elements during processing."""
        self.is_processing = is_processing
        state = 'disabled' if is_processing else 'normal'

        self.add_data_btn.config(state=state)
        self.classify_btn.config(state=state)
        self.clear_btn.config(state=state)

    def _set_status(self, message: str):
        """Update status bar."""
        self.status_bar_label.config(text=message)

def progress_callback_wrapper(gui_instance):
    """Create progress callback that works with GUI."""
    def callback(message: str, percentage: int):
        gui_instance.root.after(
            0,
            lambda: gui_instance._update_classification_progress(message, percentage)
        )
    return callback
