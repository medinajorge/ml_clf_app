import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import ttkbootstrap as ttkb
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
from PIL import Image, ImageTk
import threading
import yaml
from pathlib import Path
import traceback
import platform
from typing import Optional
import mmap
import csv
import os

from .pipeline import SpeciesClassifierPipeline
from . import params

def csv_shape_mmap(filepath, has_header=True):
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        # Get column count from header
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)

        # Memory-map the file to count lines
        f.seek(0)
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            n_rows = mm.read().count(b'\n')

    if has_header:
        n_rows -= 1

    return n_rows, n_cols

class DeepTrajectoryClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Deep Trajectory Classifier v1.0")
        self.root.geometry("800x650")
        self.root.resizable(True, True)

        # Set window icon
        icon_path = os.path.join(params.ASSETS_DIR, 'dmsc_v1.png')
        icon_photo = Image.open(icon_path)
        self._icon_photo_small = ImageTk.PhotoImage(icon_photo.resize((20, 20), Image.LANCZOS))
        self._icon_photo_mid = ImageTk.PhotoImage(icon_photo.resize((80, 80), Image.LANCZOS))
        self._icon_photo = ImageTk.PhotoImage(icon_photo)
        self.root.iconphoto(True, self._icon_photo)
        self.root._icon_photo = self._icon_photo

        # Set minimum window size
        self.root.minsize(700, 550)

        # Maximize window
        self._maximize_window()

        # Variables
        self.input_file_path = tk.StringVar()
        self.output_file_path = tk.StringVar()
        self.pipeline: Optional[SpeciesClassifierPipeline] = None
        self.config = None
        self.is_processing = False

        # Configuration variables
        self.use_entropy_var = tk.StringVar(value="Yes")
        self.min_confidence_var = tk.StringVar(value="0.96")
        self.quorum_var = tk.StringVar(value="3")

        # Load configuration
        self._load_config()

        # Setup GUI
        self._setup_styles()
        self._create_widgets()
        self._create_menu_bar()

        # Load models on startup (in background)
        self._load_models_background()

    def _maximize_window(self):
        """Maximize window in a cross-platform way."""
        system = platform.system()
        if system == 'Windows':
            self.root.state('zoomed')
        else:  # Linux/Unix
            self.root.attributes('-zoomed', True)

    def _set_config(self):
        self.config = {
            'use_entropy': self.use_entropy_var.get() == "Yes",
            'c_min': float(self.min_confidence_var.get()),
            'ensemble_threshold': (int(self.quorum_var.get()) - 1) / 5,
            'overwrite': True
        }

    def _load_config(self):
        """Load configuration file."""
        config_path = os.path.join(params.ROOT, 'config.yaml')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    # Set GUI variables from loaded config
                    self.use_entropy_var.set("Yes" if loaded_config.get('use_entropy', True) else "No")
                    self.min_confidence_var.set(str(loaded_config.get('c_min', 0.96)))
                    self.quorum_var.set(str(loaded_config.get('quorum', 0.5)))
            except Exception as e:
                messagebox.showerror(
                    "Configuration Error",
                    f"Failed to load configuration:\n{str(e)}"
                )
                self.root.quit()
        else:
            messagebox.showwarning(
                "Configuration Error",
                "config.yaml not found. Loading default configuration."
            )
        self._set_config()

    def _update_config_from_gui(self):
        """Update self.config dictionary from GUI variables."""
        try:
            self._set_config()
        except ValueError as e:
            messagebox.showerror(
                "Configuration Error",
                f"Invalid configuration value:\n{str(e)}"
            )
            # Reset to defaults
            self.use_entropy_var.set("Yes")
            self.min_confidence_var.set("0.96")
            self.quorum_var.set("3")
            self.config = {
                'use_entropy': True,
                'c_min': 0.96,
                'ensemble_threshold': 0.5,
                'overwrite': True
            }

        self._set_status("Applied configuration")

    def _setup_styles(self):
        """Configure custom styles."""
        style = ttkb.Style()

        # Configure custom button styles
        style.configure('Action.TButton', font=('Helvetica', params.FONTSIZE_BUTTON, 'bold'))
        style.configure('Info.TLabel', font=('Helvetica', params.FONTSIZE_INFO))
        style.configure('Title.TLabel', font=('Helvetica', params.FONTSIZE_TITLE, 'bold'))
        style.configure('Status.TLabel', font=('Helvetica', params.FONTSIZE_STATUS))

    def _create_widgets(self):
        """Create all GUI widgets with scrollbar."""
        # Create scrolled frame
        scrolled = ScrolledFrame(self.root, autohide=True)
        scrolled.pack(fill="both", expand=True)

        # Main container with padding
        main_frame = ttk.Frame(scrolled, padding="10")
        main_frame.pack(fill="both", expand=True)

        main_frame.columnconfigure(0, weight=1)

        # --- HEADER SECTION ---
        self._create_header(main_frame)

        # --- MODEL STATUS SECTION ---
        self._create_model_status_section(main_frame)

        # --- INPUT FILE SECTION ---
        self._create_input_section(main_frame)

        # --- CONFIGURATION SECTION ---
        self._create_configuration_section(main_frame)

        # --- PROGRESS SECTION ---
        self._create_progress_section(main_frame)

        # --- ACTION BUTTONS SECTION ---
        self._create_action_buttons(main_frame)

        # --- STATUS BAR ---
        self._create_status_bar(main_frame)

    def _create_about_section(self):
        """Create and display About/Documentation window."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About - Deep Marine Species Classifier")
        about_window.geometry("700x600")
        about_window.resizable(True, True)

        # Set window icon (same as main window)
        about_window.iconphoto(True, self._icon_photo)

        # Main frame with padding
        main_frame = ttk.Frame(about_window, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))

        ttk.Label(
            header_frame,
            text="Deep Marine Species Classifier",
            style='Title.TLabel',
            bootstyle="primary"
        ).pack(anchor=tk.W)

        ttk.Label(
            header_frame,
            text="Version 1.0",
            style='Info.TLabel',
            bootstyle="secondary"
        ).pack(anchor=tk.W, pady=(5, 0))

        ttk.Separator(main_frame, orient='horizontal').pack(fill=tk.X, pady=(0, 15))

        # Scrollable text area for documentation
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbar
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget
        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=('Helvetica', params.FONTSIZE_INFO),
            padx=10,
            pady=10
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        # Documentation content
        documentation = """OVERVIEW
========
Deep Marine Species Classifier (DMSC) is a machine learning application designed to classify satellite data from marine animal species using deep learning models with confidence-based abstention.

Github repository: https://github.com/medinajorge/ml_clf_app


HOW TO USE
==========
1. Wait for models to load (displayed in Model Status section)
2. Click "Add Data" to select your input CSV file
3. (Optional) Adjust configuration parameters, and click "Apply Configuration".
4. Click "Classify Trajectories" to begin processing
5. Choose a location to save the output file
6. Wait for processing to complete


CONFIGURATION OPTIONS
=====================
Use entropy:
    Enable or disable using entropy as a feature for confidence calculations.
    - Yes: Use entropy, along with class (species) probabilities as features for the confidence model (recommended)
    - No: Use only the species probabilities as features for the confidence model.

Minimum confidence:
    The minimum confidence threshold for accepting a classification for each deep learning model.
    Below the threshold, the sample is marked as "abstained".
    Default: 0.96
    Range: 0.0 to 1.0

Minimum predictors (quorum):
    The ensemble aggregates predictions from 5 deep learning models, which can "abstain" based on the minimum confidence.
    "Quorum" is the minimum number of non-abstaining models, required to make an ensemble prediction.

    If the number of predicting models > quorum, the ensemble predicts the most frequent species across non-abstaining models.
    Otherwise, it abstains.

    Default: 3
    Options: [1, 2, 3, 4, 5]


INPUT FILE FORMAT
=================
The input CSV file should contain satellite tracking data including (at least) the following columns:
- LATITUDE: The latitude coordinate (degrees) of the animal's position.
- LONGITUDE: The longitude coordinate (degrees) of the animal's position.
- DATE_TIME: The date and time of the recorded position.
- ID: A unique identifier for the time series corresponding to each animal.
Trajectories with less than 2 observations will not be considered for classification.


OUTPUT FILE FORMAT
==================
The output CSV will contain the following columns:
- ID: species identifier
- num_observations: number of observations in the satellite track
- num_days: number of days in the satellite track
- tracking_quality: "high" if it contains at least 50 observations from at least 5 days, "low" otherwise.
- species_prediction_fold_i: i=1,2,3,4,5. 5 columns containing the predictions of the individual models.
- confidence_fold_i: i=1,2,3,4,5. 5 columns containing the confidence scores of the individual models.
- species_predicted: ensemble prediction.
- abstained:  True/False indicator of whether the ensemble abstained.


EXPECTED PERFORMANCE
===================
Notation:
- ID = In-distribution. Species the model has been trained on (see AVAILABLE SPECIES below).
- OOD = Out-of-distribution. Species the model has not been trained on.
- HQ = High-quality (trajectory contains at least 50 observations from at least 5 days)
- LQ = Low-quality (trajectory contains less than 50 observations or less than 5 days)
- Macro-accuracy: average accuracy across species.

The goal of the classifier is to provide:
- High accuracy and low abstention rates for ID species.
- High abstention rates for OOD species, since any classification would be wrong.

Expected performance for the default configuration (use entropy=True, minimum confidence=0.96, quorum=3):
a) ID data:
    - Macro-accuracy (HQ):   99.2%   (95% CI: [99.0, 100])
    - Macro-accuracy (LQ):   97.1%   (95% CI: [87.2, 100])
    - Abstention rate (HQ):  34.6%   (95% CI: [29.6, 40.7])
    - Abstention rate (LQ):  94.2%   (95% CI: [88.4, 96.2])
b) OOD data:
    - Abstention rate (HQ):  88.8%   (95% CI: [82.9, 94.1])
    - Abstention rate (LQ):  99.9%   (95% CI: [99.1, 100.0])

NOTE: Higher minimum confidence and quorum will increase the ID accuracy and OOD abstention rates,
      at the expense of increasing the ID abstention rate.


AVAILABLE SPECIES
===================
Models can predict across 74 marine animal species distributed as follows:
- Birds (31):
    Arctic Herring gull, Ascension frigatebird, Atlantic puffin, Baraus petrel, Black-browed albatross,
    Black-footed albatross, Bullers albatross, Common eider, Corys shearwater, Great shearwater, Grey-headed albatross,
    Ivory gull, King eider, Laysan albatross, Manx shearwater, Masked booby, Murphys petrel, Northern fulmar, Northern gannet,
    Red-tailed tropic bird, Sabines gull, Scopolis shearwater, Short-tailed shearwater, Sooty tern, Streaked shearwater,
    Thick-billed murre, Trindade petrel, Wandering albatross, Wedge-tailed shearwater, Western gull, White-tailed tropic bird
- Seals (13):
    Australian sea lion, California sea lion, Galapagos sea lion, Grey seal, Harbour seal, Leopard seal, Long-nosed fur seal,
    New Zealand sea lion, Northern elephant seal, Northern fur seal, Ringed seal, Southern elephant seal, Weddell seal
- Cetaceans (9):
    Beluga whale, Blue whale, Bowhead whale, Fin whale, Humpback whale, Killer whale, Narwhal,
    Short-finned pilot whale, Sperm whale
- Fishes (8):
    Blue shark, Oceanic whitetip shark, Reef manta ray, Salmon shark, Shortfin mako shark, Tiger shark,
    Whale shark, White shark
- Turtles (6): Green turtle, Hawksbill turtle, Kemps Ridley turtle, Leatherback turtle, Loggerhead turtle, Olive Ridley turtle
- Penguins (5): Adelie penguin, Chinstrap penguin, Emperor penguin, Little penguin, Macaroni penguin
- Polar bears (1): Polar bear
- Sirenians (1): Dugong


MODEL FEATURES
===================
The deep learning models use 10 features from the input data:
- Cartesian coordinates: (x, y, z)
- Day of the year: Includes decimal places involving up to seconds. Scaled to [0, 2*pi] and splitted into sine and cosine components.
- Hour angle: Splitted into sine and cosine components.
- Bathymetry: Obtained from GEBCO 2022 dataset.
              Coarse-grained to latitude-longitude cells of width 0.25º to reduce memory and time requirements.
- Time interval between observations.
- Velocity between observations.


AUTHOR
======
© 2025 Jorge Medina Hernández

For support or questions, please refer to the documentation or contact the author: medinahdezjorge@gmail.com
        """

        text_widget.insert('1.0', documentation)
        text_widget.config(state='disabled')  # Make read-only

        # Close button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))

        ttk.Button(
            button_frame,
            text="Close",
            command=about_window.destroy,
            bootstyle="secondary",
            width=15
        ).pack(side=tk.RIGHT)

    # # Add this to your header section or create a menu bar
    # def _add_about_button_to_header(self):
    #     """Add About button to the header (call this in _create_header)."""
    #     # Add this at the end of _create_header method, before the separator
    #     about_btn = ttk.Button(
    #         header_frame,
    #         text="About",
    #         command=self._create_about_section,
    #         bootstyle="info-outline",
    #         width=10
    #     )
    #     about_btn.grid(row=0, column=3, sticky=tk.E, padx=(10, 0))

    # OR create a menu bar (add this to __init__ after _create_widgets())
    def _create_menu_bar(self):
        """Create menu bar with Help menu."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._create_about_section)

    def _create_header(self, parent):
        """Create header section with title."""
        header_frame = ttk.Frame(parent)
        header_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))

        # Author
        header_frame.columnconfigure(2, weight=1)

        author_label = ttk.Label(
            header_frame,
            text="© 2025 Jorge Medina Hernández",
            style='Info.TLabel',
            bootstyle="secondary",
        )
        author_label.grid(row=0, column=2, sticky=tk.E, padx=(0, 10))

        # Title
        title_label = ttk.Label(
            header_frame,
            text="Deep Marine Species Classifier",
            style='Title.TLabel',
            bootstyle="primary"
        )
        title_label.grid(row=0, column=0, sticky=tk.W)

        # Logo/Icon
        icon_label = ttk.Label(header_frame, image=self._icon_photo_mid)
        icon_label.grid(row=0, column=0, padx=(0, 10))
        title_label.grid(row=0, column=1, sticky=tk.W)  # Adjust column

        # Version badge with background
        version_frame = ttk.Frame(header_frame, bootstyle="primary")
        version_label = ttk.Label(
            version_frame,
            text="v1.0",
            padding=(4, 2),
            bootstyle="primary-inverse"
        )
        version_label.pack()
        version_frame.grid(row=0, column=2, sticky=tk.W, padx=(10, 0))

        # Separator
        separator = ttk.Separator(parent, orient='horizontal')
        separator.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 20))

    def _create_model_status_section(self, parent):
        """Create model loading status section."""
        status_frame = ttk.LabelFrame(
            parent,
            text=" Model Status ",
            padding="15",
            bootstyle="info"
        )
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        status_frame.columnconfigure(1, weight=1)

        # Status label
        self.model_status_label = ttk.Label(
            status_frame,
            text="Loading models...",
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
            text=" Input Data ",
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
        directory_img = Image.open(os.path.join(params.ASSETS_DIR, 'directory-icon.png'))
        directory_img = directory_img.resize((20, 20), Image.LANCZOS)
        self._directory_icon = ImageTk.PhotoImage(directory_img)

        self.add_data_btn = ttk.Button(
            input_frame,
            text=' Add Data',
            image=self._directory_icon,
            compound='left',
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

    def _create_configuration_section(self, parent):
        """Create configuration options section."""
        config_frame = ttk.LabelFrame(
            parent,
            text=" Configuration ",
            padding="15",
            bootstyle="secondary"
        )
        config_frame.grid(row=4, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
        config_frame.columnconfigure(1, weight=1)

        # Use Entropy option
        ttk.Label(
            config_frame,
            text="Use entropy:",
            style='Info.TLabel'
        ).grid(row=0, column=0, sticky=tk.W, pady=5)

        entropy_combo = ttk.Combobox(
            config_frame,
            textvariable=self.use_entropy_var,
            values=["Yes", "No"],
            state="readonly",
            width=15
        )
        entropy_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Minimum Confidence option
        ttk.Label(
            config_frame,
            text="Minimum confidence:",
            style='Info.TLabel'
        ).grid(row=1, column=0, sticky=tk.W, pady=5)

        min_conf_entry = ttk.Entry(
            config_frame,
            textvariable=self.min_confidence_var,
            width=15
        )
        min_conf_entry.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Ensemble Abstention Threshold option
        ttk.Label(
            config_frame,
            text="Minimum predictors (quorum):",
            style='Info.TLabel'
        ).grid(row=2, column=0, sticky=tk.W, pady=5)

        ensemble_combo = ttk.Combobox(
            config_frame,
            textvariable=self.quorum_var,
            values=["1", "2", "3", "4", "5"],
            state='readonly',
            width=15
        )
        ensemble_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)

        # Apply button
        self.apply_config_btn = ttk.Button(
            config_frame,
            text="Apply Configuration",
            command=self._apply_configuration,
            bootstyle="info-outline",
            width=20,
            state='disabled', # disabled until models are loaded
        )
        self.apply_config_btn.grid(row=2, column=0, columnspan=2, pady=(10, 0))

    def _apply_configuration(self):
        """Apply configuration changes."""
        try:
            # Validate and update config
            self._update_config_from_gui()

            # Update pipeline if it exists
            if self.pipeline is not None:
                self.pipeline._update_config(self.config)

            self._set_status("Configuration updated successfully")
            messagebox.showinfo(
                "Configuration Updated",
                "Configuration settings have been applied successfully."
            )
        except Exception as e:
            messagebox.showerror(
                "Configuration Error",
                f"Failed to apply configuration:\n{str(e)}"
            )

    def _create_progress_section(self, parent):
        """Create progress tracking section."""
        progress_frame = ttk.LabelFrame(
            parent,
            text=" Processing Status ",
            padding="15",
            bootstyle="success"
        )
        progress_frame.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 20))
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
        button_frame.grid(row=6, column=0, pady=(0, 20))

        # Classify button (large and prominent)
        self.classify_btn = ttk.Button(
            button_frame,
            text="Classify Trajectories",
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
            text="Clear",
            command=self._clear_data,
            bootstyle="warning-outline",
            width=15
        )
        self.clear_btn.grid(row=0, column=1, padx=5)

    def _create_status_bar(self, parent):
        """Create status bar at bottom."""
        status_frame = ttk.Frame(parent, relief=tk.SUNKEN, borderwidth=1)
        status_frame.grid(row=7, column=0, sticky=(tk.W, tk.E))

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
                    progress_callback=self._update_classification_progress
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
        self.model_status_label.config(text=f"{message} ({percentage}%)")

    def _on_models_loaded(self):
        """Called when models are successfully loaded."""
        # Update status
        self.model_status_label.config(text="Models loaded successfully")

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
        self.model_status_label.config(text="Failed to load models")
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

            # Enable classify and Apply configuration button if models are loaded
            if self.pipeline is not None:
                self.classify_btn.config(state='normal')
                self.apply_config_btn.config(state='normal')

    def _update_file_info(self, filepath: str):
        """Display information about selected file."""
        try:
            num_rows, num_cols = csv_shape_mmap(filepath)

            self.file_info_label.config(
                text=f"{num_rows:,} rows × {num_cols} columns"
            )
            self._set_status(f"Loaded: {filepath}")
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

        # Update configuration before classification
        self._update_config_from_gui()
        if self.pipeline is not None:
            self.pipeline._update_config(self.config)

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

    def _run_classification(self, input_path: str, output_path: str):
        """Run classification in background thread."""
        try:
            # Process with progress updates
            result_df = self.pipeline.process_csv(self.root, self.status_bar_label, input_path, output_path)

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

        check_img = Image.open(os.path.join(params.ASSETS_DIR, 'check-icon.png'))
        check_img = check_img.resize((20, 20), Image.LANCZOS)
        self._check_img = ImageTk.PhotoImage(check_img)

        self.progress_label.config(
            image=self._check_img,
            text=" Classification complete!",
            compound='left',
        )

        # Show success message
        messagebox.showinfo(
            "Success",
            f"Classification completed successfully!\n\n"
            f"Processed: {num_rows:,} trajectories\n"
            f"Output saved to:\n{output_path}"
        )

        self._set_status(f"Completed: {num_rows:,} species classified")

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
