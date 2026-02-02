"""gru_predictor_poc.py

PoC for your "Bonzo vs Jack" binary-sequence game.

Refined UX (as requested):
- **One editable input Text** where the user can type 0/1 and *paste motifs*.
- Input is **append-only**: cursor is forced to end; deletions/edits are blocked.
- Paste is **forced to append at end** and accepts only 0/1 (whitespace ignored).
- **Mandatory big 0 / 1 buttons** append to the end.

Prediction display (separate, disabled Text widgets):
- GRU predictions (colored hit/miss)
- Conv1D predictions (colored hit/miss)
- Ensemble predictions (colored hit/miss), weighted by a slider alpha in [0..1].

Important design rule:
- Each position i is **predicted once** (when it arrives), evaluated once, and never re-evaluated.

Requirements:
    pip install tensorflow matplotlib numpy

Run:
    python gru_predictor_poc.py
"""

from __future__ import annotations

import threading
from queue import Queue, Empty
import gc  # For explicit garbage collection
from io import BytesIO
from urllib.request import urlopen, Request
import ssl

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError("TensorFlow is required. Install via: pip install tensorflow") from e

try:
    from PIL import Image, ImageTk
except ImportError as e:
    raise ImportError("Pillow is required. Install via: pip install Pillow") from e


# -----------------------------
# Models
# -----------------------------


class BaseBinaryPredictor:
    """Common utilities for online (predict-once) binary next-step predictors."""

    def __init__(self, window_size: int = 8, epochs_per_update: int = 3):
        self.window_size = int(window_size)
        self.epochs_per_update = int(epochs_per_update)
        self.history: list[int] = []
        self.model = self._build_model()

    def reset(self) -> None:
        self.history.clear()
        # Explicitly delete old model and clear TensorFlow backend to release memory
        # Without this, TensorFlow keeps computational graphs and weights cached
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            tf.keras.backend.clear_session()
        self.model = self._build_model()

    def append(self, bit: int) -> None:
        if bit not in (0, 1):
            raise ValueError("bit must be 0 or 1")
        self.history.append(bit)

    def _build_model(self) -> tf.keras.Model:
        raise NotImplementedError

    def predict_train_at(self, index: int) -> float | None:
        if index < self.window_size:
            return None
        
        # Comment: Check if history has enough data (prevents race condition after reset)
        if len(self.history) <= index:
            return None

        # ---- 1) predict on the current window ----
        window = self.history[index - self.window_size : index]
        
        # Comment: Verify window has correct size (safety check)
        if len(window) != self.window_size:
            return None
        
        x_pred = np.array(window, dtype=np.float32).reshape(1, self.window_size, 1)
        prob = float(self.model.predict(x_pred, verbose=0)[0, 0])

        # ---- 2) build replay buffer (last 16 windows) ----
        start = max(self.window_size, index - 16)

        X, Y = [], []
        for i in range(start, index + 1):
            w = self.history[i - self.window_size : i]
            X.append(w)
            Y.append(self.history[i])

        if len(X) == 0:   # <-- key fix
            return prob
        X = np.array(X, dtype=np.float32).reshape(-1, self.window_size, 1)
        Y = np.array(Y, dtype=np.float32).reshape(-1, 1)

        # idx = np.random.permutation(len(X))
        # X, Y = X[idx], Y[idx]

        # ---- 3) train on buffer ----
        self.model.fit(
            X, Y,
            epochs=self.epochs_per_update,
            verbose=0,
            shuffle=False,
            batch_size=len(X),
        )

        return prob


class GRUPredictor(BaseBinaryPredictor):
    def __init__(self, window_size: int = 8, units: int = 16, epochs_per_update: int = 3):
        self.units = int(units)
        super().__init__(window_size=window_size, epochs_per_update=epochs_per_update)

    def _build_model(self) -> tf.keras.Model:
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.window_size, 1)),
                tf.keras.layers.GRU(self.units, return_sequences=False),
                tf.keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        opt = tf.keras.optimizers.Adam(learning_rate=6e-3)
        model.compile(optimizer=opt, loss="binary_crossentropy")
        return model


class Conv1DPredictor(BaseBinaryPredictor):
    def __init__(
        self,
        window_size: int = 8,
        filters: int = 16,
        kernel_size: int = 4,
        epochs_per_update: int = 3,
    ):
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        super().__init__(window_size=window_size, epochs_per_update=epochs_per_update)


    def _build_model(self) -> tf.keras.Model:
        # reg = tf.keras.regularizers.l2(1e-3)
        reg = tf.keras.regularizers.l2(5e-4)
        # reg = None
        opt = tf.keras.optimizers.Adam(learning_rate=6e-3)
        # opt = tf.keras.optimizers.Adam(learning_rate=3e-4)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self.window_size, 1)),
                tf.keras.layers.Conv1D(self.filters, self.kernel_size,
                                    activation="relu", padding="same",
                                    kernel_regularizer=reg,
                                    ),
                # tf.keras.layers.GlobalMaxPooling1D(),
                # tf.keras.layers.GlobalAveragePooling1D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=reg),
                tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg),
            ]
        )
        model.compile(optimizer=opt, loss="binary_crossentropy")
        return model


# -----------------------------
# Stats helpers
# -----------------------------


def compute_sma_and_cum(correct_flags: list[int | None], window: int) -> tuple[list[float], list[float]]:
    """Return (SMA, cumulative) aligned to positions.

    correct_flags[i] is 1/0 if that position was evaluated, else None.
    """
    n = len(correct_flags)
    sma: list[float] = [float("nan")] * n
    cum: list[float] = [float("nan")] * n
    seen = 0
    wins = 0
    for i in range(n):
        if correct_flags[i] is None:
            continue
        # SMA over last `window` *evaluated* positions within the last window range
        start = max(0, i - window + 1)
        chunk = [c for c in correct_flags[start : i + 1] if c is not None]
        sma[i] = float(np.mean(chunk)) if chunk else float("nan")
        wins += int(correct_flags[i])
        seen += 1
        cum[i] = wins / seen if seen else float("nan")
    return sma, cum


# -----------------------------
# UI
# -----------------------------


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        # self.root.title("GRU vs Conv1D PoC")
        self.root.title("Bonzo ⚖️ Jack the Dripper")

        # Config knobs
        self.window_size = 8
        self.epochs_per_update = 2

        # Models
        self.gru = GRUPredictor(window_size=self.window_size, units=16, epochs_per_update=self.epochs_per_update)
        self.conv = Conv1DPredictor(window_size=self.window_size, filters=16, kernel_size=4, epochs_per_update=self.epochs_per_update)

        # Canonical sequence and per-position results
        self.seq: list[int] = []
        self.gru_prob: list[float | None] = []
        self.conv_prob: list[float | None] = []
        self.gru_correct: list[int | None] = []
        self.conv_correct: list[int | None] = []
        self.ens_correct: list[int | None] = []

        # Mapping from position -> Text index (so we can update a single char later)
        self.idx_gru: list[str] = []
        self.idx_conv: list[str] = []
        self.idx_ens: list[str] = []

        # Worker queues
        self.task_q: Queue[int] = Queue()
        self.result_q: Queue[dict] = Queue()
        self.stop_event = threading.Event()
        # Lock to prevent race conditions between worker and reset
        self.data_lock = threading.Lock()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

        # Build UI
        self._build_ui()

        # Poll results
        self.root.after(40, self._poll_results)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(top, text="Can you be random? For how long…? 😂", font=("Comic Sans MS", 10), justify=tk.CENTER).pack(
            anchor="n", 
        )

        btn_row = ttk.Frame(top)
        btn_row.pack(fill=tk.X, pady=(8, 0))

        # Comment: Loading card images for buttons with yellow text
        # Create SSL context that doesn't verify certificates (for image loading)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        try:
            # Red card background for 0 button
            red_card_url = "https://clipart-library.com/new_gallery/222662_playing-card-png.png"
            req = Request(red_card_url, headers={'User-Agent': 'Mozilla/5.0'})
            red_card_data = urlopen(req, context=ssl_context).read()
            red_card_img = Image.open(BytesIO(red_card_data)).resize((80, 100), Image.Resampling.LANCZOS)
            self.red_card_photo = ImageTk.PhotoImage(red_card_img)
            
            self.btn_0 = tk.Button(
                btn_row,
                text="0",
                command=lambda: self.append_bits("0"),
                image=self.red_card_photo,
                compound=tk.CENTER,
                fg="#384872",
                font=("Courier New`", 36, "bold"),
                bd=3,
                relief=tk.RAISED,
                activeforeground="#112459"
            )
            self.btn_0.pack(side=tk.LEFT, padx=(0, 12))
        except Exception as e:
            # Fallback to simple red button
            print(f"Failed to load red card image: {e}")
            self.btn_0 = tk.Button(
                btn_row,
                text="0",
                command=lambda: self.append_bits("0"),
                width=6,
                bg="#415178",
                # fg="#2222FF",
                fg="#2243FF",
                font=("Times New Roman", 40, "bold"),
                relief=tk.RAISED,
                bd=3
            )
            self.btn_0.pack(side=tk.LEFT, padx=(0, 12), ipady=12)
        
        try:
            # Black card background for 1 button
            black_card_url = "https://clipart-library.com/images/8cxrbGE6i.jpg"
            req = Request(black_card_url, headers={'User-Agent': 'Mozilla/5.0'})
            black_card_data = urlopen(req, context=ssl_context).read()
            black_card_img = Image.open(BytesIO(black_card_data)).resize((80, 100), Image.Resampling.LANCZOS)
            self.black_card_photo = ImageTk.PhotoImage(black_card_img)
            
            self.btn_1 = tk.Button(
                btn_row,
                text="1",
                command=lambda: self.append_bits("1"),
                image=self.black_card_photo,
                compound=tk.CENTER,
                # fg="yellow",
                fg="#dd464f",
                # fg="#FF9900",
                font=("Courier New", 36, "bold"),
                bd=3,
                relief=tk.RAISED,
                activeforeground="#cf232e"
            )
            self.btn_1.pack(side=tk.LEFT, padx=(0, 12))
        except Exception as e:
            # Fallback to simple black button
            print(f"Failed to load black card image: {e}")
            self.btn_1 = tk.Button(
                btn_row,
                text="1",
                command=lambda: self.append_bits("1"),
                width=6,
                bg="#e05a61",
                fg="#2243FF",
                font=("Times New Roman", 32, "bold"),
                relief=tk.RAISED,
                bd=3
            )
            self.btn_1.pack(side=tk.LEFT, padx=(0, 12), ipady=12)
        
        # Comment: Label row with Reset button on the right side
        # Input box (editable but constrained)
        # ttk.Label(top, text="Input (append-only, supports motif paste):").pack(anchor="w", pady=(10, 0))
        # ttk.Label(top, text="Click on cards or hit 0/1 (supports motif paste, always appends at end)").pack(anchor="w", pady=(10, 0))
        
        label_row = ttk.Frame(top)
        label_row.pack(fill=tk.X, pady=(5, 0))
        # ttk.Label(label_row, text="Click cards or type 0 / 1 (paste is welcome 😉").pack(side=tk.LEFT, anchor="w")
        ttk.Label(label_row, text="Click cards or type 0 / 1 (paste is ok 😉)").pack(side=tk.LEFT, anchor="w")
        ttk.Button(label_row, text="Reset", command=self.reset).pack(side=tk.RIGHT)

        self.input_text = tk.Text(self.root, height=3, wrap="word", font=("Courier New", 14))
        self.input_text.pack(fill=tk.X, padx=6, pady=(5, 5))
        self._bind_binary_append_only(self.input_text)

        # Comment: Adding UserProportion slider after input box
        # Shows proportion of predictions from ensemble: Drummer vs Artist
        user_prop_frame = ttk.Frame(self.root)
        user_prop_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        tk.Label(user_prop_frame, text=" 👉 ", font=('Times New Roman', 12, 'normal')).pack(side=tk.LEFT, padx=(0, 0))
        # Comment: Store reference to Drummer label for dynamic font weight updates
        self.drummer_label = tk.Label(user_prop_frame, text=" Drummer ", font=('Times New Roman', 12, 'normal'))
        self.drummer_label.pack(side=tk.LEFT)
        # Comment: Loading drum icon from URL
        try:
            drum_url = "https://images.icon-icons.com/3834/PNG/512/drum_music_musical_instrument_carnival_icon_235245.png"
            req = Request(drum_url, headers={'User-Agent': 'Mozilla/5.0'})
            drum_data = urlopen(req, context=ssl_context).read()
            drum_img = Image.open(BytesIO(drum_data)).resize((20, 20), Image.Resampling.LANCZOS)
            self.drum_photo = ImageTk.PhotoImage(drum_img)
            tk.Label(user_prop_frame, image=self.drum_photo).pack(side=tk.LEFT, padx=(0, 6), pady=(0,6))
        except Exception as e:
            # Fallback if image loading fails
            print(f"Failed to load drum icon: {e}")
            tk.Label(user_prop_frame, text="🥁", font=("Segoe UI Emoji", 16)).pack(side=tk.LEFT, padx=(0, 6))
        
        # self.user_proportion_var = tk.IntVar(value=50)
        # self.user_proportion_slider = ttk.Scale(
        #     user_prop_frame,
        #     from_=0,
        #     to=100,
        #     orient=tk.HORIZONTAL,
        #     variable=self.user_proportion_var,
        #     state=tk.DISABLED
        # )
        # self.user_proportion_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        # Comment: Using separate variable for user proportion display (different from ensemble display)
        self.ens_rep_var = tk.StringVar(value="N/A yet ⚖️ 100 - N/A")
        ttk.Label(user_prop_frame, textvariable=self.ens_rep_var, font=('Times New Roman', 12, 'normal')).pack(side=tk.LEFT)

        # Comment: Loading palette icon from URL
        try:
            palette_url = "https://images.icon-icons.com/321/PNG/512/Palette_34548.png"
            req = Request(palette_url, headers={'User-Agent': 'Mozilla/5.0'})
            palette_data = urlopen(req, context=ssl_context).read()
            palette_img = Image.open(BytesIO(palette_data)).resize((20, 20), Image.Resampling.LANCZOS)
            self.palette_photo = ImageTk.PhotoImage(palette_img)
            tk.Label(user_prop_frame, image=self.palette_photo).pack(side=tk.LEFT, padx=(6, 0))
        except Exception as e:
            # Fallback if image loading fails
            print(f"Failed to load palette icon: {e}")
            tk.Label(user_prop_frame, text="🎨", font=("Segoe UI Emoji", 16)).pack(side=tk.LEFT, padx=(6, 0))

        # Comment: Store reference to Artist label for dynamic font weight updates
        self.artist_label = tk.Label(user_prop_frame, text="Artist ", font=('Times New Roman', 12, 'normal'))
        self.artist_label.pack(side=tk.LEFT, padx=(0, 0))

        # Output boxes
        out = ttk.Frame(self.root)
        out.pack(fill=tk.BOTH, expand=False, padx=10)

        
        # --- Ensemble zone ---
        ens_zone = ttk.Frame(out)
        ens_zone.pack(fill=tk.X, pady=(0, 6))


        # Ensemble mix slider (0..100)
        ens_mix = ttk.Frame(ens_zone)
        ens_mix.pack(fill=tk.X, pady=(2, 0))
        self.alpha_var = tk.IntVar(value=50)
        self.conv_pct_var = tk.StringVar(value="Ensemble repartition | Conv1D: 50%")
        ttk.Label(ens_mix, textvariable=self.conv_pct_var).pack(side=tk.LEFT, padx=(0, 6))
        self.alpha = ttk.Scale(
            ens_mix,
            from_=0,
            to=100,
            orient=tk.HORIZONTAL,
            variable=self.alpha_var,
            command=lambda _v: self._recompute_ensemble_colors(),
            length=220,
        )
        self.alpha.pack(side=tk.LEFT, padx=6)
        self.gru_pct_var = tk.StringVar(value="GRU: 50%")
        ttk.Label(ens_mix, textvariable=self.gru_pct_var).pack(side=tk.LEFT, padx=(6, 0))

        ens_head = ttk.Frame(ens_zone)
        ens_head.pack(fill=tk.X)
        ttk.Label(ens_head, text="Ensemble prediction:").pack(side=tk.LEFT)
        # self.ens_rep_var = tk.StringVar(value="Bonzo (Drummer): N/A | Jack (Hot Mess): N/A")
        # self.ens_rep_var = tk.StringVar(value=" N/A yet | 100 - N/A ")
        
        ttk.Label(ens_head, text=" Artist ").pack(side=tk.RIGHT)
        ttk.Label(ens_head, textvariable=self.ens_rep_var).pack(side=tk.RIGHT)
        ttk.Label(ens_head, text="Drummer ").pack(side=tk.RIGHT)

        self.ens_out = tk.Text(ens_zone, height=3, wrap="word", font=("Courier New", 14), state=tk.DISABLED)
        self.ens_out.pack(fill=tk.X, pady=(2, 0))

        # --- Conv1D zone (moved to second position) ---
        conv_zone = ttk.Frame(out)
        conv_zone.pack(fill=tk.X, pady=(0, 6))
        conv_head = ttk.Frame(conv_zone)
        conv_head.pack(fill=tk.X)
        ttk.Label(conv_head, text="Conv1D prediction:").pack(side=tk.LEFT)
        # self.conv_rep_var = tk.StringVar(value="Bonzo (Drummer): N/A | Jack (Hot Mess): N/A")
        self.conv_rep_var = tk.StringVar(value="N/A yet ⚖️ 100 - N/A")
        ttk.Label(conv_head, text=" Artist ").pack(side=tk.RIGHT)
        ttk.Label(conv_head, textvariable=self.conv_rep_var).pack(side=tk.RIGHT)
        ttk.Label(conv_head, text="Drummer ").pack(side=tk.RIGHT)
        self.conv_out = tk.Text(conv_zone, height=3, wrap="word", font=("Courier New", 14), state=tk.DISABLED)
        self.conv_out.pack(fill=tk.X, pady=(2, 0))

        # --- GRU zone (moved to third position) ---
        gru_zone = ttk.Frame(out)
        gru_zone.pack(fill=tk.X, pady=(0, 6))
        gru_head = ttk.Frame(gru_zone)
        gru_head.pack(fill=tk.X)
        ttk.Label(gru_head, text="GRU prediction:").pack(side=tk.LEFT)
        # ttk.Label(gru_head, text="GRU prediction: ").pack(side=tk.LEFT, padx=(0, 6))
        # self.gru_rep_var = tk.StringVar(value="Bonzo (Drummer): N/A | Jack (Hot Mess): N/A")
        self.gru_rep_var = tk.StringVar(value="N/A yet ⚖️ 100 - N/A")
        ttk.Label(gru_head, text=" Artist ").pack(side=tk.RIGHT)
        ttk.Label(gru_head, textvariable=self.gru_rep_var).pack(side=tk.RIGHT)
        ttk.Label(gru_head, text="Drummer ").pack(side=tk.RIGHT)
        # ttk.Label(gru_head, textvariable=self.gru_rep_var).pack(side=tk.LEFT)

        self.gru_out = tk.Text(gru_zone, height=3, wrap="word", font=("Courier New", 14), state=tk.DISABLED)
        self.gru_out.pack(fill=tk.X, pady=(2, 0))

        for w in (self.gru_out, self.conv_out, self.ens_out):
            w.tag_configure("hit", background="lightgreen")
            w.tag_configure("miss", background="lightcoral")
            w.tag_configure("warmup", foreground="grey")
            w.tag_configure("pending", foreground="grey")

        # Comment: Plot with maximum usable width and small inner spacing
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.fig = Figure(figsize=(10, 3.2), dpi=100)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.ax1.set_ylabel("SMA")
        self.ax2.set_ylabel("Cumulative")
        self.ax2.set_xlabel("Steps")
        self.ax1.set_ylim(0, 1)
        self.ax2.set_ylim(0, 1)
        
        # Create line objects once (reuse them to avoid memory leaks)
        # Previously: created new lines on every draw, causing Line2D objects to accumulate
        self.line_conv_sma, = self.ax1.plot([], [], label="Conv1D")
        self.line_gru_sma, = self.ax1.plot([], [], label="GRU")
        self.line_ens_sma, = self.ax1.plot([], [], label="Ensemble")
        self.line_conv_cum, = self.ax2.plot([], [], label="Conv1D")
        self.line_gru_cum, = self.ax2.plot([], [], label="GRU")
        self.line_ens_cum, = self.ax2.plot([], [], label="Ensemble")
        
        self.ax1.legend(loc="upper left")
        self.ax2.legend(loc="upper left")
        
        # Comment: Use tight_layout with small padding for maximum usable width
        self.fig.tight_layout(pad=0.5)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        def shortcut_handler(event):
            # Get the widget that currently has keyboard focus
            focused_widget = self.root.focus_get()
            
            # If focus is in any Entry widget, do nothing (allow typing)
            if isinstance(focused_widget, tk.Entry):
                return
            
            # Comment: Simulate button click with visual feedback
            if event.char == "1":
                # Visual feedback: press button down
                self.btn_1.config(relief=tk.SUNKEN)
                self.root.update()
                # Wait a bit to show the press effect
                self.root.after(100, lambda: [
                    self.btn_1.config(relief=tk.RAISED),
                    self.btn_1.invoke()  # Trigger the button's command
                ])
            elif event.char == "0":
                # Visual feedback: press button down
                self.btn_0.config(relief=tk.SUNKEN)
                self.root.update()
                # Wait a bit to show the press effect
                self.root.after(100, lambda: [
                    self.btn_0.config(relief=tk.RAISED),
                    self.btn_0.invoke()  # Trigger the button's command
                ])

        # Bind simple keys to the root window
        self.root.bind("1", shortcut_handler)
        self.root.bind("0", shortcut_handler)        

    def _make_out_box(self, parent: ttk.Frame, title: str) -> tk.Text:
        ttk.Label(parent, text=title + ":").pack(anchor="w")
        t = tk.Text(parent, height=2, wrap="word", font=("Courier New", 14), state=tk.DISABLED)
        t.pack(fill=tk.X, pady=(2, 10))
        return t

    # ---------- Input constraints (your pattern, integrated) ----------

    def _bind_binary_append_only(self, text_widget: tk.Text) -> None:
        def on_key(event):
            # Allow Ctrl+C / Ctrl+A
            if (event.state & 0x0004) and event.keysym.lower() in ("c", "a"):
                return None

            # Force cursor to end
            text_widget.mark_set("insert", "end-1c")

            # Only accept 0/1 by manually appending (so we can process immediately)
            if event.char in ("0", "1"):
                self.append_bits(event.char)
                return "break"

            # Ignore everything else (no delete, no arrows, etc.)
            return "break"

        def on_paste(_event):
            try:
                pasted = self.root.clipboard_get()
            except tk.TclError:
                return "break"

            # Keep only 0/1 from the pasted text (ignore whitespace/newlines)
            cleaned = "".join(ch for ch in pasted if ch in "01")
            if cleaned:
                self.append_bits(cleaned)
            return "break"

        def force_end(_event):
            text_widget.after_idle(lambda: text_widget.mark_set("insert", "end-1c"))

        text_widget.bind("<Key>", on_key)
        text_widget.bind("<<Paste>>", on_paste)
        text_widget.bind("<Button-1>", force_end)

    # ---------- Append bits (single source of truth) ----------

    def append_bits(self, bits: str) -> None:
        if not bits:
            return
        if any(ch not in "01" for ch in bits):
            return

        # 1) Append to input widget visually (always at end)
        self.input_text.insert("end-1c", bits)
        self.input_text.see("end-1c")

        # 2) Update canonical buffers + placeholders
        for ch in bits:
            bit = 1 if ch == "1" else 0
            
            # Brief lock: just append to lists and get index
            with self.data_lock:
                i = len(self.seq)
                self.seq.append(bit)
                self.gru.append(bit)
                self.conv.append(bit)
                self.gru_prob.append(None)
                self.conv_prob.append(None)
                self.gru_correct.append(None)
                self.conv_correct.append(None)
                self.ens_correct.append(None)

            # Append placeholders to output texts and record indices (no lock needed)
            self._out_append_placeholder(i)

            # Enqueue index for prediction/training
            self.task_q.put(i)

    def _out_append_placeholder(self, i: int) -> None:
        # For outputs, append one char aligned to input length:
        # - warmup: 'x'
        # - else pending: '?' until result arrives
        warmup = i < self.window_size
        ch = "x" if warmup else "?"
        tag = "warmup" if warmup else "pending"

        for widget, idx_list in (
            (self.gru_out, self.idx_gru),
            (self.conv_out, self.idx_conv),
            (self.ens_out, self.idx_ens),
        ):
            widget.configure(state=tk.NORMAL)
            
            insert_at = widget.index("end-1c")   # position right before the trailing newline
            widget.insert(insert_at, ch)         # insert there
            idx_list.append(insert_at)           # store exactly where we inserted
            widget.tag_add(tag, insert_at, f"{insert_at} + 1c")

            # widget.insert(tk.END, ch)
            # index = widget.index("end-1c")
            # idx_list.append(index)
            # widget.tag_add(tag, index, f"{index} + 1c")

            widget.configure(state=tk.DISABLED)

    # ---------- Worker + UI updates ----------

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                i = self.task_q.get(timeout=0.1)
            except Empty:
                continue
            if i < 0:
                continue

            # Quick lock: just validate index, then release immediately
            with self.data_lock:
                # Validate index is still valid (reset may have cleared history)
                if i >= len(self.seq):
                    continue
                
            # Predict and train WITHOUT holding lock (TensorFlow can take time)
            # Models maintain their own history internally (no need to pass seq)
            gru_p = self.gru.predict_train_at(i)
            conv_p = self.conv.predict_train_at(i)

            payload = {"i": i, "gru_p": gru_p, "conv_p": conv_p}
            self.result_q.put(payload)

    def _poll_results(self) -> None:
        changed = False
        while True:
            try:
                payload = self.result_q.get_nowait()
            except Empty:
                break

            i = int(payload["i"])
            
            # Quick lock: just read the data we need
            with self.data_lock:
                # Validate index before accessing (reset may have cleared seq)
                if i >= len(self.seq):
                    continue
                actual = self.seq[i]

            # Update GRU output (no lock needed - UI operations)
            if payload["gru_p"] is not None:
                p = float(payload["gru_p"])
                self.gru_prob[i] = p
                pred = 1 if p >= 0.5 else 0
                corr = 1 if pred == actual else 0
                self.gru_correct[i] = corr
                self._out_set_char(self.gru_out, self.idx_gru[i], str(pred), "hit" if corr else "miss")
                changed = True

            # Update Conv output
            if payload["conv_p"] is not None:
                p = float(payload["conv_p"])
                self.conv_prob[i] = p
                pred = 1 if p >= 0.5 else 0
                corr = 1 if pred == actual else 0
                self.conv_correct[i] = corr
                self._out_set_char(self.conv_out, self.idx_conv[i], str(pred), "hit" if corr else "miss")
                changed = True

            # Update Ensemble for this position (if both probs exist)
            self._update_ensemble_at(i)
            changed = True

        if changed:
            self._refresh_stats_and_plot()

        self.root.after(40, self._poll_results)

    def _out_set_char(self, widget: tk.Text, index: str, ch: str, tag: str) -> None:
        # print(f"Updating output at {index} to '{ch}' with tag '{tag}' for widget {widget.widgetName}")
        widget.configure(state=tk.NORMAL)
        widget.delete(index, f"{index} + 1c")
        widget.insert(index, ch)
        # Clear existing tags on that char, then apply
        for t in ("hit", "miss", "warmup", "pending"):
            widget.tag_remove(t, index, f"{index} + 1c")
        widget.tag_add(tag, index, f"{index} + 1c")
        widget.configure(state=tk.DISABLED)

    def _update_ensemble_at(self, i: int) -> None:
        if i < self.window_size:
            return
        gp = self.gru_prob[i]
        cp = self.conv_prob[i]
        if gp is None or cp is None:
            return
        a = float(self.alpha_var.get()) / 100.0
        p = a * float(gp) + (1.0 - a) * float(cp)
        pred = 1 if p >= 0.5 else 0
        corr = 1 if pred == self.seq[i] else 0
        self.ens_correct[i] = corr
        self._out_set_char(self.ens_out, self.idx_ens[i], str(pred), "hit" if corr else "miss")

    def _recompute_ensemble_colors(self) -> None:
        # Slider changed: recompute ensemble for all positions where probs exist.
        a = int(self.alpha_var.get())
        self.conv_pct_var.set(f"Ensemble repartition | Conv1D: {100 - a}%")
        self.gru_pct_var.set(f"GRU: {a}%")
        for i in range(len(self.seq)):
            self._update_ensemble_at(i)
        self._refresh_stats_and_plot()

    # ---------- Stats + plots ----------

    def _acc(self, flags: list[int | None]) -> float | None:
        vals = [v for v in flags if v is not None]
        if not vals:
            return None
        return float(np.mean(vals))

    def _refresh_stats_and_plot(self) -> None:
        a_gru = self._acc(self.gru_correct)
        a_conv = self._acc(self.conv_correct)
        a_ens = self._acc(self.ens_correct)
        def _set_distrib(var: tk.StringVar, acc: float | None) -> None:
            if acc is None:
                var.set("N/A yet ⚖️  100 - N/A")
                return
            drummer = int(acc * 100.0)
            hot_mess = 100 - drummer
            var.set(f"{drummer:2d}% ⚖️ {hot_mess:2d}%")
            # var.set(f"Bonzo (Drummer): {drummer:.1f}% | Jack (Hot Mess): {hot_mess:.1f}%")

        _set_distrib(self.gru_rep_var, a_gru)
        _set_distrib(self.conv_rep_var, a_conv)
        _set_distrib(self.ens_rep_var, a_ens)
        
        # Comment: Update Drummer/Artist label font weight based on ensemble accuracy
        if a_ens is not None:
            ens_pct = a_ens * 100.0
            if ens_pct >= 55.0:
                # Drummer is winning (ensemble predicts well)
                self.drummer_label.config(font=('Times New Roman', 12, 'bold'))
                self.artist_label.config(font=('Times New Roman', 12, 'normal'))
            elif ens_pct <= 45.0:
                # Artist is winning (user is unpredictable)
                self.drummer_label.config(font=('Times New Roman', 12, 'normal'))
                self.artist_label.config(font=('Times New Roman', 12, 'bold'))
            else:
                # Neither is dominant
                self.drummer_label.config(font=('Times New Roman', 12, 'normal'))
                self.artist_label.config(font=('Times New Roman', 12, 'normal'))
        else:
            # No data yet
            self.drummer_label.config(font=('Times New Roman', 12, 'normal'))
            self.artist_label.config(font=('Times New Roman', 12, 'normal'))
        
        # Comment: Update UserProportion slider based on ensemble predictions
        # if a_ens is not None:
        #     artist_proportion = (1.0 - a_ens) * 100.0
        #     self.user_proportion_var.set(int(artist_proportion))
        # else:
        #     self.user_proportion_var.set(50)

        # Plot SMA and cumulative for each model
        n = len(self.seq)
        x = list(range(n))

        gru_sma, gru_cum = compute_sma_and_cum(self.gru_correct, self.window_size)
        conv_sma, conv_cum = compute_sma_and_cum(self.conv_correct, self.window_size)
        ens_sma, ens_cum = compute_sma_and_cum(self.ens_correct, self.window_size)

        # Update existing line objects instead of recreating (prevents memory leaks)
        # Old approach (causes memory leaks - Line2D objects and render buffers accumulate):
        # self.ax1.clear()
        # self.ax2.clear()
        # self.ax1.set_ylabel("SMA")
        # self.ax2.set_ylabel("Cumulative")
        # self.ax2.set_xlabel("Step")
        # self.ax1.plot(x, gru_sma, label="GRU")
        # self.ax1.plot(x, conv_sma, label="Conv1D")
        # self.ax1.plot(x, ens_sma, label="Ensemble")
        # self.ax2.plot(x, gru_cum, label="GRU")
        # self.ax2.plot(x, conv_cum, label="Conv1D")
        # self.ax2.plot(x, ens_cum, label="Ensemble")
        # self.ax1.set_ylim(0, 1)
        # self.ax2.set_ylim(0, 1)
        # self.ax1.legend(loc="upper left")
        # self.ax2.legend(loc="upper left")
        # self.fig.tight_layout()
        # self.canvas.draw()
        
        # New approach: reuse line objects by updating their data
        self.line_gru_sma.set_data(x, gru_sma)
        self.line_conv_sma.set_data(x, conv_sma)
        self.line_ens_sma.set_data(x, ens_sma)
        self.line_gru_cum.set_data(x, gru_cum)
        self.line_conv_cum.set_data(x, conv_cum)
        self.line_ens_cum.set_data(x, ens_cum)
        
        # Update x-axis limits dynamically
        if n > 0:
            self.ax1.set_xlim(0, max(n, 1))
            self.ax2.set_xlim(0, max(n, 1))
        
        self.ax1.relim()
        self.ax2.relim()
        self.canvas.draw_idle()  # draw_idle() is more efficient than draw()

    # ---------- Reset ----------

# import os, sys, gc, subprocess
# from tkinter import messagebox

    def reset(self) -> None:
        if not messagebox.askyesno("Reset", "Reset the sequence and reinitialize models?"):
            return

        # Clear all pending tasks (worker will finish current task if any)
        # This is safe - at most 1-2 tasks are being processed
        while not self.task_q.empty():
            try:
                self.task_q.get_nowait()
            except Empty:
                break
        
        # Clear any pending results
        while not self.result_q.empty():
            try:
                self.result_q.get_nowait()
            except Empty:
                break

        # Acquire lock - worker will finish current task quickly (1-2 predictions max)
        # Then we can safely reset everything
        with self.data_lock:
            # Reset models and state
            self.seq.clear()
            self.gru_prob.clear()
            self.conv_prob.clear()
            self.gru_correct.clear()
            self.conv_correct.clear()
            self.ens_correct.clear()
            self.idx_gru.clear()
            self.idx_conv.clear()
            self.idx_ens.clear()

            self.gru.reset()
            self.conv.reset()
        
        # Force Python garbage collection to release memory immediately
        # TensorFlow models can leave references that prevent automatic cleanup
        gc.collect()

        # Clear UI widgets
        self.input_text.delete("1.0", tk.END)
        for w in (self.gru_out, self.conv_out, self.ens_out):
            w.configure(state=tk.NORMAL)
            w.delete("1.0", tk.END)
            w.configure(state=tk.DISABLED)

        # Comment: Reset status displays to initial state
        self.gru_rep_var.set("N/A yet ⚖️ 100 - N/A")
        self.conv_rep_var.set("N/A yet ⚖️ 100 - N/A")
        self.ens_rep_var.set("N/A yet ⚖️ 100 - N/A")
        a = int(self.alpha_var.get())
        self.gru_pct_var.set(f"GRU: {100 - a}%")
        self.conv_pct_var.set(f"Conv1D: {a}%")
        
        # Comment: Reset Drummer/Artist labels to normal font weight
        self.drummer_label.config(font=('Times New Roman', 12, 'normal'))
        self.artist_label.config(font=('Times New Roman', 12, 'normal'))
        
        # Clear line data instead of clearing axes (preserves line objects, prevents memory leaks)
        # Old approach (destroys and recreates artists):
        # self.ax1.clear()
        # self.ax2.clear()
        # self.canvas.draw()
        
        # New approach: just clear the data from existing line objects
        self.line_gru_sma.set_data([], [])
        self.line_conv_sma.set_data([], [])
        self.line_ens_sma.set_data([], [])
        self.line_gru_cum.set_data([], [])
        self.line_conv_cum.set_data([], [])
        self.line_ens_cum.set_data([], [])
        self.ax1.set_xlim(0, 1)
        self.ax2.set_xlim(0, 1)
        self.canvas.draw_idle()
# -----------------------------


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
