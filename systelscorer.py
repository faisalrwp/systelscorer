import os
import json
import csv
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Try pandas first, fallback to csv
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import ttkbootstrap as tb



SETTINGS_FILE = "settings.json"
DEFAULT_SETTINGS = {"theme": "cosmo"}


# ---------------------------------------------------------
# SETTINGS HANDLING
# ---------------------------------------------------------
def load_settings():
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, "w") as f:
            json.dump(DEFAULT_SETTINGS, f, indent=4)
        return DEFAULT_SETTINGS
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)


settings = load_settings()


# ---------------------------------------------------------
# TCRI CALC + NORMALIZATION
# ---------------------------------------------------------
def normalize(value, min_val, max_val, invert=False):
    norm = (value - min_val) / (max_val - min_val)
    return 1 - norm if invert else norm


def calculate_tcri(data, weights):
    DAR = normalize(data["DAR"], 0, 1)
    MTTD = normalize(data["MTTD_T"], 0, 60, invert=True)
    ARAT = normalize(data["ARAT"], 0, 30, invert=True)
    DRE = normalize(data["DRE"], 0, 1)
    CWRT = normalize(data["CWRT"], 0, 300, invert=True)

    return round(
        DAR * weights["DAR"] +
        MTTD * weights["MTTD_T"] +
        ARAT * weights["ARAT"] +
        DRE * weights["DRE"] +
        CWRT * weights["CWRT"],
        4
    )


# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------
class ResilienceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Telemetry Cyber Resilience Calculator")

        self.style = tb.Style(theme=settings["theme"])

        # Save last loaded CSV for exporting
        self.last_loaded_file = None
        self.csv_has_weights = False

        # Notebook
        self.notebook = tb.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True)

        self.tab_main = tb.Frame(self.notebook)
        self.tab_info = tb.Frame(self.notebook)
        self.tab_settings = tb.Frame(self.notebook)
        self.tab_graphs = tb.Frame(self.notebook)

        self.notebook.add(self.tab_main, text="Calculator")
        self.notebook.add(self.tab_info, text="Info")
        self.notebook.add(self.tab_graphs, text="Graphs")
        self.notebook.add(self.tab_settings, text="Settings")

        self.build_main_tab()
        self.build_info_tab()
        self.build_graphs_tab()
        self.build_settings_tab()


    # ========================================================================
    # MAIN TAB
    # ========================================================================
    def build_main_tab(self):
        frame = tb.Frame(self.tab_main, padding=15)
        frame.pack(fill="both", expand=True)

        tb.Label(
            frame,
            text="Telemetry Cyber Resilience Calculator",
            font=("Helvetica", 16, "bold")
        ).pack(pady=10)

        form = tb.Frame(frame)
        form.pack(fill="x", pady=5)

        # CSName
        row_cs = tb.Frame(form)
        row_cs.pack(fill="x", pady=5)

        tb.Label(row_cs, text="CSName", width=12).pack(side="left")
        self.csname_entry = tb.Entry(row_cs, width=20)
        self.csname_entry.pack(side="left")

        # Metrics + Weights
        self.metric_entries = {}
        self.weight_entries = {}
        self.weight_sliders = {}

        metrics = [
            ("DAR", 0, 1),
            ("MTTD_T", 0, 60),
            ("ARAT", 0, 30),
            ("DRE", 0, 1),
            ("CWRT", 0, 300)
        ]

        self.weight_vars = {
            "DAR": tk.DoubleVar(value=0.2),
            "MTTD_T": tk.DoubleVar(value=0.2),
            "ARAT": tk.DoubleVar(value=0.2),
            "DRE": tk.DoubleVar(value=0.2),
            "CWRT": tk.DoubleVar(value=0.2),
        }

        for name, mn, mx in metrics:
            row = tb.Frame(form)
            row.pack(fill="x", pady=5)

            tb.Label(row, text=name, width=12).pack(side="left")

            ent = tb.Entry(row, width=10)
            ent.pack(side="left", padx=5)
            self.metric_entries[name] = ent

            tb.Label(row, text="Weight:", width=10).pack(side="left")

            slider = tb.Scale(
                row, from_=0, to=1, orient="horizontal",
                variable=self.weight_vars[name], length=200,
                command=lambda event, n=name: self.update_weight_from_slider(n)
            )
            slider.pack(side="left", padx=5)
            self.weight_sliders[name] = slider

            w_ent = tb.Entry(row, width=6)
            w_ent.insert(0, "0.2")
            w_ent.pack(side="left", padx=5)
            w_ent.bind("<KeyRelease>", lambda e, n=name: self.update_slider_from_entry(n))
            self.weight_entries[name] = w_ent

        # Control Buttons
        control_frame = tb.Frame(frame)
        control_frame.pack(fill="x", pady=10)

        self.total_weight_label = tb.Label(
            control_frame,
            text="Total Weight: 1.0",
            font=("Helvetica", 12)
        )
        self.total_weight_label.pack(side="left", padx=10)

        tb.Button(
            control_frame, text="Add Row", bootstyle="primary",
            command=self.add_row
        ).pack(side="left", padx=10)

        tb.Button(
            control_frame, text="Remove Row", bootstyle="danger",
            command=self.remove_row
        ).pack(side="left", padx=10)

        tb.Button(
            control_frame, text="Load CSV", bootstyle="info",
            command=self.load_csv
        ).pack(side="left", padx=10)

        self.export_btn = tb.Button(
            control_frame, text="Export CSV", bootstyle="secondary",
            command=self.export_csv, state="disabled"
        )
        self.export_btn.pack(side="left", padx=10)

        self.process_btn = tb.Button(
            control_frame, text="Process Data", bootstyle="success",
            state="disabled",
            command=self.process_all_rows
        )
        self.process_btn.pack(side="right")

        # Data table
        table_frame = tb.Labelframe(frame, text="Data Rows", padding=10)
        table_frame.pack(fill="both", expand=True, pady=10)

        columns = ("CSName", "DAR", "MTTD_T", "ARAT", "DRE", "CWRT", "TCRI")
        self.tree = tb.Treeview(table_frame, columns=columns, show="headings")
        self.tree.pack(fill="both", expand=True)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100)

        self.update_total_weight()


    # ========================================================================
    # MANUAL ROW MANAGEMENT
    # ========================================================================
    def add_row(self):
        try:
            cs = self.csname_entry.get()
            vals = {
                key: float(self.metric_entries[key].get())
                for key in self.metric_entries
            }

            self.tree.insert("", "end", values=(
                cs, vals["DAR"], vals["MTTD_T"], vals["ARAT"],
                vals["DRE"], vals["CWRT"], ""
            ))
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")


    def remove_row(self):
        for sel in self.tree.selection():
            self.tree.delete(sel)


    # ========================================================================
    # CSV LOADING (WITH MANDATORY + OPTIONAL FIELDS)
    # ========================================================================
    def load_csv(self):
        file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file:
            return

        try:
            # ------------------------------
            # LOAD CSV (pandas or csv)
            # ------------------------------
            if PANDAS_AVAILABLE:
                df = pd.read_csv(file)
            else:
                with open(file, "r") as f:
                    df = list(csv.DictReader(f))
                    df = pd.DataFrame(df)  # we convert to DataFrame anyway
            # ------------------------------

            self.last_loaded_file = file
            self.csv_has_weights = False

            # Check mandatory fields
            required = ["DAR", "MTTD_T", "ARAT", "DRE", "CWRT"]
            for col in required:
                if col not in df.columns:
                    messagebox.showerror(
                        "Missing Field",
                        f"CSV file missing required field: {col}"
                    )
                    return

            # Check optional weight fields
            optional = ["DAR_WT", "MTTD_WT", "ARAT_WT", "DRE_WT", "CWRT_WT"]
            if all(col in df.columns for col in optional):
                self.csv_has_weights = True

            # Load rows into the UI
            for _, r in df.iterrows():
                self.tree.insert("", "end", values=(
                    r.get("CSName", "Unknown"),
                    float(r["DAR"]),
                    float(r["MTTD_T"]),
                    float(r["ARAT"]),
                    float(r["DRE"]),
                    float(r["CWRT"]),
                    ""
                ))

            self.export_btn.configure(state="normal")

        except Exception as e:
            messagebox.showerror("CSV Error", str(e))


    # ========================================================================
    # EXPORT CSV
    # ========================================================================
    def export_csv(self):
        if not self.last_loaded_file:
            messagebox.showerror("Error", "No CSV loaded to export.")
            return

        outfile = os.path.splitext(self.last_loaded_file)[0] + "_telemetry.csv"

        rows = []
        for row_id in self.tree.get_children():
            rows.append(self.tree.item(row_id)["values"])

        headers = ["CSName", "DAR", "MTTD_T", "ARAT", "DRE", "CWRT", "TCRI"]

        if PANDAS_AVAILABLE:
            df = pd.DataFrame(rows, columns=headers)
            df.to_csv(outfile, index=False)
        else:
            with open(outfile, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for row in rows:
                    writer.writerow(row)

        messagebox.showinfo("Export Successful", f"Saved as:\n{outfile}")


    # ========================================================================
    # PROCESS ROWS (WITH OPTIONAL CSV WEIGHT OVERRIDE)
    # ========================================================================
    def process_all_rows(self):
        # Application weights
        app_weights = {k: float(self.weight_vars[k].get()) for k in self.weight_vars}

        # Load CSV DataFrame if present
        df_csv = None
        if self.csv_has_weights and self.last_loaded_file:
            if PANDAS_AVAILABLE:
                df_csv = pd.read_csv(self.last_loaded_file)
            else:
                with open(self.last_loaded_file, "r") as f:
                    df_csv = list(csv.DictReader(f))
                    df_csv = pd.DataFrame(df_csv)

        updated_rows = []

        for row_id in self.tree.get_children():
            vals = self.tree.item(row_id)["values"]

            csname = vals[0]
            data = {
                "DAR": float(vals[1]),
                "MTTD_T": float(vals[2]),
                "ARAT": float(vals[3]),
                "DRE": float(vals[4]),
                "CWRT": float(vals[5])
            }

            if self.csv_has_weights and df_csv is not None:
                row_df = df_csv[df_csv["CSName"] == csname].iloc[0]

                weights = {
                    "DAR": float(row_df["DAR_WT"]),
                    "MTTD_T": float(row_df["MTTD_WT"]),
                    "ARAT": float(row_df["ARAT_WT"]),
                    "DRE": float(row_df["DRE_WT"]),
                    "CWRT": float(row_df["CWRT_WT"])
                }
            else:
                weights = app_weights

            tcri = calculate_tcri(data, weights)

            updated_rows.append((row_id, (
                csname, data["DAR"], data["MTTD_T"], data["ARAT"],
                data["DRE"], data["CWRT"], tcri
            )))

        # Update UI rows
        for row_id, vals in updated_rows:
            self.tree.item(row_id, values=vals)

        source = "CSV-Provided Weights" if self.csv_has_weights else "Application Weights"
        messagebox.showinfo("Completed", f"Processed using: {source}")

        self.export_btn.configure(state="normal")


    # ========================================================================
    # WEIGHT LOGIC
    # ========================================================================
    def update_total_weight(self):
        total = sum(v.get() for v in self.weight_vars.values())
        total = round(total, 4)
        self.total_weight_label.config(text=f"Total Weight: {total}")

        if total == 1:
            self.process_btn.configure(state="normal")
        else:
            self.process_btn.configure(state="disabled")

    def update_weight_from_slider(self, name):
        val = round(self.weight_vars[name].get(), 4)
        self.weight_entries[name].delete(0, tk.END)
        self.weight_entries[name].insert(0, val)
        self._enforce_limit(name)
        self.update_total_weight()

    def update_slider_from_entry(self, name):
        try:
            val = float(self.weight_entries[name].get())
            self.weight_vars[name].set(max(0, min(1, val)))
        except:
            return
        self._enforce_limit(name)
        self.update_total_weight()

    def _enforce_limit(self, name):
        total = sum(self.weight_vars[k].get() for k in self.weight_vars)
        if total > 1:
            excess = total - 1
            new_val = max(0, self.weight_vars[name].get() - excess)
            self.weight_vars[name].set(new_val)
            self.weight_entries[name].delete(0, tk.END)
            self.weight_entries[name].insert(0, round(new_val, 4))


    # ========================================================================
    # INFO TAB
    # ========================================================================
    def build_info_tab(self):
        frame = tb.Frame(self.tab_info, padding=15)
        frame.pack(fill="both", expand=True)

        text = """
Telemetry-Based Resilience Metrics
----------------------------------

DAR  = Disturbance Absorption Ratio
MTTD = Mean Time to Telemetry Detection
ARAT = Automated Response Activation Time
DRE  = Dynamic Reconfiguration Efficiency
CWRT = Critical Workflow Recovery Time

T-CRI = Weighted composite of all normalized metrics.
"""
        tb.Label(frame, text=text, justify="left", font=("Consolas", 11)).pack(anchor="w")


    # ========================================================================
    # GRAPHS TAB
    # ========================================================================
    def build_graphs_tab(self):
        frame = tb.Frame(self.tab_graphs, padding=10)
        frame.pack(fill="both", expand=True)

        tb.Label(frame, text="Select Graph Type:", font=("Helvetica", 12)).pack()

        self.graph_type = tb.Combobox(
            frame,
            values=["Line", "Bar", "Scatter", "Boxplot", "Funnel"]
        )
        self.graph_type.set("Line")
        self.graph_type.pack(pady=5)

        tb.Button(
            frame, text="Generate Graph",
            bootstyle="primary", command=self.generate_graph
        ).pack(pady=5)

        tb.Button(
            frame, text="Save Graph",
            bootstyle="success", command=self.save_graph
        ).pack(pady=5)

        # Graph area
        self.graph_area = tb.Labelframe(frame, text="Graph", padding=10)
        self.graph_area.pack(fill="both", expand=True)

        self.figure = plt.Figure(figsize=(7, 5), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.graph_area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)


    def get_dataframe(self):
        """
        Returns a DataFrame or list of dicts of current tree data.
        Ensures numeric columns are properly converted.
        """
        rows = []
        for rid in self.tree.get_children():
            rows.append(self.tree.item(rid)["values"])

        columns = ["CSName", "DAR", "MTTD_T", "ARAT", "DRE", "CWRT", "TCRI"]

        if PANDAS_AVAILABLE:
            df = pd.DataFrame(rows, columns=columns)
            numeric_cols = ["DAR", "MTTD_T", "ARAT", "DRE", "CWRT", "TCRI"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            return df
        else:
            # fallback using simple list of dicts
            df_list = []
            for r in rows:
                if len(r) < 7:
                    continue
                row_dict = {
                    "CSName": r[0],
                    "DAR": float(r[1]),
                    "MTTD_T": float(r[2]),
                    "ARAT": float(r[3]),
                    "DRE": float(r[4]),
                    "CWRT": float(r[5]),
                    "TCRI": float(r[6]) if r[6] != "" else 0
                }
                df_list.append(row_dict)
            return df_list


    def generate_graph(self):
        """
        Generates a graph based on the selected type.
        Handles numeric conversion and ensures non-numeric columns don't break plotting.
        """
        df = self.get_dataframe()
        gtype = self.graph_type.get()

        # Proper emptiness check
        if PANDAS_AVAILABLE:
            if df.empty:
                messagebox.showerror("Error", "No data available for plotting.")
                return
        else:
            if not df:
                messagebox.showerror("Error", "No data available for plotting.")
                return

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        try:
            if gtype == "Line":
                if PANDAS_AVAILABLE:
                    df.plot(x="CSName", y="TCRI", ax=ax, marker="o")
                else:
                    csnames = [r["CSName"] for r in df]
                    tcri_vals = [r["TCRI"] for r in df]
                    ax.plot(csnames, tcri_vals, marker="o")
                ax.set_title("T-CRI Line Chart")

            elif gtype == "Bar":
                if PANDAS_AVAILABLE:
                    df.plot(x="CSName", y="TCRI", kind="bar", ax=ax)
                else:
                    csnames = [r["CSName"] for r in df]
                    tcri_vals = [r["TCRI"] for r in df]
                    ax.bar(csnames, tcri_vals)
                ax.set_title("T-CRI Bar Chart")

            elif gtype == "Scatter":
                if PANDAS_AVAILABLE:
                    ax.scatter(df["CSName"], df["TCRI"])
                else:
                    csnames = [r["CSName"] for r in df]
                    tcri_vals = [r["TCRI"] for r in df]
                    ax.scatter(csnames, tcri_vals)
                ax.set_title("T-CRI Scatter Chart")

            elif gtype == "Boxplot":
                if PANDAS_AVAILABLE:
                    df[["DAR", "MTTD_T", "ARAT", "DRE", "CWRT", "TCRI"]].boxplot(ax=ax)
                else:
                    data = [[r[c] for r in df] for c in ["DAR","MTTD_T","ARAT","DRE","CWRT","TCRI"]]
                    ax.boxplot(data, labels=["DAR","MTTD_T","ARAT","DRE","CWRT","TCRI"])
                ax.set_title("Metric Distribution Boxplot")

            elif gtype == "Funnel":
                if PANDAS_AVAILABLE:
                    df_s = df.sort_values("TCRI", ascending=False)
                    ax.barh(df_s["CSName"], df_s["TCRI"])
                else:
                    df_s = sorted(df, key=lambda x: x["TCRI"], reverse=True)
                    csnames = [r["CSName"] for r in df_s]
                    tcri_vals = [r["TCRI"] for r in df_s]
                    ax.barh(csnames, tcri_vals)
                ax.invert_yaxis()
                ax.set_title("T-CRI Funnel Chart")

            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Graph Error", f"Error generating graph: {e}")


    def save_graph(self):
        file = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG Image", "*.png")]
        )
        if file:
            self.figure.savefig(file)
            messagebox.showinfo("Saved", f"Graph saved:\n{file}")


    # ========================================================================
    # SETTINGS TAB
    # ========================================================================
    def build_settings_tab(self):
        frame = tb.Frame(self.tab_settings, padding=15)
        frame.pack(fill="both", expand=True)

        tb.Label(frame, text="Select Theme:", font=("Helvetica", 12)).pack()

        themes = self.style.theme_names()
        self.theme_box = tb.Combobox(frame, values=themes)
        self.theme_box.set(settings["theme"])
        self.theme_box.pack(pady=5)

        self.theme_box.bind("<<ComboboxSelected>>", self.apply_theme)

    def apply_theme(self, event=None):
        theme = self.theme_box.get()
        self.style.theme_use(theme)
        settings["theme"] = theme
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=4)


# ---------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------
if __name__ == "__main__":
    root = tb.Window(themename=settings["theme"])
    app = ResilienceApp(root)
    root.mainloop()
