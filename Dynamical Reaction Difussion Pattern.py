import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from numba import jit

@jit(nopython=True)
def manual_roll(arr, shift, axis):
    result = np.empty_like(arr)
    if axis == 0:
        if shift > 0:
            result[shift:] = arr[:-shift]
            result[:shift] = arr[-shift:]
        else:
            result[:shift] = arr[-shift:]
            result[shift:] = arr[:-shift]
    else:
        if shift > 0:
            result[:, shift:] = arr[:, :-shift]
            result[:, :shift] = arr[:, -shift:]
        else:
            result[:, :shift] = arr[:, -shift:]
            result[:, shift:] = arr[:, :-shift]
    return result

@jit(nopython=True)
def laplacian_jit(Z):
    return (
        manual_roll(Z, 1, 0) +
        manual_roll(Z, -1, 0) +
        manual_roll(Z, 1, 1) +
        manual_roll(Z, -1, 1) - 4 * Z
    )

@jit(nopython=True)
def update_jit(U, V, Du, Dv, f, k):
    Lu = laplacian_jit(U)
    Lv = laplacian_jit(V)
    uvv = U * V * V
    U += (Du * Lu - uvv + f * (1 - U)) * 0.9
    V += (Dv * Lv + uvv - (f + k) * V) * 0.9
    U = np.clip(U, 0, 1)
    V = np.clip(V, 0, 1)
    return U, V

class ReactionDiffusionSystem:
    def __init__(self, size=200, Du=0.16, Dv=0.08, f=0.035, k=0.065):
        self.size = size
        self.Du = Du
        self.Dv = Dv
        self.f = f
        self.k = k
        self.reset()

    def reset(self):
        self.U = np.random.uniform(0.5, 1.0, (self.size, self.size))
        self.V = np.random.uniform(0.0, 0.2, (self.size, self.size))
        for _ in range(10):
            x, y = np.random.randint(0, self.size, 2)
            self.U[x-3:x+3, y-3:y+3] = 0.5
            self.V[x-3:x+3, y-3:y+3] = 0.25

    def update(self):
        self.U, self.V = update_jit(self.U, self.V, self.Du, self.Dv, self.f, self.k)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reaction-Diffusion Simulator")
        self.geometry("1300x800")

        self.rd_system = ReactionDiffusionSystem()
        self.setup_ui()

    def setup_ui(self):
        self.fig, (self.ax, self.ax_params) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
        plt.close(self.fig)  # Close the figure to prevent a separate window from opening

        self.im = self.ax.imshow(self.rd_system.U, cmap='viridis', animated=True, vmin=0, vmax=1)
        self.fig.colorbar(self.im, ax=self.ax)
        self.ax.set_title("Reaction-Diffusion Pattern")

        self.param_lines = {}
        for param in ['Du', 'Dv', 'f', 'k']:
            self.param_lines[param], = self.ax_params.plot([], [], label=param)
        self.ax_params.legend()
        self.ax_params.set_xlim(0, 100)
        self.ax_params.set_ylim(0, 0.2)
        self.ax_params.set_title("Parameter History")
        self.ax_params.set_xlabel("Time")
        self.ax_params.set_ylabel("Parameter Value")
        self.param_history = {param: [] for param in ['Du', 'Dv', 'f', 'k']}

        control_frame = ttk.Frame(self)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Label(control_frame, text="Simulation Controls", font=('Helvetica', 16, 'bold')).pack(pady=10)

        self.param_vars = {}
        self.value_labels = {}
        for param, initial, step in [('Du', 0.16, 0.001), ('Dv', 0.08, 0.001), ('f', 0.035, 0.0001), ('k', 0.065, 0.0001)]:
            frame = ttk.Frame(control_frame)
            frame.pack(pady=5)
            
            ttk.Label(frame, text=f"{param}:").pack(side=tk.LEFT)
            
            minus_btn = ttk.Button(frame, text="-", width=3, command=lambda p=param, s=-step: self.adjust_param(p, s))
            minus_btn.pack(side=tk.LEFT, padx=(5, 0))
            
            var = tk.DoubleVar(value=initial)
            self.param_vars[param] = var
            scale = ttk.Scale(frame, from_=0.001, to=0.2, value=initial, variable=var, command=lambda v, p=param: self.update_param(p))
            scale.pack(side=tk.LEFT, padx=5)
            
            plus_btn = ttk.Button(frame, text="+", width=3, command=lambda p=param, s=step: self.adjust_param(p, s))
            plus_btn.pack(side=tk.LEFT)
            
            value_label = ttk.Label(frame, text=f"{initial:.4f}")
            value_label.pack(side=tk.LEFT, padx=5)
            self.value_labels[param] = value_label

        self.cmap_var = tk.StringVar(value='viridis')
        ttk.Label(control_frame, text="Color Map").pack()
        cmap_combo = ttk.Combobox(control_frame, textvariable=self.cmap_var, values=['viridis', 'plasma', 'inferno', 'magma'])
        cmap_combo.pack()
        cmap_combo.bind('<<ComboboxSelected>>', self.update_cmap)

        ttk.Button(control_frame, text="Reset Simulation", command=self.reset).pack(pady=5)
        ttk.Button(control_frame, text="Save State", command=self.save_state).pack(pady=5)
        ttk.Button(control_frame, text="Load State", command=self.load_state).pack(pady=5)
        ttk.Button(control_frame, text="Help", command=self.show_help).pack(pady=5)

        self.animation = FuncAnimation(self.fig, self.update, frames=200, interval=50, blit=True)

    def update(self, frame):
        for _ in range(20):
            self.rd_system.update()
        self.im.set_array(self.rd_system.U)

        for param in ['Du', 'Dv', 'f', 'k']:
            self.param_history[param].append(getattr(self.rd_system, param))
            if len(self.param_history[param]) > 100:
                self.param_history[param].pop(0)
            self.param_lines[param].set_data(range(len(self.param_history[param])), self.param_history[param])

        self.ax_params.relim()
        self.ax_params.autoscale_view()

        return [self.im] + list(self.param_lines.values())

    def adjust_param(self, param, step):
        current_value = self.param_vars[param].get()
        new_value = max(0.001, min(0.2, current_value + step))
        self.param_vars[param].set(new_value)
        self.update_param(param)

    def update_param(self, param):
        value = self.param_vars[param].get()
        setattr(self.rd_system, param, value)
        self.value_labels[param].config(text=f"{value:.4f}")

    def update_cmap(self, event):
        self.im.set_cmap(self.cmap_var.get())

    def reset(self):
        self.rd_system.reset()
        initial_values = {'Du': 0.16, 'Dv': 0.08, 'f': 0.035, 'k': 0.065}
        for param, initial in initial_values.items():
            self.param_vars[param].set(initial)
            setattr(self.rd_system, param, initial)
            self.value_labels[param].config(text=f"{initial:.4f}")

    def save_state(self):
        state = {
            'U': self.rd_system.U.tolist(),
            'V': self.rd_system.V.tolist(),
            'Du': self.rd_system.Du,
            'Dv': self.rd_system.Dv,
            'f': self.rd_system.f,
            'k': self.rd_system.k
        }
        with open('rd_state.json', 'w') as f:
            json.dump(state, f)
        messagebox.showinfo("Save State", "Simulation state saved successfully!")

    def load_state(self):
        try:
            with open('rd_state.json', 'r') as f:
                state = json.load(f)
            self.rd_system.U = np.array(state['U'])
            self.rd_system.V = np.array(state['V'])
            for param in ['Du', 'Dv', 'f', 'k']:
                value = state[param]
                setattr(self.rd_system, param, value)
                self.param_vars[param].set(value)
                self.value_labels[param].config(text=f"{value:.4f}")
            messagebox.showinfo("Load State", "Simulation state loaded successfully!")
        except FileNotFoundError:
            messagebox.showerror("Error", "No saved state found.")

    def show_help(self):
        help_text = """
        Reaction-Diffusion Simulator Help:

        1. The top graph shows the current reaction-diffusion pattern.
        2. The bottom graph shows the history of parameter values.
        3. Adjust parameters using the sliders or +/- buttons:
           - Du: Diffusion rate of U
           - Dv: Diffusion rate of V
           - f: Feed rate
           - k: Kill rate
        4. Use +/- buttons for fine-tuning parameters.
        5. Change the color map using the dropdown menu.
        6. Use the buttons to reset the simulation, save/load states, or view this help.

        Experiment with different parameter values to create various patterns!
        Adjust parameters slowly to avoid pattern collapse.
        """
        messagebox.showinfo("Help", help_text)

if __name__ == "__main__":
    app = App()
    app.mainloop()