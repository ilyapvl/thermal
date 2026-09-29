import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def read_field(path: str) -> dict:
    with open(path, "rb") as f:
        Nx, Ny = struct.unpack("<II", f.read(8))
        Lx, Ly = struct.unpack("<dd", f.read(16))

        Tb, Tr, Tt, Tl = struct.unpack("<dddd", f.read(32))

        raw = f.read(Nx * Ny * 8)

        T = np.frombuffer(raw, dtype="<f8").reshape(Ny, Nx).copy()

    return dict(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, T_bottom=Tb, T_right=Tr, T_top=Tt, T_left=Tl, T=T)




def plot(field: dict, out_png: str = "temperature.png") -> None:
    Nx, Ny = field["Nx"], field["Ny"]
    Lx, Ly = field["Lx"], field["Ly"]
    T = field["T"]

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(
        f"T: bottom={field['T_bottom']:.1f}, right={field['T_right']:.1f}, "
        f"top={field['T_top']:.1f}, left={field['T_left']:.1f}"
    )
    pm = ax.pcolormesh(X, Y, T, cmap="inferno", shading="nearest")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(pm, ax=ax, label="T")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"saved {out_png}")
    plt.show()

if __name__ == "__main__":
    plot(read_field("field.bin"))
