# [AI]
import struct
import sys
import os
import numpy as np
import pyvista as pv


def read_field_3d(path):
    with open(path, "rb") as f:
        Nx, Ny, Nz = struct.unpack("<III", f.read(12))
        Lx, Ly, Lz = struct.unpack("<ddd", f.read(24))
        Txmin, Txmax = struct.unpack("<dd", f.read(16))
        Tymin, Tymax = struct.unpack("<dd", f.read(16))
        Tzmin, Tzmax = struct.unpack("<dd", f.read(16))

        raw = f.read(Nx * Ny * Nz * 4)
        T = np.frombuffer(raw, dtype="<f4").reshape(Nz, Ny, Nx).astype(np.float64)

    return dict(
        Nx=Nx, Ny=Ny, Nz=Nz,
        Lx=Lx, Ly=Ly, Lz=Lz,
        Txmin=Txmin, Txmax=Txmax,
        Tymin=Tymin, Tymax=Tymax,
        Tzmin=Tzmin, Tzmax=Tzmax,
        T=T,
    )


def build_grid(field):
    Nx, Ny, Nz = field["Nx"], field["Ny"], field["Nz"]
    Lx, Ly, Lz = field["Lx"], field["Ly"], field["Lz"]
    T = field["T"]

    grid = pv.ImageData(dimensions=(Nx, Ny, Nz))
    grid.spacing = (
        Lx / (Nx - 1) if Nx > 1 else Lx,
        Ly / (Ny - 1) if Ny > 1 else Ly,
        Lz / (Nz - 1) if Nz > 1 else Lz,
    )
    grid.point_data["T"] = T.ravel(order="C")
    return grid


def make_discrete_slice_mesh(T_slice, axis, position, hx, hy, hz):
    """
    T_slice: 2D массив.
      axis='z' → форма (Ny, Nx), срез в плоскости xy на z=position
      axis='y' → форма (Nz, Nx), срез в плоскости xz на y=position
      axis='x' → форма (Nz, Ny), срез в плоскости yz на x=position
    """
    if axis == "z":
        n_rows, n_cols = T_slice.shape
        u_spacing = hx
        v_spacing = hy
    elif axis == "y":
        n_rows, n_cols = T_slice.shape
        u_spacing = hx
        v_spacing = hz
    else:
        n_rows, n_cols = T_slice.shape
        u_spacing = hy
        v_spacing = hz

    us = np.arange(n_cols) * u_spacing
    vs = np.arange(n_rows) * v_spacing
    uu, vv = np.meshgrid(us, vs)

    if axis == "z":
        points = np.column_stack([uu.ravel(), vv.ravel(), np.full(uu.size, position)])
    elif axis == "y":
        points = np.column_stack([uu.ravel(), np.full(uu.size, position), vv.ravel()])
    else:
        points = np.column_stack([np.full(uu.size, position), uu.ravel(), vv.ravel()])

    cells = []
    for j in range(n_rows - 1):
        for i in range(n_cols - 1):
            p00 = j * n_cols + i
            p10 = j * n_cols + (i + 1)
            p11 = (j + 1) * n_cols + (i + 1)
            p01 = (j + 1) * n_cols + i
            cells.append([4, p00, p10, p11, p01])
    cells = np.array(cells, dtype=np.int64).ravel()

    mesh = pv.PolyData(points, cells)

    cell_vals = []
    for j in range(n_rows - 1):
        for i in range(n_cols - 1):
            cell_vals.append(T_slice[j, i])
    mesh.cell_data["T"] = np.array(cell_vals, dtype=float)

    return mesh


def extract_slice(field, axis, position):
    Nx, Ny, Nz = field["Nx"], field["Ny"], field["Nz"]
    Lx, Ly, Lz = field["Lx"], field["Ly"], field["Lz"]
    T = field["T"]

    hx = Lx / (Nx - 1) if Nx > 1 else Lx
    hy = Ly / (Ny - 1) if Ny > 1 else Ly
    hz = Lz / (Nz - 1) if Nz > 1 else Lz

    if axis == "z":
        k = int(round(position / hz))
        k = max(0, min(Nz - 1, k))
        return T[k, :, :], k * hz

    if axis == "y":
        j = int(round(position / hy))
        j = max(0, min(Ny - 1, j))
        return T[:, j, :], j * hy

    i = int(round(position / hx))
    i = max(0, min(Nx - 1, i))
    return T[:, :, i], i * hx


def show_interactive_slice(field, axis="z", screenshot=None):
    import vtk

    grid = build_grid(field)
    Nx, Ny, Nz = field["Nx"], field["Ny"], field["Nz"]
    Lx, Ly, Lz = field["Lx"], field["Ly"], field["Lz"]
    Tmin = float(field["T"].min())
    Tmax = float(field["T"].max())

    hx = Lx / (Nx - 1) if Nx > 1 else Lx
    hy = Ly / (Ny - 1) if Ny > 1 else Ly
    hz = Lz / (Nz - 1) if Nz > 1 else Lz

    state = {
        "axis": axis,
        "show_volume": True,
        "show_iso": False,
        "slice_actor": None,
        "volume_actor": None,
        "iso_actors": [],
        "slider_widget": None,
    }

    limits = {"x": Lx, "y": Ly, "z": Lz}

    pl = pv.Plotter(window_size=[1500, 1000])
    pl.set_background("black")

    def rebuild_slice(position):
        T_slice, snapped = extract_slice(field, state["axis"], position)
        mesh = make_discrete_slice_mesh(T_slice, state["axis"], snapped, hx, hy, hz)

        if state["slice_actor"] is None:
            state["slice_actor"] = pl.add_mesh(
                mesh,
                cmap="inferno",
                clim=[Tmin, Tmax],
                show_scalar_bar=False,
                lighting=False,
                show_edges=False,
            )
        else:
            state["slice_actor"].mapper.SetInputData(mesh)
            state["slice_actor"].mapper.SetScalarRange(Tmin, Tmax)
            state["slice_actor"].mapper.Update()
        pl.render()

    def add_axis_slider(axis_name):
        slider_widget = vtk.vtkSliderWidget()
        slider_widget.SetInteractor(pl.iren.interactor)
        slider_widget.SetAnimationModeToAnimate()

        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(0.0)
        rep.SetMaximumValue(float(limits[axis_name]))
        rep.SetValue(float(limits[axis_name]) * 0.5)
        rep.SetTitleText(f"Position along {axis_name}")

        rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint1Coordinate().SetValue(0.05, 0.06)
        rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint2Coordinate().SetValue(0.75, 0.06)

        rep.SetTubeWidth(0.005)
        rep.SetSliderLength(0.025)
        rep.SetSliderWidth(0.025)
        rep.SetTitleHeight(0.02)
        rep.SetLabelHeight(0.015)

        rep.GetSliderProperty().SetColor(1.0, 0.3, 0.3)
        rep.GetTitleProperty().SetColor(1.0, 1.0, 1.0)
        rep.GetLabelProperty().SetColor(1.0, 1.0, 1.0)
        rep.GetTubeProperty().SetColor(0.7, 0.7, 0.7)
        rep.GetCapProperty().SetColor(0.7, 0.7, 0.7)

        slider_widget.SetRepresentation(rep)

        def on_interaction(obj, event):
            val = obj.GetRepresentation().GetValue()
            rebuild_slice(val)

        slider_widget.AddObserver("InteractionEvent", on_interaction)
        slider_widget.AddObserver("EndInteractionEvent", on_interaction)

        slider_widget.EnabledOn()
        state["slider_widget"] = slider_widget

    def rebuild_axis(new_axis):
        state["axis"] = new_axis

        if state["slider_widget"] is not None:
            state["slider_widget"].EnabledOff()
            state["slider_widget"] = None

        add_axis_slider(new_axis)
        rebuild_slice(float(limits[new_axis]) * 0.5)

    def toggle_volume():
        state["show_volume"] = not state["show_volume"]
        if state["volume_actor"] is not None:
            state["volume_actor"].SetVisibility(state["show_volume"])
        pl.render()

    def toggle_iso():
        state["show_iso"] = not state["show_iso"]
        if not state["iso_actors"] and state["show_iso"]:
            iso_values = np.linspace(Tmin + 0.15 * (Tmax - Tmin),
                                     Tmax - 0.15 * (Tmax - Tmin), 5)
            for iso in iso_values:
                surf = grid.contour([iso], scalars="T")
                if surf.n_points > 0:
                    a = pl.add_mesh(surf, color="white", opacity=0.15,
                                    smooth_shading=True)
                    state["iso_actors"].append(a)
        for a in state["iso_actors"]:
            a.SetVisibility(state["show_iso"])
        pl.render()

    state["volume_actor"] = pl.add_volume(
        grid,
        cmap="inferno",
        opacity="sigmoid_5",
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "T",
            "n_labels": 6,
            "color": "white",
            "position_x": 0.85,
            "position_y": 0.15,
            "width": 0.05,
            "height": 0.7,
            "vertical": True,
        },
    )

    pl.add_axes(color="white")
    pl.add_bounding_box(color="gray")
    pl.camera_position = "iso"

    add_axis_slider(axis)
    rebuild_slice(float(limits[axis]) * 0.5)

    pl.add_key_event("x", lambda: rebuild_axis("x"))
    pl.add_key_event("y", lambda: rebuild_axis("y"))
    pl.add_key_event("z", lambda: rebuild_axis("z"))
    pl.add_key_event("v", toggle_volume)
    pl.add_key_event("i", toggle_iso)

    print("Controls:")
    print("  Slider     — move the slice (updates live)")
    print("  x / y / z  — switch slice axis")
    print("  v          — toggle volume rendering")
    print("  i          — toggle isosurfaces")
    print("  r          — reset camera")
    print("  q          — quit")

    if screenshot:
        pl.show(screenshot=screenshot, auto_close=True)
        print(f"saved {screenshot}")
    else:
        pl.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "field.bin"
    mode = sys.argv[2] if len(sys.argv) > 2 else "slice"
    out = sys.argv[3] if len(sys.argv) > 3 else None

    field = read_field_3d(path)
    print(f"Grid: {field['Nx']} x {field['Ny']} x {field['Nz']}")
    print(f"Domain: [{field['Lx']}, {field['Ly']}, {field['Lz']}]")
    print(f"T range: [{field['T'].min():.3f}, {field['T'].max():.3f}]")

    if mode == "slice":
        show_interactive_slice(field, axis="z", screenshot=out)
    else:
        print(f"unknown mode '{mode}'. use: slice")
        sys.exit(1)
