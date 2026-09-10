import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import open3d as o3d
import pandas as pd
from dotenv import load_dotenv


load_dotenv()

DATA_DIR = Path(__file__).parent / "data"


def resolve_path(name: str) -> Path:
    """
    Busca un fichero:
    - tal cual
    - dentro de data/
    - dentro de data/ añadiendo .csv
    """

    for candidate in (
        Path(name),
        DATA_DIR / name,
        DATA_DIR / f"{name}.csv",
    ):
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No existe '{name}' en {DATA_DIR}"
    )


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina puntos inválidos:
    - NaN
    - infinitos
    - range <= 0
    """

    data = data.copy()

    mask = (
        np.isfinite(data["x"])
        & np.isfinite(data["y"])
        & np.isfinite(data["z"])
    )

    if "range" in data.columns:
        mask &= (
            np.isfinite(data["range"])
            & (data["range"] > 0)
        )

    return data.loc[mask].copy()


def remove_background(
    data: pd.DataFrame,
    background: pd.DataFrame,
    threshold: float = 300
) -> pd.DataFrame:
    """
    Elimina el fondo usando carretera.csv como referencia.

    IMPORTANTE:
    Esta función debe ejecutarse antes de limpiar/eliminar filas,
    porque necesita mantener la correspondencia punto a punto
    entre ambos frames.

    Si un punto del frame actual está significativamente más cerca
    que el mismo rayo en carretera.csv, se considera un objeto nuevo.
    """

    if len(data) != len(background):
        raise ValueError(
            "El frame actual y carretera.csv deben tener "
            "el mismo número de puntos."
        )

    current_range = data["range"].to_numpy(dtype=float)
    background_range = background["range"].to_numpy(dtype=float)

    difference = background_range - current_range

    mask = (
        np.isfinite(current_range)
        & np.isfinite(background_range)
        & (current_range > 0)
        & (background_range > 0)
        & (difference > threshold)
    )

    return data.loc[mask].copy()


def filter_noise(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filtrado adicional después de eliminar el fondo.

    Se eliminan:
    - puntos inválidos
    - puntos demasiado lejanos
    - puntos fuera de una zona razonable de trabajo
    """

    data = clean_data(data)

    # Limitar puntos demasiado lejanos
    if "range" in data.columns:
        data = data[
            data["range"] < 40000
        ]

    # Región de interés amplia
    data = data[
        (data["x"] > -30)
        & (data["x"] < 30)
        & (data["y"] > -10)
        & (data["y"] < 15)
        & (data["z"] > -10)
        & (data["z"] < 10)
    ]

    return data.copy()


def colorize(
    data: pd.DataFrame,
    field: str
) -> np.ndarray:
    """
    Colorea los puntos según una columna.

    Ejemplos:
    - z
    - intensity
    - range
    - reflectivity
    """

    if field not in data.columns:
        raise ValueError(
            f"No existe el campo '{field}'"
        )

    values = data[field].to_numpy(dtype=float)

    minimum = values.min()
    maximum = values.max()

    if maximum == minimum:
        normalized = np.zeros_like(values)
    else:
        normalized = (
            values - minimum
        ) / (
            maximum - minimum
        )

    colors = mpl.colormaps["viridis"](
        normalized
    )[:, :3]

    return colors


def create_pointcloud(
    data: pd.DataFrame,
    color_field: str | None = "z"
) -> o3d.geometry.PointCloud:
    """
    Convierte un DataFrame en una nube de puntos Open3D.
    """

    pcd = o3d.geometry.PointCloud()

    points = data[
        ["x", "y", "z"]
    ].to_numpy()

    pcd.points = o3d.utility.Vector3dVector(
        points
    )

    if (
        color_field is not None
        and color_field in data.columns
    ):
        colors = colorize(
            data,
            color_field
        )

        pcd.colors = o3d.utility.Vector3dVector(
            colors
        )

    return pcd


def show_pointcloud(
    data: pd.DataFrame,
    title: str,
    color_field: str | None = "z"
) -> None:
    """
    Muestra la nube de puntos con Open3D.
    """

    if len(data) == 0:
        print("No hay puntos que mostrar.")
        return

    pcd = create_pointcloud(
        data,
        color_field=color_field
    )

    axes = (
        o3d.geometry
        .TriangleMesh
        .create_coordinate_frame(
            size=1.0,
            origin=[0, 0, 0]
        )
    )

    o3d.visualization.draw_geometries(
        [
            pcd,
            axes
        ],
        window_name=title,
        width=1200,
        height=800
    )


def main() -> None:
    filename = os.getenv(
        "POINTCLOUD_FILE"
    )

    if not filename:
        raise ValueError(
            "No está definida POINTCLOUD_FILE "
            "en el fichero .env"
        )

    # -----------------------------------------
    # Cargar frame actual COMPLETO
    # -----------------------------------------

    path = resolve_path(
        filename
    )

    data = pd.read_csv(
        path
    )

    print()
    print(f"Archivo: {path.name}")
    print(
        f"Puntos originales: {len(data)}"
    )

    # -----------------------------------------
    # Si es carretera.csv
    # -----------------------------------------

    if path.name == "carretera.csv":
        print(
            "carretera.csv es el frame "
            "de referencia."
        )

        data = clean_data(
            data
        )

        print(
            f"Puntos válidos: {len(data)}"
        )

    # -----------------------------------------
    # Si es otro frame, eliminar fondo PRIMERO
    # -----------------------------------------

    else:
        background_path = (
            DATA_DIR / "carretera.csv"
        )

        background = pd.read_csv(
            background_path
        )

        print(
            f"Usando como fondo: "
            f"{background_path.name}"
        )

        print(
            f"Puntos del frame actual: "
            f"{len(data)}"
        )

        print(
            f"Puntos del fondo: "
            f"{len(background)}"
        )

        # MUY IMPORTANTE:
        # quitar fondo antes de clean_data()
        data = remove_background(
            data=data,
            background=background,
            threshold=300
        )

        print(
            f"Puntos después de eliminar "
            f"el fondo: {len(data)}"
        )

        # Ahora sí limpiamos
        data = filter_noise(
            data
        )

        print(
            f"Puntos después del filtrado: "
            f"{len(data)}"
        )

    # -----------------------------------------
    # Mostrar nube resultante
    # -----------------------------------------

    show_pointcloud(
        data=data,
        title=path.name,
        color_field="z"
    )


if __name__ == "__main__":
    main()