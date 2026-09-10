import os

import pandas as pd
from sklearn.cluster import DBSCAN

from filter import DATA_DIR, filter_noise, remove_background, resolve_path, show_pointcloud

EPS = 1
MIN_SAMPLES = 10


def load_filtered_frame() -> tuple[pd.DataFrame, str]:
    """
    Carga el frame indicado en POINTCLOUD_FILE y le aplica el mismo
    filtrado de fondo/ruido que filter.py (usando carretera.csv de referencia).
    """

    filename = os.getenv("POINTCLOUD_FILE")

    if not filename:
        raise ValueError("No está definida POINTCLOUD_FILE en el fichero .env")

    path = resolve_path(filename)

    if path.name == "carretera.csv":
        raise ValueError("carretera.csv es el fondo de referencia, no contiene vehículos")

    data = pd.read_csv(path)
    background = pd.read_csv(DATA_DIR / "carretera.csv")

    data = remove_background(data=data, background=background)
    data = filter_noise(data)

    return data, path.name


def find_vehicles(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa los puntos en vehículos con DBSCAN.

    Cada cluster encontrado es un vehículo (un centroide). Los puntos
    que DBSCAN marca como ruido (label == -1, es decir, demasiado
    alejados de cualquier centroide) se descartan directamente.
    """

    coords = data[["x", "y", "z"]].to_numpy()
    labels = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit_predict(coords)

    data = data.copy()
    data["cluster"] = labels

    return data[data["cluster"] != -1].copy()


def main() -> None:
    data, title = load_filtered_frame()

    vehicles = find_vehicles(data)

    show_pointcloud(vehicles, title=title, color_field="cluster")


if __name__ == "__main__":
    main()
