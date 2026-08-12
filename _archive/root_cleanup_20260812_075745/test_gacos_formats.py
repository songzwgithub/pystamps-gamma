#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from pystamps.pipeline.gacos_correction import GacosProduct, discover_products, sample_product


def main() -> int:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        grid = np.arange(12, dtype="<f4").reshape(3, 4)

        ztd = root / "20200101.ztd"
        grid.tofile(ztd)
        rsc = root / "20200101.ztd.rsc"
        rsc.write_text(
            "WIDTH 4\n"
            "FILE_LENGTH 3\n"
            "X_FIRST 10\n"
            "Y_FIRST 50\n"
            "X_STEP 1\n"
            "Y_STEP -1\n",
            encoding="utf-8",
        )
        ztd_product = GacosProduct("20200101", ztd, "ztd", rsc)
        ztd_values, _ = sample_product(
            ztd_product,
            np.asarray([10.0, 11.5, 13.0]),
            np.asarray([50.0, 49.0, 48.0]),
            "m",
        )
        np.testing.assert_allclose(ztd_values, [0.0, 5.5, 11.0])
        print("ZTD sampling: PASSED")

        try:
            import rasterio
            from rasterio.transform import from_origin
        except Exception:
            print("GeoTIFF sampling: SKIPPED (rasterio unavailable)")
        else:
            tif = root / "20200102.ztd.tif"
            with rasterio.open(
                tif,
                "w",
                driver="GTiff",
                height=3,
                width=4,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(9.5, 50.5, 1.0, 1.0),
                nodata=-9999.0,
            ) as dataset:
                dataset.write(grid.astype(np.float32), 1)
                dataset.update_tags(unit="m")
            tif_product = GacosProduct("20200102", tif, "tif")
            tif_values, _ = sample_product(
                tif_product,
                np.asarray([10.0, 11.5, 13.0]),
                np.asarray([50.0, 49.0, 48.0]),
                "auto",
            )
            np.testing.assert_allclose(tif_values, [0.0, 5.5, 11.0])
            print("GeoTIFF sampling: PASSED")

        products = discover_products(root, "auto")
        assert set(products) == {"20200101", "20200102"}
        assert products["20200101"].kind == "ztd"
        assert products["20200102"].kind == "tif"
        print("Product discovery: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
