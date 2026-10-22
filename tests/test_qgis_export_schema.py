import sqlite3

from pystamps.qgis_export import _create_schema


def test_geopackage_rtree_schema():
    conn = sqlite3.connect(":memory:")

    try:
        rtree = _create_schema(
            conn,
            "insar_ps",
            [
                ("PS_ID", "INTEGER"),
                ("LOS_20240101", "REAL"),
            ],
            (120.0, 28.0, 123.0, 32.0),
        )

        assert rtree == "rtree_insar_ps_geom"

        row = conn.execute(
            "SELECT extension_name, definition "
            "FROM gpkg_extensions"
        ).fetchone()

        assert row == (
            "gpkg_rtree_index",
            "http://www.geopackage.org/spec/#extension_rtree",
        )

        table = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE name='rtree_insar_ps_geom'"
        ).fetchone()

        assert table is not None

    finally:
        conn.close()
