"""Test cases for rewrite_columns."""

import tempfile
from pathlib import Path

import lance
import lance_ray as lr
import pyarrow as pa
import pyarrow.compute as pc
import pytest


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def _write(path, ids, vals, *, max_rows_per_file=2, **kwargs):
    table = pa.table(
        {
            "id": pa.array(ids, pa.int64()),
            "v": pa.array(vals, pa.float64()),
        }
    )
    return lance.write_dataset(
        table, str(path), max_rows_per_file=max_rows_per_file, **kwargs
    )


class TestRewriteColumns:
    def test_full_rewrite_single_column(self, temp_dir):
        path = Path(temp_dir) / "full.lance"
        _write(path, list(range(6)), [float(i) for i in range(6)])
        ds_before = lance.dataset(str(path))
        frag_ids_before = sorted(f.metadata.id for f in ds_before.get_fragments())
        version_before = ds_before.version

        def double(tbl):
            return {"v": pc.multiply(tbl.column("v"), 2)}

        lr.rewrite_columns(str(path), columns=["v"], transform=double)

        ds = lance.dataset(str(path))
        result = ds.to_table().to_pandas().sort_values("id").reset_index(drop=True)
        assert result["v"].tolist() == [float(i) * 2 for i in range(6)]
        # id column (not rewritten) unchanged, row count unchanged.
        assert result["id"].tolist() == list(range(6))
        # Fragment ids preserved, new version created.
        assert sorted(f.metadata.id for f in ds.get_fragments()) == frag_ids_before
        assert ds.version == version_before + 1

    def test_rewrite_multiple_columns(self, temp_dir):
        path = Path(temp_dir) / "multi.lance"
        table = pa.table(
            {
                "id": pa.array(range(5), pa.int64()),
                "a": pa.array([float(i) for i in range(5)], pa.float64()),
                "b": pa.array([i * 10 for i in range(5)], pa.int64()),
            }
        )
        lance.write_dataset(table, str(path), max_rows_per_file=2)

        def bump(tbl):
            return {
                "a": pc.add(tbl.column("a"), 1.0),
                "b": pc.add(tbl.column("b"), 1),
            }

        lr.rewrite_columns(str(path), columns=["a", "b"], transform=bump)

        result = (
            lance.dataset(str(path))
            .to_table()
            .to_pandas()
            .sort_values("id")
            .reset_index(drop=True)
        )
        assert result["a"].tolist() == [float(i) + 1 for i in range(5)]
        assert result["b"].tolist() == [i * 10 + 1 for i in range(5)]

    def test_row_level_filter_partial_fragments(self, temp_dir):
        # 3 fragments x 5 rows; filter hits a subset of rows in each.
        path = Path(temp_dir) / "filter.lance"
        _write(
            path,
            list(range(15)),
            [float(i) for i in range(15)],
            max_rows_per_file=5,
        )

        def set999(tbl):
            return {"v": pa.array([999.0] * tbl.num_rows, pa.float64())}

        lr.rewrite_columns(
            str(path), columns=["v"], transform=set999, filter="id % 2 == 0"
        )

        result = (
            lance.dataset(str(path))
            .to_table()
            .to_pandas()
            .sort_values("id")
            .reset_index(drop=True)
        )
        # Even ids rewritten; odd ids keep original values.
        expected = [999.0 if i % 2 == 0 else float(i) for i in range(15)]
        assert result["v"].tolist() == expected

    def test_filter_untouched_fragment_stays_identical(self, temp_dir):
        # Filter only hits fragment 0; fragment 1 must be byte-for-byte untouched.
        path = Path(temp_dir) / "untouched.lance"
        _write(path, list(range(4)), [float(i) for i in range(4)], max_rows_per_file=2)
        frag1_files_before = [
            df.path for df in lance.dataset(str(path)).get_fragment(1).data_files()
        ]

        def negate(tbl):
            return {"v": pc.multiply(tbl.column("v"), -1.0)}

        lr.rewrite_columns(str(path), columns=["v"], transform=negate, filter="id < 2")

        ds = lance.dataset(str(path))
        result = ds.to_table().to_pandas().sort_values("id").reset_index(drop=True)
        assert result["v"].tolist() == [0.0, -1.0, 2.0, 3.0]
        # Fragment 1's data files were not rewritten.
        frag1_files_after = [df.path for df in ds.get_fragment(1).data_files()]
        assert frag1_files_after == frag1_files_before

    def test_deletion_vector_preserved(self, temp_dir):
        path = Path(temp_dir) / "deleted.lance"
        _write(path, list(range(6)), [float(i) for i in range(6)], max_rows_per_file=6)
        ds = lance.dataset(str(path))
        ds.delete("id = 2")

        def double(tbl):
            return {"v": pc.multiply(tbl.column("v"), 2)}

        lr.rewrite_columns(str(path), columns=["v"], transform=double)

        result = (
            lance.dataset(str(path))
            .to_table()
            .to_pandas()
            .sort_values("id")
            .reset_index(drop=True)
        )
        # id=2 stays deleted; all remaining live rows rewritten.
        assert result["id"].tolist() == [0, 1, 3, 4, 5]
        assert result["v"].tolist() == [0.0, 2.0, 6.0, 8.0, 10.0]

    def test_empty_match_is_noop(self, temp_dir):
        path = Path(temp_dir) / "noop.lance"
        _write(path, list(range(4)), [float(i) for i in range(4)])
        version_before = lance.dataset(str(path)).version

        def double(tbl):
            return {"v": pc.multiply(tbl.column("v"), 2)}

        lr.rewrite_columns(
            str(path), columns=["v"], transform=double, filter="id > 1000"
        )

        ds = lance.dataset(str(path))
        assert ds.version == version_before
        assert ds.to_table()["v"].to_pylist() == [0.0, 1.0, 2.0, 3.0]

    def test_stable_row_ids_supported(self, temp_dir):
        path = Path(temp_dir) / "stable.lance"
        _write(
            path,
            list(range(6)),
            [float(i) for i in range(6)],
            max_rows_per_file=3,
            enable_stable_row_ids=True,
        )
        assert lance.dataset(str(path)).has_stable_row_ids

        def double(tbl):
            return {"v": pc.multiply(tbl.column("v"), 2)}

        lr.rewrite_columns(str(path), columns=["v"], transform=double)

        ds = lance.dataset(str(path))
        before = ds.to_table(columns=["id"], with_row_id=True)
        assert ds.to_table()["v"].to_pylist() == [float(i) * 2 for i in range(6)]
        # Row ids unchanged by a column rewrite.
        assert before.column("_rowid").to_pylist() == list(range(6))

    def test_indexed_column_query_correct_after_rewrite(self, temp_dir):
        path = Path(temp_dir) / "indexed.lance"
        _write(
            path,
            list(range(10)),
            [float(i) for i in range(10)],
            max_rows_per_file=5,
        )
        ds = lance.dataset(str(path))
        ds.create_scalar_index("v", index_type="BTREE")

        def offset(tbl):
            return {"v": pc.add(tbl.column("v"), 100.0)}

        lr.rewrite_columns(str(path), columns=["v"], transform=offset)

        ds = lance.dataset(str(path))
        # A query on the rewritten indexed column must return the new values.
        hit = ds.to_table(filter="v = 103.0")
        assert hit.num_rows == 1
        assert hit.column("id")[0].as_py() == 3
        # Old value no longer present.
        assert ds.to_table(filter="v = 3.0").num_rows == 0

    def test_directory_namespace_end_to_end(self, temp_dir):
        import ray

        import pandas as pd

        table_id = ["rewrite_ns"]
        data = pd.DataFrame({"id": [1, 2, 3, 4], "v": [1.0, 2.0, 3.0, 4.0]})
        lr.write_lance(
            ray.data.from_pandas(data),
            namespace_impl="dir",
            namespace_properties={"root": temp_dir},
            table_id=table_id,
            min_rows_per_file=1,
            max_rows_per_file=2,
        )

        def double(tbl):
            return {"v": pc.multiply(tbl.column("v"), 2)}

        lr.rewrite_columns(
            columns=["v"],
            transform=double,
            namespace_impl="dir",
            namespace_properties={"root": temp_dir},
            table_id=table_id,
        )

        result = lr.read_lance(
            namespace_impl="dir",
            namespace_properties={"root": temp_dir},
            table_id=table_id,
        )
        df = result.to_pandas().sort_values("id").reset_index(drop=True)
        assert df["v"].tolist() == [2.0, 4.0, 6.0, 8.0]


class TestRewriteColumnsValidation:
    def _base(self, temp_dir, name="val.lance"):
        path = Path(temp_dir) / name
        _write(path, list(range(4)), [float(i) for i in range(4)])
        return path

    def test_nonexistent_column_rejected(self, temp_dir):
        path = self._base(temp_dir)
        with pytest.raises(ValueError, match="does not exist"):
            lr.rewrite_columns(
                str(path), columns=["missing"], transform=lambda tbl: {"missing": []}
            )

    def test_metadata_column_rejected(self, temp_dir):
        path = self._base(temp_dir)
        with pytest.raises(ValueError, match="metadata column"):
            lr.rewrite_columns(
                str(path), columns=["_rowaddr"], transform=lambda tbl: {}
            )

    def test_empty_columns_rejected(self, temp_dir):
        path = self._base(temp_dir)
        with pytest.raises(ValueError, match="non-empty"):
            lr.rewrite_columns(str(path), columns=[], transform=lambda tbl: {})

    def test_wrong_row_count_rejected(self, temp_dir):
        path = self._base(temp_dir, "rows.lance")
        version_before = lance.dataset(str(path)).version

        def drop_rows(tbl):
            return {"v": pa.array([1.0], pa.float64())}

        with pytest.raises(ValueError, match="same number of rows"):
            lr.rewrite_columns(str(path), columns=["v"], transform=drop_rows)
        # Dataset unchanged on failure.
        assert lance.dataset(str(path)).version == version_before

    def test_missing_column_in_output_rejected(self, temp_dir):
        path = self._base(temp_dir, "miss.lance")
        with pytest.raises(ValueError, match="exactly the rewritten"):
            lr.rewrite_columns(
                str(path),
                columns=["v"],
                transform=lambda tbl: {"other": [0.0] * tbl.num_rows},
            )

    def test_extra_column_in_output_rejected(self, temp_dir):
        path = self._base(temp_dir, "extra.lance")

        def add_extra(tbl):
            return {
                "v": pc.multiply(tbl.column("v"), 2),
                "unexpected": [0.0] * tbl.num_rows,
            }

        with pytest.raises(ValueError, match="unexpected"):
            lr.rewrite_columns(str(path), columns=["v"], transform=add_extra)

    def test_wrong_type_rejected(self, temp_dir):
        path = self._base(temp_dir, "type.lance")

        def wrong_type(tbl):
            # v is float64; return int32.
            return {"v": pa.array([1] * tbl.num_rows, pa.int32())}

        with pytest.raises(ValueError, match="type"):
            lr.rewrite_columns(str(path), columns=["v"], transform=wrong_type)

    def test_non_table_result_rejected(self, temp_dir):
        path = self._base(temp_dir, "bad.lance")
        with pytest.raises((TypeError, ValueError)):
            lr.rewrite_columns(
                str(path), columns=["v"], transform=lambda tbl: "not a table"
            )
