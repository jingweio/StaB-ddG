"""
test_eval_format.py — pure-format test for run.eval.write_csv.

Asserts that ``write_csv`` reproduces the EXACT original ``skempi_eval.py`` CSV
output: a header line ``,#Pdb,Mutation,ddG,ddG_pred`` (leading *unnamed* pandas
index column), the prediction stored RAW (no sign flip) under ``ddG_pred``, and
columns in the order ``#Pdb, Mutation, ddG, ddG_pred``.

No model is involved — this is a fast format check that protects
``reproduce/compute_metrics.py`` and the mutation-analysis tools that consume
the CSV.
"""

import pandas as pd

from run.eval import write_csv

# Header line of the author/ibex baseline CSV
# (ibex_records/stab-ddg_repro/eval_skempi_ft_47437925.csv).
BASELINE_HEADER = ",#Pdb,Mutation,ddG,ddG_pred"


def _toy_pred_df():
    return pd.DataFrame({
        "#Pdb": ["1A4Y_A_B", "1A4Y_A_B", "2XYZ_C_D"],
        "Mutation": ["['HB8A']", "['QB12A']", "['KC5A']"],
        "ddG": [-0.9036602, -0.3001140, 1.234],
        "Prediction": [-0.078729, -0.157874, 0.456],
    })


def test_header_matches_baseline(tmp_path):
    out = tmp_path / "out.csv"
    write_csv(_toy_pred_df(), str(out))

    with open(out) as f:
        header = f.readline().rstrip("\n")
    assert header == BASELINE_HEADER, repr(header)


def test_reread_has_unnamed_index_column(tmp_path):
    out = tmp_path / "out.csv"
    write_csv(_toy_pred_df(), str(out))

    # Re-read exactly as compute_metrics.py does (pd.read_csv, no index_col).
    df = pd.read_csv(out)
    assert list(df.columns) == ["Unnamed: 0", "#Pdb", "Mutation", "ddG", "ddG_pred"]
    # The unnamed column is the integer row index 0..N-1.
    assert df["Unnamed: 0"].tolist() == [0, 1, 2]


def test_prediction_stored_raw_no_sign_flip(tmp_path):
    out = tmp_path / "out.csv"
    src = _toy_pred_df()
    write_csv(src, str(out))

    df = pd.read_csv(out)
    # ddG_pred must equal Prediction verbatim (no x-1), matching skempi_eval.py.
    assert df["ddG_pred"].tolist() == src["Prediction"].tolist()
    # ddG column passed through unchanged.
    assert df["ddG"].tolist() == src["ddG"].tolist()


def test_returned_df_columns(tmp_path):
    out = tmp_path / "out.csv"
    combined = write_csv(_toy_pred_df(), str(out))
    assert list(combined.columns) == ["#Pdb", "Mutation", "ddG", "ddG_pred"]
