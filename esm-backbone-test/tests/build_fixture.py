"""
build_fixture.py — one-shot fixture builder for the MPNNScorer regression test.

Instantiates the SKEMPI test-split dataset (parsing PDBs) and dumps a single
sample (a complex with >=1 mutation) to tests/fixtures/one_complex.pkl, so the
regression test can run without re-parsing the full dataset every time.

Run from the worktree root (so the data/ relative paths resolve), e.g.:
    cd <worktree-root> && python esm-backbone-test/tests/build_fixture.py
"""

import os
import sys
import pickle

# Resolve paths relative to the worktree root (parent of esm-backbone-test/).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))            # .../tests
_WORKTREE_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_FIXTURE_PATH = os.path.join(_THIS_DIR, "fixtures", "one_complex.pkl")

# Allow running as a plain script (outside pytest/conftest): make `stabddg`
# importable from the worktree root.
if _WORKTREE_ROOT not in sys.path:
    sys.path.insert(0, _WORKTREE_ROOT)

from stabddg.ppi_dataset import SKEMPIDataset


def build():
    csv_path = os.path.join(_WORKTREE_ROOT, "data", "SKEMPI", "filtered_skempi.csv")
    split_path = os.path.join(_WORKTREE_ROOT, "data", "SKEMPI", "test_pdb.pkl")
    pdb_dir = os.path.join(_WORKTREE_ROOT, "data", "SKEMPI2_PDBs")
    # Cache the parsed structure dict next to the fixture to speed re-runs.
    cache_path = os.path.join(_THIS_DIR, "fixtures", "skempi_test_pdb_dict.pkl")

    ds = SKEMPIDataset(
        csv_path=csv_path,
        split_path=split_path,
        pdb_dir=pdb_dir,
        pdb_dict_cache_path=cache_path,
        af_apo_structures=False,
    )

    # Pick the first sample that has at least one mutation.
    sample = None
    for i in range(len(ds)):
        cand = ds[i]
        if len(cand["mutation_list"]) >= 1 and cand["complex_mut_seqs"].shape[0] >= 1:
            sample = cand
            break
    assert sample is not None, "No SKEMPI sample with >=1 mutation found."

    os.makedirs(os.path.dirname(_FIXTURE_PATH), exist_ok=True)
    with open(_FIXTURE_PATH, "wb") as f:
        pickle.dump(sample, f)

    print(f"Wrote fixture for complex '{sample['name']}' "
          f"({sample['complex_mut_seqs'].shape[0]} mutant seq(s), "
          f"{len(sample['mutation_list'])} mutation entr(y/ies)) to {_FIXTURE_PATH}")


if __name__ == "__main__":
    build()
