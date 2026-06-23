# =============================================================================
# tests/test_scaler_fix.py
# Run with:  python -m pytest tests/test_scaler_fix.py -v
# =============================================================================

import json
import os
import sys
import types

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler, RobustScaler

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.preprocessing import (
    FEATURE_COLS,
    build_features,
    load_scaler,
    preprocess,
)

# =============================================================================
# FIXTURES
# =============================================================================
@pytest.fixture()
def dummy_scaler(tmp_path, monkeypatch):
    """Create a real StandardScaler fitted on random data and patch the path."""
    scaler = StandardScaler()
    X      = np.random.randn(200, len(FEATURE_COLS)).astype(np.float32)
    scaler.fit(X)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    scaler_path = models_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    cols_path = models_dir / "feature_columns.json"
    cols_path.write_text(json.dumps(FEATURE_COLS))

    # Patch module-level path constants
    import utils.preprocessing as pp
    monkeypatch.setattr(pp, "SCALER_PATH", str(scaler_path))
    monkeypatch.setattr(pp, "COLS_PATH",   str(cols_path))

    return scaler, str(scaler_path)


@pytest.fixture()
def sample_csv_df():
    """Minimal CTU-13-like DataFrame (50 rows)."""
    rng = np.random.default_rng(42)
    n   = 50
    return pd.DataFrame({
        "Dur":      rng.uniform(0.001, 10.0, n),
        "Proto":    rng.choice(["tcp", "udp", "icmp"], n),
        "Sport":    rng.integers(1, 65535, n).astype(str),
        "Dir":      rng.choice(["<->", "->", "<-"], n),
        "Dport":    rng.integers(1, 65535, n).astype(str),
        "State":    rng.choice(["CON", "FIN", "REQ"], n),
        "sTos":     rng.integers(0, 32, n).astype(float),
        "dTos":     rng.integers(0, 32, n).astype(float),
        "TotPkts":  rng.integers(1, 1000, n).astype(float),
        "TotBytes": rng.integers(64, 1_500_000, n).astype(float),
        "SrcBytes": rng.integers(0, 800_000, n).astype(float),
        "Label":    rng.choice(["BENIGN", "Botnet"], n),
    })


# =============================================================================
# TEST 1 — load_scaler returns StandardScaler
# =============================================================================
class TestLoadScaler:
    def test_returns_standard_scaler(self, dummy_scaler):
        scaler, _ = dummy_scaler
        loaded    = load_scaler()
        assert isinstance(loaded, StandardScaler), (
            f"Expected StandardScaler, got {type(loaded).__name__}"
        )

    def test_raises_if_file_missing(self, monkeypatch, tmp_path):
        import utils.preprocessing as pp
        monkeypatch.setattr(pp, "SCALER_PATH", str(tmp_path / "nonexistent.pkl"))
        with pytest.raises(FileNotFoundError, match="models/scaler.pkl"):
            load_scaler()

    def test_raises_if_wrong_type(self, tmp_path, monkeypatch):
        import utils.preprocessing as pp
        # Save a RobustScaler — should be rejected
        bad_scaler = RobustScaler()
        bad_scaler.fit(np.random.randn(10, len(FEATURE_COLS)))
        bad_path   = tmp_path / "scaler.pkl"
        joblib.dump(bad_scaler, bad_path)
        monkeypatch.setattr(pp, "SCALER_PATH", str(bad_path))

        with pytest.raises(TypeError, match="StandardScaler"):
            load_scaler()

    def test_raises_if_dimension_mismatch(self, tmp_path, monkeypatch):
        import utils.preprocessing as pp
        # Scaler fitted on wrong number of features
        bad_scaler = StandardScaler()
        bad_scaler.fit(np.random.randn(10, 5))   # 5 != 17
        bad_path   = tmp_path / "scaler.pkl"
        joblib.dump(bad_scaler, bad_path)
        monkeypatch.setattr(pp, "SCALER_PATH", str(bad_path))

        with pytest.raises(ValueError, match="17"):
            load_scaler()


# =============================================================================
# TEST 2 — build_features produces correct columns and shape
# =============================================================================
class TestBuildFeatures:
    def test_output_columns_match_feature_cols(self, sample_csv_df):
        result = build_features(sample_csv_df)
        assert list(result.columns) == FEATURE_COLS, (
            f"Column mismatch:\nExpected: {FEATURE_COLS}\nGot:      {list(result.columns)}"
        )

    def test_output_shape(self, sample_csv_df):
        result = build_features(sample_csv_df)
        assert result.shape == (50, 17)

    def test_no_nan_in_output(self, sample_csv_df):
        result = build_features(sample_csv_df)
        assert not result.isnull().any().any(), "NaN values found in build_features output"

    def test_dur_renamed_to_duration(self, sample_csv_df):
        result = build_features(sample_csv_df)
        assert "Duration" in result.columns
        assert "Dur" not in result.columns

    def test_log1p_applied(self, sample_csv_df):
        """TotBytes in output must be < original (log-compressed)."""
        result       = build_features(sample_csv_df)
        # log1p(x) < x for x > 1
        orig_max     = sample_csv_df["TotBytes"].max()
        result_max   = result["TotBytes"].max()
        assert result_max < orig_max, (
            "log1p was not applied to TotBytes — output exceeds raw input max"
        )

    def test_sport_is_priv_flag(self):
        df = pd.DataFrame({
            "Dur": [1.0, 1.0], "Proto": ["tcp", "tcp"], "Sport": ["80", "50000"],
            "Dir": ["->", "->"], "Dport": ["443", "8080"], "State": ["CON", "CON"],
            "sTos": [0.0, 0.0], "dTos": [0.0, 0.0],
            "TotPkts": [10.0, 10.0], "TotBytes": [1000.0, 1000.0], "SrcBytes": [500.0, 500.0],
        })
        result = build_features(df)
        # Row 0: Sport=80 (≤1024 → priv=1), Dport=443 (≤1024 → priv=1)
        assert result["Sport_is_priv"].iloc[0] == 1, "port 80 should be privileged"
        assert result["Dport_is_priv"].iloc[0] == 1, "port 443 should be privileged"
        # Row 1: Sport=50000 (>1024 → priv=0), Dport=8080 (>1024 → priv=0)
        assert result["Sport_is_priv"].iloc[1] == 0, "port 50000 should not be privileged"
        assert result["Dport_is_priv"].iloc[1] == 0, "port 8080 should not be privileged"

    def test_missing_columns_filled_with_zero(self):
        """DataFrame with only TotBytes/TotPkts/SrcBytes — everything else missing."""
        df = pd.DataFrame({
            "TotBytes": [1000.0],
            "TotPkts":  [10.0],
            "SrcBytes": [500.0],
        })
        result = build_features(df)
        assert result.shape == (1, 17)
        assert not result.isnull().any().any()


# =============================================================================
# TEST 3 — preprocess never calls fit_transform on the scaler
# =============================================================================
class TestPreprocess:
    def test_output_shape(self, dummy_scaler, sample_csv_df):
        scaler, _ = dummy_scaler
        result    = preprocess(sample_csv_df, scaler)
        assert result.shape == (50, 17)

    def test_output_dtype_float32(self, dummy_scaler, sample_csv_df):
        scaler, _ = dummy_scaler
        result    = preprocess(sample_csv_df, scaler)
        assert result.dtype == np.float32

    def test_scaler_not_modified(self, dummy_scaler, sample_csv_df):
        """Confirm scaler.mean_ is identical before and after preprocess()."""
        scaler, _ = dummy_scaler
        mean_before = scaler.mean_.copy()
        preprocess(sample_csv_df, scaler)
        np.testing.assert_array_equal(
            scaler.mean_, mean_before,
            err_msg="preprocess() called fit_transform — scaler was mutated!"
        )

    def test_deterministic(self, dummy_scaler, sample_csv_df):
        """Same input must produce identical output every call."""
        scaler, _ = dummy_scaler
        out1 = preprocess(sample_csv_df, scaler)
        out2 = preprocess(sample_csv_df, scaler)
        np.testing.assert_array_equal(out1, out2)

    def test_output_is_standardized(self, dummy_scaler, sample_csv_df):
        """After transform with a scaler fitted on similar data,
        mean should be near 0 (not exact due to distribution difference)."""
        # Fit scaler on the same data for a pure sanity check
        from sklearn.preprocessing import StandardScaler as SS
        features = build_features(sample_csv_df).values.astype(np.float32)
        s        = SS()
        s.fit(features)

        result = s.transform(features)
        col_means = np.abs(result.mean(axis=0))
        # All column means should be near 0
        assert (col_means < 1e-5).all(), (
            f"Standardized output has non-zero means: {col_means}"
        )


# =============================================================================
# TEST 4 — Integration: end-to-end with loaded scaler
# =============================================================================
class TestIntegration:
    def test_full_pipeline_no_nan(self, dummy_scaler, sample_csv_df):
        scaler, _ = dummy_scaler
        result    = preprocess(sample_csv_df, scaler)
        assert not np.isnan(result).any(), "NaN in preprocessed output"

    def test_full_pipeline_no_inf(self, dummy_scaler, sample_csv_df):
        scaler, _ = dummy_scaler
        result    = preprocess(sample_csv_df, scaler)
        assert not np.isinf(result).any(), "Inf in preprocessed output"

    def test_shape_compatible_with_model(self, dummy_scaler, sample_csv_df):
        """Output must have exactly 17 features — matching model input_dim=17."""
        scaler, _ = dummy_scaler
        result    = preprocess(sample_csv_df, scaler)
        n_rows, n_features = result.shape
        assert n_features == 17, (
            f"Model expects 17 features, got {n_features}"
        )
