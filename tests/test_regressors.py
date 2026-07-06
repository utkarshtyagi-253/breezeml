from breezeml import datasets, regressors


def test_linear_regressor():
    df = datasets.diabetes()
    model, report = regressors.linear(df, "target")
    assert hasattr(model, "predict")
    assert "r2" in report
    assert "mae" in report
    assert "rmse" in report
    assert "adjusted_r2" in report
    assert "mape" in report


def test_compare_regressors():
    df = datasets.diabetes()
    results = regressors.compare(df, "target", show=False)
    assert len(results) > 0
    assert "regressor" in results[0]
    assert "r2" in results[0]


def test_regressor_detailed_report():
    df = datasets.diabetes()
    info = regressors.detailed_report(df, "target", algo="decision_tree")
    assert "explained_variance" in info
    assert "residuals" in info
    assert "prediction_vs_actual" in info


def test_regressor_quick_tune():
    df = datasets.diabetes()
    model, params, report = regressors.quick_tune(df, "target", algo="decision_tree", n_iter=2, cv=2)
    assert hasattr(model, "predict")
    assert isinstance(params, dict)
    assert "r2" in report

from unittest.mock import patch
from breezeml import plot


def test_residuals_plot_smoke():
    """Smoke test to ensure plot.residuals executes without crashing."""
    import matplotlib.pyplot as plt

    # 1. Reuse the existing diabetes dataset setup
    df = datasets.diabetes()
    
    # 2. Train a fast linear model using your framework's regressors module
    model, _ = regressors.linear(df, "target")
    
    # 3. Extract features and target to pass into the plotting utility
    X_test = df.drop(columns=["target"])
    y_test = df["target"]

    # 4. Mock plt.show() so the test suite runs instantly without holding up execution
    with patch.object(plt, "show") as mock_show:
        try:
            plot.residuals(model, X_test, y_test)
        except Exception as e:
            import pytest
            pytest.fail(f"plot.residuals raised an unexpected exception: {e}")

        # Ensure that the display loop reached its final step successfully
        mock_show.assert_called_once()


def test_regressor_cv_report():
    df = datasets.diabetes()
    model, report = regressors.ridge(df, "target", cv=3)
    assert hasattr(model, "predict")
    assert "r2_std" in report
    assert "mae_std" in report
    assert "rmse_std" in report
