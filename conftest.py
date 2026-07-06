# Present so pytest puts the repo root on sys.path and `import taa` works without install.
import pytest

from taa.data.synthetic import make_synthetic_bundle, make_synthetic_prices


@pytest.fixture(scope="session")
def synthetic_bundle():
    """One shared read-only synthetic bundle. Tests poison copies, never this instance."""
    return make_synthetic_bundle()


@pytest.fixture(scope="session")
def synthetic_prices():
    return make_synthetic_prices()
