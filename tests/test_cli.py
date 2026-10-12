"""The command-line entry point."""
import pytest

from cre_router import __version__
from cre_router.cli import main


def test_version_reports_the_package_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"cre {__version__}"
