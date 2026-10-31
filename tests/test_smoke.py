import pytest

import ninety
from ninety.cli import main


def test_package_imports():
    assert ninety.__version__ == "0.1.0"


def test_cli_version(capsys):
    with pytest.raises(SystemExit):
        main(["--version"])
    assert capsys.readouterr().out.strip() == "ninety 0.1.0"
