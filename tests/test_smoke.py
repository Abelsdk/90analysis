import ninety


def test_package_imports():
    assert ninety.__version__ == "0.1.0"


def test_cli_entry_point_runs(capsys):
    ninety.main()
    assert capsys.readouterr().out.strip() == "ninety 0.1.0"
