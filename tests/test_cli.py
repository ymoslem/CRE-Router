"""The command-line entry point."""
import json

import pytest

from cre_router import __version__
from cre_router.cli import main


def test_version_reports_the_package_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out.strip() == f"cre {__version__}"


def test_fit_reports_and_resolves_a_crossover_near_tie(tmp_path, capsys):
    stats = {"cluster_sizes": {"0": 5211, "1": 3789},
             "models": {"Qwen": {"errors": {"0": 0.336395, "1": 0.363244},
                                 "cluster_tpot_ms": {"0": 8.61, "1": 8.47}},
                        "Gemma": {"errors": {"0": 0.269445, "1": 0.298392},
                                  "cluster_tpot_ms": {"0": 19.73, "1": 19.63}}}}
    path = tmp_path / "stats.json"
    path.write_text(json.dumps(stats))
    main(["fit", "--stats", str(path), "--budget", "14"])
    plain = capsys.readouterr().out
    assert "switching order unresolved" in plain and "set --crossover-tol" in plain
    assert "{'0': 'Qwen', '1': 'Qwen'}" in plain
    main(["fit", "--stats", str(path), "--budget", "14", "--crossover-tol", "auto"])
    tied = capsys.readouterr().out
    assert "{'0': 'Qwen', '1': 'Gemma'}" in tied and "lambda* = none" in tied
