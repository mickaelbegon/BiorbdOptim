def test_muscle_driven_block_shooting_benchmark_builds_and_reports_sparsity():
    from bioptim.examples.benchmarks import block_shooting_arm_reaching as benchmark

    result = benchmark.run_case(n_shooting=4, n_blocks=2, solve=False)

    assert result["transcription"] == "B=2"
    assert result["n_shooting"] == 4
    assert result["n_blocks"] == 2
    assert result["variables"] > 0
    assert result["constraints"] > 0
    assert result["constraint_jacobian_nnz"] > 0
    assert 0 < result["constraint_jacobian_density"] <= 1
    assert result["build_seconds"] > 0
    assert result["peak_rss_mb"] > 0
    assert result["solve_seconds"] is None
