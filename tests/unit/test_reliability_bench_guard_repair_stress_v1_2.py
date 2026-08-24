from eval.reliability_bench.guard_repair_stress_v1_2 import run_controlled_repair_stress


def test_controlled_stress_covers_each_reason_and_each_repair_allows():
    report = run_controlled_repair_stress()
    assert report["population"] == "controlled synthetic candidates; not natural LLM errors"
    assert report["case_count"] == 7
    assert report["all_reason_codes_covered"] is True
    assert report["all_repairs_allowed"] is True
    assert {item["case"] for item in report["results"]} == {
        "invalid_schema",
        "unknown_entity",
        "invalid_reference",
        "missing_required_support",
        "stale_cited_evidence",
        "unverified_revalidation",
        "premature_finalization",
    }
