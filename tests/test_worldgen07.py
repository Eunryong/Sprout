"""The randomly made 0-7 years worlds: every candidate relation is possible under the base affordances, and every seed gives a valid, playable world."""
import yaml

from server.world.worldgen import CONFIG, check_sheet, load_pools, ontology_dict, template_world

POOLS = load_pools(CONFIG / "world_pools_age07.yaml")


def test_every_candidate_relation_can_really_be_done():
    base = yaml.safe_load((CONFIG / POOLS["base"]).read_text())
    aff = {v: set(ks) for v, ks in base["affordances"].items()}
    bad = [(t, v, k) for t, p in POOLS["problems"].items() for v, k, _ in p["rules"] if k not in aff.get(v, ())]
    assert bad == []
    assert set(POOLS["problems"]) == set(base["rules"])


def test_template_worlds_are_valid_and_differ():
    sheets = [template_world(s, POOLS) for s in range(30)]
    assert all(check_sheet(s, POOLS) == [] for s in sheets)
    assert len({str(sorted(s.rules.items())) for s in sheets}) > 25
    assert len({tuple(sorted((v, k) for v, k, x in s.rules["COLD"] if x == 2)) for s in sheets}) > 2  # what warms the baby differs between households


def test_the_written_world_keeps_the_ages_stages_and_affordances():
    d = ontology_dict(template_world(5, POOLS), POOLS)
    assert d["schedule"]["closes"] == {"BABY_HUNGRY": 12} and d["followups"]["lingers"] == []
    assert d["stages"][0]["voice"] == "baby" and d["acts_from_month"] == 12 and d["game"] == {"days": 14, "per_day": 6} and d["affordances"]
