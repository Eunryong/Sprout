"""Unit tests for Knowledge Synthesis & Progressive Intelligence Scaling Engine."""
from pathlib import Path

import pytest
import torch

from server.learning.codec import TOK, consequence_ce, encode_try
from server.learning.knowledge import (
    ConceptNode,
    EpisodeNode,
    KnowledgeGraph,
    PrincipleNode,
    cognitive_status,
    compute_personality,
    expand_knowledge,
    extract_key_terms,
    load_knowledge,
    principle_to_example,
    principle_to_fact,
    save_knowledge,
)
from server.learning.model import ModelCfg, build_model


def test_knowledge_models_and_persistence(tmp_path: Path):
    graph = KnowledgeGraph()
    assert len(graph.episodes) == 0
    assert len(graph.concepts) == 0
    assert len(graph.principles) == 0

    ep = EpisodeNode(
        turn=1,
        scene_text="식탁 위에 물이 엎질러져 있다.",
        action_text="수건으로 물을 닦았다.",
        outcome_text="물이 깨끗하게 닦였다.",
        takeaway="수건은 물을 잘 흡수한다.",
        outcome="solved",
        parent_mode="leave",
    )
    graph.episodes.append(ep)

    concept = ConceptNode(name="수건", observations=["물을 잘 흡수함"], valence_sum=1.0, count=1)
    graph.concepts["수건"] = concept

    prin = PrincipleNode(
        id="prin_001",
        principle="액체류는 흡수성 직물로 닦아내면 해결된다.",
        source_turns=[1],
        hop_depth=2,
        outcome_tendency="solved",
    )
    graph.principles.append(prin)

    save_file = tmp_path / "knowledge_graph.json"
    save_knowledge(graph, save_file)
    assert save_file.exists()

    loaded = load_knowledge(save_file)
    assert len(loaded.episodes) == 1
    assert loaded.episodes[0].turn == 1
    assert loaded.episodes[0].takeaway == "수건은 물을 잘 흡수한다."
    assert "수건" in loaded.concepts
    assert loaded.concepts["수건"].count == 1
    assert len(loaded.principles) == 1
    assert loaded.principles[0].hop_depth == 2


def test_cognitive_status_empty():
    graph = KnowledgeGraph()
    status_ko = cognitive_status(graph, lang="ko")
    assert status_ko["scale_b"] == 3.0
    assert status_ko["reasoning_hops"] == 2
    assert "3B" in status_ko["tier_name"]
    assert "3세" in status_ko["tier_name"]
    assert status_ko["tier_name"] == "3B (3세 수준 · 기초 개념 형성)"
    assert status_ko["n_episodes"] == 0
    assert status_ko["n_concepts"] == 0
    assert status_ko["n_principles"] == 0
    assert status_ko["recent_principles"] == []

    status_en = cognitive_status(graph, lang="en")
    assert status_en["scale_b"] == 3.0
    assert status_en["reasoning_hops"] == 2
    assert "3B" in status_en["tier_name"]
    assert status_en["tier_name"] == "3B (Age 3 · Basic Concepts)"


def test_cognitive_status_progressive_scaling_3b_to_7b():
    graph = KnowledgeGraph()

    # Initial state: 3B (Age 3)
    s0 = cognitive_status(graph)
    assert s0["scale_b"] == 3.0
    assert s0["reasoning_hops"] == 2
    assert "3B" in s0["tier_name"]
    assert "3세" in s0["tier_name"]
    assert "3세 수준" in s0["tier_name"]

    # Stage 1: Add episodes, concepts, and principles -> reaches 4B tier (4.0 <= scale_b < 5.0, hops = 3)
    for i in range(1, 9):
        graph.episodes.append(
            EpisodeNode(
                turn=i,
                scene_text=f"Scene {i}",
                action_text=f"Action {i}",
                outcome_text=f"Outcome {i}",
                takeaway=f"Takeaway {i}",
            )
        )
        graph.concepts[f"concept_{i}"] = ConceptNode(name=f"concept_{i}", count=1, valence_sum=1.0)

    for p_idx in range(1, 4):
        graph.principles.append(
            PrincipleNode(id=f"p{p_idx}", principle=f"Principle {p_idx}", source_turns=[p_idx], hop_depth=2)
        )

    s1 = cognitive_status(graph)
    assert 4.0 <= s1["scale_b"] < 5.0
    assert s1["reasoning_hops"] == 3
    assert "4B" in s1["tier_name"]
    assert "4세 수준" in s1["tier_name"]

    # Stage 2: Higher depth principles and more experiences -> reaches 5B tier (5.0 <= scale_b < 6.0, hops = 4)
    for i in range(9, 17):
        graph.episodes.append(
            EpisodeNode(
                turn=i,
                scene_text=f"Scene {i}",
                action_text=f"Action {i}",
                outcome_text=f"Outcome {i}",
                takeaway=f"Takeaway {i}",
            )
        )
        graph.concepts[f"concept_{i}"] = ConceptNode(name=f"concept_{i}", count=1, valence_sum=1.0)

    for j in range(4, 7):
        graph.principles.append(
            PrincipleNode(id=f"p{j}", principle=f"Principle {j}", source_turns=[j], hop_depth=3)
        )
    s2 = cognitive_status(graph)
    assert 5.0 <= s2["scale_b"] < 6.0
    assert s2["reasoning_hops"] == 4
    assert "5B" in s2["tier_name"]
    assert "5세 수준" in s2["tier_name"]

    # Stage 3: Reaches 6B tier (6.0 <= scale_b < 6.6, hops = 4)
    for i in range(17, 23):
        graph.episodes.append(
            EpisodeNode(
                turn=i,
                scene_text=f"Scene {i}",
                action_text=f"Action {i}",
                outcome_text=f"Outcome {i}",
                takeaway=f"Takeaway {i}",
            )
        )
        graph.concepts[f"concept_{i}"] = ConceptNode(name=f"concept_{i}", count=1, valence_sum=1.0)

    for j in range(7, 9):
        graph.principles.append(
            PrincipleNode(id=f"p{j}", principle=f"Principle {j}", source_turns=[j], hop_depth=3)
        )
    s3 = cognitive_status(graph)
    assert 6.0 <= s3["scale_b"] < 6.6
    assert s3["reasoning_hops"] == 4
    assert "6B" in s3["tier_name"]
    assert "6세 수준" in s3["tier_name"]

    # Stage 4: Deep synthesis and many episodes -> reaches 7B tier (clamped to 7.0, hops = 5)
    for k in range(23, 31):
        graph.episodes.append(
            EpisodeNode(
                turn=k,
                scene_text=f"Scene {k}",
                action_text=f"Action {k}",
                outcome_text=f"Outcome {k}",
                takeaway=f"Takeaway {k}",
            )
        )
    for k in range(23, 36):
        graph.concepts[f"concept_{k}"] = ConceptNode(name=f"concept_{k}", count=1, valence_sum=1.0)

    for j in range(9, 13):
        graph.principles.append(
            PrincipleNode(id=f"p{j}", principle=f"Deep Principle {j}", source_turns=[j], hop_depth=6)
        )

    s4 = cognitive_status(graph)
    assert s4["scale_b"] == 7.0  # Clamped at 7.0
    assert s4["reasoning_hops"] == 5
    assert "7B" in s4["tier_name"]
    assert "7세 수준" in s4["tier_name"]
    assert len(s4["recent_principles"]) == 5


def test_compute_personality_archetypes():
    # 1. Warm empathetic episodes produce warm_affectionate
    g_warm = KnowledgeGraph()
    for i in range(2):
        g_warm.episodes.append(
            EpisodeNode(
                turn=i + 1,
                scene_text="아이가 넘어져서 울고 있다.",
                action_text="다정하게 안아주고 토닥이며 위로했다.",
                outcome_text="아이가 울음을 그치고 방긋 웃었다.",
                takeaway="부모가 따뜻하게 안아주면 안심이 된다.",
                outcome="solved",
                parent_mode="say",
            )
        )
    p_warm = compute_personality(g_warm)
    assert p_warm["archetype"] == "warm_affectionate"
    assert p_warm["warm_count"] == 2
    assert p_warm["scold_count"] == 0
    assert p_warm["attachment"] > 0
    assert "사랑둥이" in p_warm["label"]

    status_warm = cognitive_status(g_warm)
    assert "personality" in status_warm
    assert status_warm["personality"]["archetype"] == "warm_affectionate"

    # 2. Scolding episodes produce timid_anxious
    g_scold = KnowledgeGraph()
    for i in range(2):
        g_scold.episodes.append(
            EpisodeNode(
                turn=i + 1,
                scene_text="장난감을 어질러 놓았다.",
                action_text="소리 지르며 크게 혼내고 야단쳤다.",
                outcome_text="아이가 무서워서 눈치를 보며 울었다.",
                takeaway="혼나면 무섭고 주눅이 든다.",
                outcome="worse",
                parent_mode="say",
            )
        )
    p_scold = compute_personality(g_scold)
    assert p_scold["archetype"] == "timid_anxious"
    assert p_scold["scold_count"] == 2
    assert p_scold["attachment"] < 0
    assert p_scold["confidence"] < 0
    assert "소심이" in p_scold["label"]

    # 3. Neglect/ignoring episodes produce lonely_sulky
    g_neglect = KnowledgeGraph()
    for i in range(2):
        g_neglect.episodes.append(
            EpisodeNode(
                turn=i + 1,
                scene_text="아이가 같이 놀자고 불렀다.",
                action_text="스마트폰만 보며 아이 말을 무시하고 혼자 두었다.",
                outcome_text="아이가 토라져서 혼자 구석에 앉아 있다.",
                takeaway="혼자 남겨지면 외롭고 서운하다.",
                outcome="worse",
                parent_mode="leave",
            )
        )
    p_neglect = compute_personality(g_neglect)
    assert p_neglect["archetype"] == "lonely_sulky"
    assert p_neglect["ignore_count"] == 2
    assert p_neglect["attachment"] < 0
    assert "투정이" in p_neglect["label"]

    # 4. Mixed warm + scolding/ignoring episodes produce wary_sensitive
    g_mixed = KnowledgeGraph()
    g_mixed.episodes.append(
        EpisodeNode(
            turn=1,
            scene_text="장난감을 쏟았다.",
            action_text="소리 지르며 혼냈다.",
            outcome_text="아이가 울었다.",
            takeaway="혼나서 무서웠다.",
            outcome="worse",
            parent_mode="say",
        )
    )
    g_mixed.episodes.append(
        EpisodeNode(
            turn=2,
            scene_text="아이가 울고 있다.",
            action_text="다정하게 안아주고 위로했다.",
            outcome_text="아이가 안정을 되찾았다.",
            takeaway="엄마 품은 따뜻하다.",
            outcome="solved",
            parent_mode="say",
        )
    )
    p_mixed = compute_personality(g_mixed)
    assert p_mixed["archetype"] == "wary_sensitive"
    assert p_mixed["warm_count"] == 1
    assert p_mixed["scold_count"] == 1
    assert -0.5 <= p_mixed["attachment"] <= 0.35
    assert "눈치 보지만 다정함에 기대는 아이" in p_mixed["label"]

    # 5. Solo solved episodes produce bold_explorer
    g_solo = KnowledgeGraph()
    for i in range(2):
        g_solo.episodes.append(
            EpisodeNode(
                turn=i + 1,
                scene_text="어려운 퍼즐이 있다.",
                action_text="혼자 끈기 있게 맞춰 보았다.",
                outcome_text="퍼즐을 스스로 완성했다.",
                takeaway="혼자 힘으로 해내니 뿌듯하다.",
                outcome="solved",
                parent_mode="solo",
            )
        )
    p_solo = compute_personality(g_solo)
    assert p_solo["archetype"] == "bold_explorer"
    assert p_solo["confidence"] >= 0.25
    assert p_solo["solved_count"] == 2
    assert "탐험가" in p_solo["label"]

    # 6. Empty graph produces innocent_toddler
    g_empty = KnowledgeGraph()
    p_empty = compute_personality(g_empty)
    assert p_empty["archetype"] == "innocent_toddler"
    assert p_empty["attachment"] == 0.0
    assert p_empty["confidence"] == 0.0


def test_cognitive_status_progressive_scaling_1b_to_7b():
    # Verify custom base_scale_b=1.0 starts at 1B
    graph = KnowledgeGraph()
    s0 = cognitive_status(graph, base_scale_b=1.0)
    assert s0["scale_b"] == 1.0
    assert s0["reasoning_hops"] == 1
    assert "1B" in s0["tier_name"]
    assert "1세 수준" in s0["tier_name"]


def test_extract_key_terms_dynamic():
    text_ko = "방 안에 있는 차가운 얼음을 따뜻한 물에 넣으면 녹는다."
    terms_ko = extract_key_terms(text_ko)
    assert "얼음" in terms_ko or any("얼음" in t for t in terms_ko)
    assert len(terms_ko) >= 2
    # Verify common stop particles are filtered
    assert "에" not in terms_ko
    assert "는" not in terms_ko

    text_en = "The heavy wooden box falls when dropped on the floor."
    terms_en = extract_key_terms(text_en)
    assert "heavy" in terms_en
    assert "the" not in terms_en
    assert "on" not in terms_en


def test_expand_knowledge_fallback_mode():
    graph = KnowledgeGraph()

    # Episode 1: No prior episodes, dynamic extraction registers concepts
    ep1 = EpisodeNode(
        turn=1,
        scene_text="바닥에 쏟아진 주스를 보았다.",
        action_text="수건으로 주스를 닦았다.",
        outcome_text="바닥이 마르고 깨끗해졌다.",
        takeaway="수건은 주스를 흡수해서 깨끗하게 만든다.",
        outcome="solved",
    )
    res1 = expand_knowledge(graph, ep1, client=None, lang="ko")
    assert len(graph.episodes) == 1
    assert len(graph.concepts) > 0
    assert len(graph.principles) == 0  # No prior turns to synthesize
    assert res1["synthesized_principle"] is None

    # Episode 2: Shares terms/context with turn 1 -> dynamic synthesis triggers
    ep2 = EpisodeNode(
        turn=2,
        scene_text="책상에 물이 엎질러졌다.",
        action_text="수건으로 물을 닦았다.",
        outcome_text="책상이 말끔해졌다.",
        takeaway="수건은 물을 빠르게 흡수한다.",
        outcome="solved",
    )
    res2 = expand_knowledge(graph, ep2, client=None, lang="ko")
    assert len(graph.episodes) == 2
    assert len(graph.principles) == 1
    assert res2["synthesized_principle"] is not None
    assert 1 in graph.principles[0].source_turns
    assert 2 in graph.principles[0].source_turns
    assert res2["fact"] is not None
    assert res2["cognitive_status"]["scale_b"] > res1["cognitive_status"]["scale_b"]


def test_expand_knowledge_with_mock_client():
    class DummyClient:
        def complete_json(self, agent, system, user, schema):
            return {
                "concepts": [
                    {"name": "인과율", "observation": "원인과 결과는 일정한 법칙을 따른다."}
                ],
                "synthesized_principle": "원인에 맞는 적절한 행동이 언제나 문제를 해결한다.",
                "source_turns": [1, 2],
                "hop_depth": 3,
            }

    graph = KnowledgeGraph()
    ep = EpisodeNode(
        turn=2,
        scene_text="화분에 물이 부족하다.",
        action_text="물을 주었다.",
        outcome_text="식물이 생기를 되찾았다.",
        takeaway="식물에는 물이 필요하다.",
        outcome="solved",
    )
    res = expand_knowledge(graph, ep, client=DummyClient(), world_premise="정원", lang="ko")
    assert "인과율" in graph.concepts
    assert len(graph.principles) == 1
    assert graph.principles[0].principle == "원인에 맞는 적절한 행동이 언제나 문제를 해결한다."
    assert graph.principles[0].hop_depth == 3
    assert res["fact"] is not None


def test_principle_to_fact_and_tinylm_training():
    principle = PrincipleNode(
        id="prin_test",
        principle="뜨거운 물체는 식힌 후에 만져야 안전하다",
        source_turns=[1, 2],
        hop_depth=3,
        outcome_tendency="solved",
    )

    fact = principle_to_fact(principle)
    assert fact.result == "solved"
    assert fact.action_text == principle.principle
    assert fact.takeaway == principle.principle
    assert fact.option.option_text == principle.principle

    example = principle_to_example(principle, day=2)
    assert example.id == "synth_prin_test"
    assert len(example.facts) == 1

    # Verify that encode_try incorporates text tokens
    encoded_ids = encode_try(example.obs, fact.option, said=False)
    txt_ids = [tok_id for tok, tok_id in TOK.items() if tok.startswith("TXT_")]
    matched_txt_tokens = [tok_id for tok_id in encoded_ids if tok_id in txt_ids]
    assert len(matched_txt_tokens) > 0, "Expected text tokens from the principle text"

    # Train TinyLM on this synthetic principle example
    cfg = ModelCfg(d_model=32, n_layers=2, n_heads=2, d_ff=64)
    model = build_model(cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    loss_before = consequence_ce(model, [example])
    assert torch.isfinite(loss_before)
    loss_before.backward()

    # Verify gradients computed across model parameters
    has_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
    assert has_grad, "Expected non-zero gradients on TinyLM parameters"

    optimizer.step()
    with torch.no_grad():
        loss_after = consequence_ce(model, [example])
    assert torch.isfinite(loss_after)
