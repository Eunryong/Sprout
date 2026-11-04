"""Unit tests for Knowledge Synthesis & Progressive Intelligence Scaling Engine."""
from pathlib import Path

import pytest
import torch

from server.learning.codec import TOK, consequence_ce, encode_try
from server.learning.knowledge import (
    ConceptNode,
    EpisodeNode,
    KnowledgeGraph,
    PersonalityNode,
    PrincipleNode,
    SynthesisResponse,
    cognitive_status,
    compute_personality,
    expand_knowledge,
    extract_key_terms,
    load_knowledge,
    personality_to_example,
    personality_to_fact,
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

    # Initial state (0 scenes): 3B (Age 3)
    s0 = cognitive_status(graph)
    assert s0["scale_b"] == 3.0
    assert s0["reasoning_hops"] == 2
    assert "3B" in s0["tier_name"]
    assert "3세" in s0["tier_name"]
    assert "3세 수준" in s0["tier_name"]

    # After 1 scene (1 month: 2 episodes, 4 concepts, 1 principle with hop_depth=5)
    g_early = KnowledgeGraph()
    for i in range(1, 3):
        g_early.episodes.append(
            EpisodeNode(
                turn=i,
                scene_text=f"Scene {i}",
                action_text=f"Action {i}",
                outcome_text=f"Outcome {i}",
                takeaway=f"Takeaway {i}",
            )
        )
    for i in range(1, 5):
        g_early.concepts[f"concept_{i}"] = ConceptNode(name=f"concept_{i}", count=1, valence_sum=1.0)
    g_early.principles.append(
        PrincipleNode(id="p1", principle="Principle 1", source_turns=[1], hop_depth=5)
    )
    s_1scene = cognitive_status(g_early)
    assert s_1scene["scale_b"] == 3.1  # Increases by ~0.08B, NOT 0.5B, even with high hop_depth

    # After 3 scenes (3 months: 6 episodes, 13 concepts, 4 principles with hop_depth=5)
    for i in range(3, 7):
        g_early.episodes.append(
            EpisodeNode(
                turn=i,
                scene_text=f"Scene {i}",
                action_text=f"Action {i}",
                outcome_text=f"Outcome {i}",
                takeaway=f"Takeaway {i}",
            )
        )
    for i in range(5, 14):
        g_early.concepts[f"concept_{i}"] = ConceptNode(name=f"concept_{i}", count=1, valence_sum=1.0)
    for j in range(2, 5):
        g_early.principles.append(
            PrincipleNode(id=f"p{j}", principle=f"Principle {j}", source_turns=[j], hop_depth=5)
        )
    s_3scenes = cognitive_status(g_early)
    assert 3.2 <= s_3scenes["scale_b"] <= 3.3  # +0.25B for 3 months

    # Stage 1 (1 year = 12 scenes = 24 episodes, 45 concepts, 15 principles with hop_depth=4)
    for i in range(1, 25):
        graph.episodes.append(
            EpisodeNode(
                turn=i,
                scene_text=f"Scene {i}",
                action_text=f"Action {i}",
                outcome_text=f"Outcome {i}",
                takeaway=f"Takeaway {i}",
            )
        )
    for i in range(1, 46):
        graph.concepts[f"concept_{i}"] = ConceptNode(name=f"concept_{i}", count=1, valence_sum=1.0)
    for p_idx in range(1, 16):
        graph.principles.append(
            PrincipleNode(id=f"p{p_idx}", principle=f"Principle {p_idx}", source_turns=[p_idx], hop_depth=4)
        )

    s1 = cognitive_status(graph)
    assert 4.0 <= s1["scale_b"] < 4.2
    assert s1["reasoning_hops"] == 3
    assert "4B" in s1["tier_name"]
    assert "4세 수준" in s1["tier_name"]

    # Stage 2 (2 years = 24 scenes = 48 episodes, 90 concepts, 30 principles with hop_depth=5)
    for i in range(25, 49):
        graph.episodes.append(
            EpisodeNode(
                turn=i,
                scene_text=f"Scene {i}",
                action_text=f"Action {i}",
                outcome_text=f"Outcome {i}",
                takeaway=f"Takeaway {i}",
            )
        )
    for i in range(46, 91):
        graph.concepts[f"concept_{i}"] = ConceptNode(name=f"concept_{i}", count=1, valence_sum=1.0)
    for j in range(16, 31):
        graph.principles.append(
            PrincipleNode(id=f"p{j}", principle=f"Principle {j}", source_turns=[j], hop_depth=5)
        )
    s2 = cognitive_status(graph)
    assert 5.0 <= s2["scale_b"] < 5.2
    assert s2["reasoning_hops"] == 4
    assert "5B" in s2["tier_name"]
    assert "5세 수준" in s2["tier_name"]

    # Stage 3 (3 years = 36 scenes = 72 episodes, 135 concepts, 45 principles with hop_depth=6)
    for i in range(49, 73):
        graph.episodes.append(
            EpisodeNode(
                turn=i,
                scene_text=f"Scene {i}",
                action_text=f"Action {i}",
                outcome_text=f"Outcome {i}",
                takeaway=f"Takeaway {i}",
            )
        )
    for i in range(91, 136):
        graph.concepts[f"concept_{i}"] = ConceptNode(name=f"concept_{i}", count=1, valence_sum=1.0)
    for j in range(31, 46):
        graph.principles.append(
            PrincipleNode(id=f"p{j}", principle=f"Principle {j}", source_turns=[j], hop_depth=6)
        )
    s3 = cognitive_status(graph)
    assert 6.0 <= s3["scale_b"] < 6.2
    assert s3["reasoning_hops"] == 4
    assert "6B" in s3["tier_name"]
    assert "6세 수준" in s3["tier_name"]

    # Stage 4 (4 years = 48 scenes = 96 episodes, 180 concepts, 60 principles with hop_depth=6)
    for k in range(73, 97):
        graph.episodes.append(
            EpisodeNode(
                turn=k,
                scene_text=f"Scene {k}",
                action_text=f"Action {k}",
                outcome_text=f"Outcome {k}",
                takeaway=f"Takeaway {k}",
            )
        )
    for k in range(136, 181):
        graph.concepts[f"concept_{k}"] = ConceptNode(name=f"concept_{k}", count=1, valence_sum=1.0)
    for j in range(46, 61):
        graph.principles.append(
            PrincipleNode(id=f"p{j}", principle=f"Principle {j}", source_turns=[j], hop_depth=6)
        )

    s4 = cognitive_status(graph)
    assert s4["scale_b"] == 7.0  # Clamped at 7.0
    assert s4["reasoning_hops"] == 5
    assert "7B" in s4["tier_name"]
    assert "7세 수준" in s4["tier_name"]
    assert len(s4["recent_principles"]) == 5


def test_dynamic_personality_evolution_and_tinylm_training():
    # 1. Empty graph has empty dynamic personality (label == "", attachment == 0.0, confidence == 0.0)
    g_empty = KnowledgeGraph()
    p_empty = compute_personality(g_empty)
    assert p_empty["label"] == ""
    assert p_empty["traits"] == []
    assert p_empty["tone_guide"] == ""
    assert p_empty["core_self"] == ""
    assert p_empty["history"] == []
    assert p_empty["attachment"] == 0.0
    assert p_empty["confidence"] == 0.0
    assert "archetype" not in p_empty

    # 2. Calling expand_knowledge() across guided solved episodes vs solo worse episodes
    # dynamically evolves graph.personality (label, traits, tone_guide, core_self, history)
    # and produces distinct attachment, confidence, and core_self without any static archetype strings.
    g_guided = KnowledgeGraph()
    res_g1 = expand_knowledge(
        g_guided,
        EpisodeNode(
            turn=1,
            scene_text="식탁 위에 우유를 쏟았다.",
            action_text="부모와 함께 걸레로 우유를 닦았다.",
            outcome_text="식탁이 깨끗해졌다.",
            takeaway="함께 닦으니 잘 해결된다.",
            outcome="solved",
            parent_mode="show",
        ),
        client=None,
    )
    assert g_guided.personality.label != ""
    assert len(g_guided.personality.traits) > 0
    assert g_guided.personality.core_self == "함께 닦으니 잘 해결된다."
    assert g_guided.personality.history == ["함께 닦으니 잘 해결된다."]

    res_g2 = expand_knowledge(
        g_guided,
        EpisodeNode(
            turn=2,
            scene_text="장난감 상자가 무겁다.",
            action_text="부모의 도움을 받아 정리했다.",
            outcome_text="방이 말끔해졌다.",
            takeaway="도움을 받으면 든든하다.",
            outcome="solved",
            parent_mode="say",
        ),
        client=None,
    )
    p_guided = compute_personality(g_guided)
    assert p_guided["attachment"] > 0.0
    assert p_guided["confidence"] > 0.0
    assert g_guided.personality.core_self == "도움을 받으면 든든하다."
    assert len(g_guided.personality.history) == 2

    # Solo worse episodes
    g_solo = KnowledgeGraph()
    res_s1 = expand_knowledge(
        g_solo,
        EpisodeNode(
            turn=1,
            scene_text="높은 곳에 있는 장난감을 꺼내려 했다.",
            action_text="혼자 의자를 밟고 올라갔다가 넘어졌다.",
            outcome_text="무릎을 다치고 장난감도 떨어졌다.",
            takeaway="혼자 하려다 다치니 무섭다.",
            outcome="worse",
            parent_mode="leave",
        ),
        client=None,
    )
    res_s2 = expand_knowledge(
        g_solo,
        EpisodeNode(
            turn=2,
            scene_text="어두운 방에 장난감이 있다.",
            action_text="혼자 들어가려다 벽에 부딪혔다.",
            outcome_text="어둡고 아파서 울었다.",
            takeaway="어두운 곳에 혼자 남겨지면 서럽다.",
            outcome="worse",
            parent_mode="leave",
        ),
        client=None,
    )
    p_solo = compute_personality(g_solo)
    assert p_solo["attachment"] < 0.0
    assert p_solo["confidence"] < 0.0
    assert p_guided["attachment"] > p_solo["attachment"]
    assert p_guided["confidence"] > p_solo["confidence"]
    assert g_guided.personality.core_self != g_solo.personality.core_self
    assert g_guided.personality.label != g_solo.personality.label

    # Verify no static archetype strings
    for static_name in ["warm_affectionate", "timid_anxious", "lonely_sulky", "bold_explorer", "innocent_toddler"]:
        assert static_name not in p_guided
        assert static_name not in p_solo
        assert static_name != p_guided["label"]
        assert static_name != p_solo["label"]

    # 3. Calling expand_knowledge() with a mock LLM client returning SynthesisResponse updates
    # graph.personality with the LLM-synthesized personality_label, personality_traits, personality_tone, and personality_shift.
    class MockSynthClient:
        def complete_json(self, agent, system, user, schema):
            return SynthesisResponse(
                concepts=[],
                synthesized_principle="경험을 통해 조금씩 배워간다.",
                source_turns=[1],
                hop_depth=2,
                personality_label="호기심 많고 조심성 있는 탐구자",
                personality_traits=["호기심", "신중함", "관찰력"],
                personality_tone="눈을 반짝이며 차분하고 솔직하게 말한다.",
                personality_shift="부모가 지켜봐줄 때 더 안심하고 새로운 것을 시도할 수 있음을 느꼈다.",
            )

    g_llm = KnowledgeGraph()
    res_llm = expand_knowledge(
        g_llm,
        EpisodeNode(
            turn=1,
            scene_text="나비가 꽃에 앉아 있다.",
            action_text="가만히 숨죽이고 관찰했다.",
            outcome_text="나비가 도망가지 않고 꿀을 빨았다.",
            takeaway="조용히 다가가면 관찰할 수 있다.",
            outcome="solved",
            parent_mode="show",
        ),
        client=MockSynthClient(),
        lang="ko",
    )
    assert g_llm.personality.label == "호기심 많고 조심성 있는 탐구자"
    assert g_llm.personality.traits == ["호기심", "신중함", "관찰력"]
    assert g_llm.personality.tone_guide == "눈을 반짝이며 차분하고 솔직하게 말한다."
    assert g_llm.personality.core_self == "부모가 지켜봐줄 때 더 안심하고 새로운 것을 시도할 수 있음을 느꼈다."
    assert g_llm.personality.history[-1] == "부모가 지켜봐줄 때 더 안심하고 새로운 것을 시도할 수 있음을 느꼈다."

    # 4. expand_knowledge() returns "personality_example" which encodes into TXT_* tokens
    # and produces non-zero gradients in TinyLM (consequence_ce(model, [pers_ex]).backward()).
    pers_ex = res_llm["personality_example"]
    assert pers_ex is not None
    assert pers_ex.source == "personality"
    encoded_ids = encode_try(pers_ex.obs, pers_ex.facts[0].option, said=False)
    txt_ids = [tok_id for tok, tok_id in TOK.items() if tok.startswith("TXT_")]
    matched_txt_tokens = [tok_id for tok_id in encoded_ids if tok_id in txt_ids]
    assert len(matched_txt_tokens) > 0, "Expected text tokens from personality text"

    cfg = ModelCfg(d_model=32, n_layers=2, n_heads=2, d_ff=64)
    model = build_model(cfg)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    loss = consequence_ce(model, [pers_ex])
    assert torch.isfinite(loss)
    loss.backward()

    has_grad = any(p.grad is not None and torch.sum(torch.abs(p.grad)) > 0 for p in model.parameters())
    assert has_grad, "Expected non-zero gradients on TinyLM parameters from personality_example"

    optimizer.step()
    with torch.no_grad():
        loss_after = consequence_ce(model, [pers_ex])
    assert torch.isfinite(loss_after)


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
