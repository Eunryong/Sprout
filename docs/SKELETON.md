# 뼈대(Walking Skeleton) 설계 — 기본 기능

> **현행 구현 안내 (2026-10-26)**: 이 문서는 v1(단어 하나를 요청하고 슬롯 하나를 고르는 세계)의 설계와 안정화 실험 기록이다. 코드는 WORLD.md의 엔진 v2로 교체됐고, v1 전체 스냅샷은 `archive/v1_skeleton_2026-10-26.tar.gz`, 당시 실험 스크립트는 `scripts/legacy_v1/`에 있다(스냅샷의 코드에 대해서만 동작). 온라인 학습 안정화 결과(10장)와 불변식은 v2에도 그대로 이어진다. 현행 구조는 [WORLD.md](WORLD.md) 13장.

작성: 2026-10-26 · 개정: 핵심 루프를 "상황 → 선택 → 학습"으로 변경 (수면 학습 폐기) · 상위 문서: [DESIGN.md](DESIGN.md) · 상태: 설계

## 0. 핵심 루프

```
Director(서브에이전트)가 상황 카드 2장 제안 ─▶ 엔진이 검증·확정
   ─▶ 플레이어가 카드 1장 선택 ─▶ 상황 시작 (요청 + 물건들)
   ─▶ 플레이어가 이 상황에서 할 행동을 선택  = 시범(Show)
   ─▶ 엔진이 결과 확정(세계 변화) ─▶ 선택이 Example로 기록
   ─▶ 즉시 학습 스텝 (아이 모델 가중치 갱신, 목표 ≤2초)
   ─▶ 다음 상황부터 아이가 스스로 시도(Try) ─▶ 플레이어가 칭찬/교정 ─▶ 다시 학습
```

- 플레이어는 아이의 **본보기**다. 플레이어가 고른 행동이 곧 학습 데이터다
- **정보 비대칭**: 플레이어는 물건의 종류 라벨(예: cup)까지 보고, 아이는 속성(색·모양·크기·재질)만 본다
- 플레이어가 틀린 행동을 골라도 **그대로 학습**한다. 결과(요청 미충족)는 세계에 반영된다 — 데이터 품질이 아이를 만든다
- 한 상황은 정확히 한 번 해결된다: `Show` / `Try`(+칭찬·교정) / `Skip`. Skip은 학습 없음
- 하루 = 상황 6개 + 저녁 퀴즈(학습에 안 쓰는 평가). 수면 단계는 없다

**뼈대 완료 기준** — 터미널에서 아래가 한 번에 된다:

```
새 게임 → 카드 선택 → 상황 → 행동 선택(시범) → 즉시 학습 → … → 아이가 스스로 시도 → 칭찬/교정 → 학습
→ 저녁 퀴즈 점수가 ckpt_000·무작위보다 높음 → 저장·종료 → 재개하면 모델·세계 그대로
```

- UI: 터미널 CLI만. 웹은 D5
- LLM: `MockClient`(템플릿 응답)만. Meta API는 키가 나오면 같은 인터페이스로 교체
- 놀이: A(이름 알아듣기)만. B는 D6

## 1. 기능 목록

| ID | 기능 | 뼈대 | 이후 확장 |
|---|---|---|---|
| F1 | 세계 상태 + 이벤트 로그 (append / fold / 멱등) | ● | 후속 사건 연쇄 |
| F2 | 물건 카탈로그 + 개체 샘플링 | ● 4종 (cup, cloth, paper, ball) | 7종 |
| F3 | 상황(Situation) 모델 + 플레이어 뷰 / 아이 관찰 분리 | ● `find` 요청만 | `use`(놀이 B), 상황 플래그 |
| F4 | 어휘·토크나이저 | ● 전체 어휘 고정 | — |
| F5 | 아이 모델 + 행동 선택 (문법 마스크) | ● | — |
| F6 | 선택·피드백 → Example 기록 (출처 규칙 4.5) | ● show / praise / correct | — |
| F7 | **즉시 학습 스텝** (새 Example + replay) + 체크포인트 | ● SFT만 | 관찰 마스킹 보조 손실, DPO-lite |
| F8 | 평가 — 분리 퀴즈, 대조군, 무작위 기준 | ● | 탐침 패널(4.6) |
| F9 | 게임 루프 + 저장/재개 + CLI | ● | FastAPI + 웹 |
| F10 | LLM 클라이언트 인터페이스 + Mock + 폴백 라우터 | ● Mock만 | Meta/파트너 연결 |
| F11 | **Director(상황 카드 2장 제안)** / Narrator | ● 템플릿 구현 | LLM 구현 |

뼈대에서 **뺀 것**: 웹 UI, 탐침 패널, 놀이 B, NPC, 이야기 연쇄, DPO, Muse Image, 배포, 한국어 토글, Interpreter(자유문장 입력).

---

## 2. 리포 구조

```
meta/
  pyproject.toml            # uv, Python 3.12
  config/
    game.yaml               # 하루 상황 수(6), 슬롯 수, 퀴즈 수, 시드
    model.yaml              # d_model, layers, heads, lr, online{replay, steps}
    agents.yaml             # 에이전트별 모델명·폴백 체인
  server/
    world/
      catalog.py            # F2
      events.py  state.py   # F1 이벤트 + apply/fold
      engine.py             # F1 명령 검증 → 이벤트
      situation.py          # F3 Situation, PlayerView, child Observation
    learning/               # ※ agents/ import 금지
      vocab.py  tokenizer.py           # F4
      model.py  policy.py              # F5
      dataset.py  augment.py           # F6
      learner.py  checkpoints.py       # F7
      evaluator.py                     # F8
    agents/
      client.py             # F10
      director.py  narrator.py         # F11
      schemas.py
    game/
      session.py            # F9 게임 루프
      saves.py
    cli.py
  scripts/
    exp_task_a.py           # 합성 데이터 학습 가능성 실험 (D1 게이트)
  tests/
  saves/                    # .gitignore
```

---

## 3. 데이터 계약

### 3.1 세계

```python
class Phase(StrEnum):  CARDS, SITUATION, EVENING, DONE   # 하루의 진행 단계

class ObjectInst(BaseModel):
    id: str; kind: str           # kind는 엔진·플레이어 뷰 전용
    color: str; shape: str; size: str; material: str

class Situation(BaseModel):
    id: str; type: Literal["find"]        # 이후 "use"
    word: str; word_token: str            # "bobo" / "NAME_03" — word의 뜻(kind)은 엔진만 안다
    word_kind: str
    scene: list[str]                      # 물건 id, 화면 순서
    location: str
    resolution: Literal["show","try","skip"] | None

class WorldState(BaseModel):
    day: int; phase: Phase
    objects: dict[str, ObjectInst]
    cards: list[Situation] | None         # 제안된 2장
    current: Situation | None
    situations_done: int                  # 하루 6개
    words: dict[str, str]                 # word → word_token (kind는 두지 않음: 아이 쪽 지식이 아님)
    last_try: ChildAct | None             # 칭찬/교정 대상
    applied_action_ids: set[str]
```

### 3.2 이벤트

```python
EventType = DAY_STARTED | CARDS_OFFERED | CARD_PICKED
          | SHOWN | CHILD_TRIED | PRAISED | CORRECTED | SKIPPED
          | LEARNED | QUIZ_TAKEN | DAY_ENDED

class Event(BaseModel):  seq; action_id; day; type; payload
```

- `apply(state, event) -> WorldState` 순수 함수, `fold(events)`로 재구성
- 상태 변경은 **오직 이벤트로만**. `LEARNED{ckpt, report}`는 학습 후 Session이 제출한다(엔진은 PyTorch를 모름)

### 3.3 명령

```python
DrawCards(action_id)            # Director → 후보 2장 (엔진이 검증)
PickCard(action_id, index)
Show(action_id, slot)           # 시범. 틀려도 허용
Try(action_id)                  # 아이가 스스로 행동 (모델 호출)
Praise(action_id)               # 직전 Try가 맞았을 때만
Correct(action_id, slot)        # 직전 Try가 틀렸을 때만
Skip(action_id)
EndDay(action_id)               # 6개 해결 후 → 퀴즈
```

`Engine.submit(cmd) -> list[Event]`: ① 이미 적용된 `action_id`면 빈 리스트(멱등) ② 전제조건 검사 → `RuleError(code)` ③ 이벤트 생성·append.
전제조건 예: 카드 없이 PickCard, 진행 중 상황 없이 Show, 이미 해결된 상황에 재선택, 범위 밖 슬롯, 직전 Try 없이 Praise.

### 3.4 두 가지 뷰 (정답 누출 방지의 핵심)

```python
class PlayerView(BaseModel):      # 사람에게 보임
    narration: str                # "Grandma: 'Could you bring me the bobo?'"
    word_hint: str | None         # 처음 나온 단어 → "bobo = cup (Grandma's word)"
    cards: list[SlotCard]         # index, color, shape, size, material, kind_label

class Observation(BaseModel):     # 아이에게 보임 — 유일한 모델 입력
    request: str                  # "REQ_FIND"
    name_token: str               # "NAME_03"
    slots: list[SlotView]         # color, shape, size, material 만. id·kind 없음
```

`answer_slots`(정답)는 엔진이 `Situation.word_kind`로 계산하는 별도 값이며 Observation에 없다.

### 3.5 학습 데이터

```python
class Example(BaseModel):
    id: str; day: int
    source: Literal["show", "praise", "correct"]
    obs: Observation; target_slot: int; weight: float     # show·correct 1.0, praise 0.5
    rejected_slot: int | None                             # correct일 때만 (DPO용 저장)
    was_right: bool                                       # 로그·분석용. 학습 입력 아님
```

- `Show`가 틀린 슬롯이어도 `target_slot`은 플레이어가 고른 슬롯. `was_right=False`로만 표시
- `Try`에 칭찬/교정이 없으면(Skip) Example 없음

### 3.6 저장 폴더

```
saves/<save_id>/
  meta.json            # 시드, 설정 스냅샷
  events.jsonl         # 진실의 원천
  dataset.jsonl        # Example
  quiz.jsonl           # 평가셋 (학습 금지) + 해시
  ckpt/ckpt_000.pt     # 무작위 초기 (절대 덮어쓰지 않음)
  ckpt/latest.pt       # 학습 스텝마다 갱신
  ckpt/day_01.pt …     # 하루 끝 스냅샷 (성장 앨범)
```

---

## 4. 모듈 인터페이스

### learning
```python
encode(obs: Observation) -> list[int]
target_tokens(slot: int) -> list[int]               # [ACT_PICK, SLOT_k, EOS]

class TinyLM(nn.Module): forward(ids) -> logits
act(model, obs) -> ChildAct(slot, probs)            # ACT_PICK 뒤를 존재하는 모든 슬롯으로만 마스킹

# learner.py — 수면이 아니라 선택 직후의 학습
learn(model, new: Example, buffer: list[Example], cfg: OnlineCfg, rng) -> LearnReport
#   배치 = new(oversample) + replay k개. n step. 문법 마스크 밖 토큰은 손실에 불포함
#   LearnReport{steps, loss_before, loss_after, seconds, data_hash}

# evaluator.py
make_quiz(state, rng, n) -> list[QuizItem]          # 이미 나온 단어의 kind, 새 개체·새 배치
score(model, items) -> float
baselines(items, ckpt0) -> {"random", "untrained"}

# checkpoints.py
save(model, path, meta); load(path) -> (model, meta)
```

### agents
```python
class LLMClient(Protocol):
    def complete_json(self, *, agent, system, user, schema) -> BaseModel

MockClient       # 규칙·템플릿 기반
RouterClient     # agents.yaml 체인 순서로 시도, 전부 실패 시 MockClient

director.propose_cards(state, rng, n=2) -> list[SituationProposal]    # word, kind, 물건 사양, location
narrator.narrate(situation, outcome, lang) -> str                      # 뼈대: 고정 문구
```

Director는 **제안만** 한다. 엔진이 존재·중복·물건 수를 검사해 `CARDS_OFFERED`로 확정한다. 제안이 검증에 실패하면 템플릿 카드로 대체한다. 두 카드는 서로 다른 개체 조합이 되게 하고(색·크기 다양성), 이 다양성이 플레이어 선택의 전략성을 만든다.

### game
```python
class Session:
    new(save_dir, seed); load(save_dir)
    view() -> PlayerView
    draw_cards(); pick_card(i)
    show(slot) -> LearnReport; try_child() -> ChildAct
    praise() -> LearnReport; correct(slot) -> LearnReport; skip()
    end_day() -> QuizResult        # 6개 해결 후. 스냅샷 저장, day+1
```

`show / praise / correct`는 Example을 기록하고 즉시 `learner.learn`을 호출해 `LEARNED` 이벤트를 남긴다. CLI와 이후 FastAPI는 `Session`만 감싼다.

---

## 5. 한 바퀴 흐름 (예시 CLI)

```
$ uv run python -m server.cli new --seed 1
Day 1 · situation 1/6
  Cards:  [1] house · 3 items    [2] yard · 4 items
> pick 1
  Grandma: "Could you bring me the bobo?"   (bobo = cup, Grandma's word)
   [0] red   cylinder small ceramic   (cup)
   [1] blue  flat     big   fabric    (cloth)
   [2] red   flat     small paper     (paper)
> show 0
  You bring the cup. Grandma smiles.
  learned: 30 steps · loss 1.42 → 0.88 · 0.9s
Situation 2/6 · pick 2
  Grandma: "bobo, please."
   [0] green flat     small paper  ...  [1] blue cylinder big ceramic ...
> try
  The child picks [0].  (wrong: that's paper)
> correct 1
  learned: pair (chose 0 → should be 1) · 0.9s
...
> end
  Evening quiz (3): 2/3   random 0.33 · untrained 0.0
  Day 2 ...
> quit      # 저장
$ uv run python -m server.cli load saves/<id>   # 그대로 재개
```

---

## 6. 불변식 = 뼈대 테스트

| 테스트 | 확인 내용 |
|---|---|
| `test_fold_roundtrip` | 저장·재로드한 `fold(events)` == 메모리 상태 |
| `test_idempotent` | 같은 `action_id` 두 번 → 이벤트 1번 |
| `test_rule_errors` | 카드 없이 선택, 상황 없이 Show, 이미 해결된 상황 재해결, 범위 밖 슬롯, 직전 Try 없는 칭찬/교정 |
| `test_child_obs_no_leak` | `Observation` 직렬화에 kind·id·정답 없음. `PlayerView`에만 kind_label |
| `test_mask_allows_all_slots` | 문법 마스크가 장면의 모든 슬롯 허용 |
| `test_wrong_show_recorded_as_is` | 틀린 슬롯을 Show하면 그 슬롯이 target으로 학습 + `was_right=False` |
| `test_skip_no_example` | Skip/피드백 없는 Try는 Example 0개 |
| `test_import_boundary` | `server.learning` 이 `server.agents` 를 import하지 않음 |
| `test_quiz_not_in_train` | 퀴즈 해시 ∩ 학습 해시 = ∅ |
| `test_learn_changes_weights` | 학습 스텝 후 가중치 변화, 해당 Example 손실 하강 |
| `test_ckpt0_immutable` | `ckpt_000.pt` 불변 |
| `test_reset_forgets` | 학습 후 점수 > ckpt_000 점수 |
| `test_director_invalid_fallback` | Director가 잘못된 제안(없는 kind, 중복) → 템플릿 카드로 대체 |
| `test_router_fallback` | 1·2순위 실패 → Mock으로 진행 |

---

## 7. 만드는 순서

| # | 작업 | 예상 | 끝나면 |
|---|---|---|---|
| 1 | `pyproject.toml`, uv venv(3.12), torch·pydantic·pyyaml·pytest | 0.5h | `pytest` 실행됨 |
| 2 | catalog · vocab · tokenizer · situation(뷰 분리) | 1.5h | 상황 → PlayerView / Observation 토큰 |
| 3 | events · state · engine + 테스트 | 2h | 불변식 1~3 통과 |
| 4 | model · policy · learner · checkpoints | 2h | 새 Example 1개로 학습 스텝, 지연 측정 |
| 5 | `scripts/exp_task_a.py` 합성 실험 | 1.5h | **D1 게이트** (아래) |
| 6 | evaluator · 퀴즈 분리 | 1h | 테스트 9~12 |
| 7 | agents (client/router/mock + Director 카드·Narrator 템플릿) | 1h | 테스트 13~14 |
| 8 | session · saves · cli | 2h | 5장 흐름이 터미널에서 돈다 |

**D1 게이트 (5번)**: 합성 플레이어(규칙대로 시범 보이는 스크립트)가 하루 6상황 × 3일 진행.
① 학습 스텝 지연 ≤2초 ② 퀴즈 정답률이 무작위·ckpt_000보다 높음 ③ 시드 3개 모두 ④ **즉시 학습 vs 하루 끝 배치 학습** 비교 — 즉시 학습이 마지막 예시에 쏠리거나(과적합) 이전 지식을 잊으면(망각) replay 비율·스텝 수를 조정하고, 그래도 안 되면 배치 학습으로 후퇴한다.
이 결과가 나쁘면 6~8로 넘어가지 않고 모델·물건 수를 줄여 재실험한다.

## 8. 뼈대 이후 붙일 것 (순서)

1. Meta API (`muse-spark-1.3-contributor`) — Director·Narrator를 `RouterClient` 체인에 추가
2. 탐침 패널(4.6) — evaluator 재사용
3. FastAPI + 웹 UI (D5)
4. 놀이 B(`use` 상황), 상황 플래그, 이야기 연쇄 (D6)
5. 자유문장 입력(Interpreter) — "내가 이렇게 할게" 같은 자연어를 행동 선택으로 변환 (선택 사항)

---

## 9. 구현 상태 (2026-10-26)

단계 1~8 완료, pytest 36개 통과. 설계 대비 달라진 점:

| 항목 | 설계 | 구현 |
|---|---|---|
| 단어 → NAME 토큰 | 엔진이 첫 등장 순서로 부여 | 저장 생성 시 고정(`meta.word_map`). 퀴즈 토큰과 일치시키기 위해 엔진이 카드의 `word_token`을 검증 후 채택 |
| 학습 스텝 수 | 30 | 60 (시드 100~102로 8개 설정 비교 후 선택) |
| 퀴즈 | 저녁 3문제 | 저녁 8문제 + 만난 단어 전체 held-out 풀(최대 80) 점수 병기. 3문제는 너무 거칠어서 |
| Director | 카드 2장 제안 | 덜 본 종류 우선 템플릿 + `RouterClient`(현재 Mock). 제안이 무효/퀴즈 장면과 충돌하면 템플릿으로 대체 |
| Narrator | 고정 문구 | 고정 문구 + `refs_ok()` 참조 검사(LLM 서술 도입 시 사용) |
| 추가 | — | `game/autoplay.py`(합성 플레이어), `python -m server.cli autoplay` |

실행:

```bash
uv run pytest -q
uv run python -m server.cli new --seed 1          # 직접 플레이
uv run python -m server.cli load saves/<id>       # 재개
uv run python -m server.cli autoplay --seed 1 --days 3
uv run python scripts/exp_task_a.py --seeds 0,1,2 --days 10   # D1 게이트 실험
```

알려진 한계: 저장 중 강제 종료 시 예시·체크포인트·LEARNED 이벤트가 어긋날 수 있다(로드 시 경고만 하고 복구는 안 함). 영어 문구만 있음(한국어 토글 미구현).

---

## 10. 온라인 학습 안정화 (2026-10-26)

**문제**: 5단계 실험(시드 0~2)에서 즉시 학습은 후반에 정답률이 떨어지고(최대 −0.06), 3일 차 정답률이 시드마다 크게 달랐다. 그 시드 3개는 운이 좋은 편이었다 — 튜닝 시드 20개로 다시 재면 기존 설정의 3일 차는 평균 0.53(최저 0.30)이다.

**진단**: 3일 차에 약한 시드는 학습 데이터를 거의 다 맞히는데(train acc 0.97~1.0) 새 장면에서만 틀렸다. 단어마다 예시는 7~8개로 고르게 들어갔다. 최적화 실패가 아니라 **외우기(일반화 실패)** 였다.

**바꾼 것** (`config/model.yaml`, `OnlineCfg` 기본값)

| 항목 | 이전 | 이후 | 효과 |
|---|---|---|---|
| lr | 3e-3 | 1e-3 | 후반 하락 제거 |
| weight_decay | 0 | 0.05 | 후반 하락·편차 감소 |
| oversample | 4 | 1 | 새 예시 하나에 과적합하는 것 방지 |
| **redraw** | 없음 | 1.0 | 방해 물건 재추첨 증강: 정답 슬롯은 그대로 두고 나머지 슬롯만 다른 종류로 교체. 3일 차 크게 개선 |
| Director 노출 기준 | 테이블 위 물건 수 | **요청된 횟수** (`state.requests`) | 단어별 예시 수 균형 |

시도했으나 채택하지 않은 것: lr 감쇠(전역/코사인), 조기 종료, 그래디언트 클리핑, replay 8/32, steps 30/120, 층 3개·d_model 96. 단독으로는 후반 안정화에 일부 도움(코사인·감쇠·wd)이 되었지만 lr 1e-3 + wd + oversample 1 조합이 가장 안정적이었고, 3일 차는 위 증강 없이는 하루 상황 수를 10~12로 늘려야 80% 근처였다.

**검증** — 실제 `Session` 파이프라인(Director·엔진·퀴즈 풀·합성 플레이어), 튜닝에 쓰지 않은 **새 시드 200~239(40개)**, 하루 6상황

| | 기존 설정 | 새 설정 |
|---|---|---|
| 3일 차 정답률 (greedy, 평균 / 최저) | 0.55 / 0.30 | **0.90 / 0.53** |
| 3일 차 80% 이상 시드 | 4/40 | **34/40** |
| 4일 차 80% 이상 시드 | 8/40 | 38/40 (최저 0.78) |
| 10일 차 (평균 / 최저) | 0.92 / 0.59 | **1.00 / 1.00** |
| 후반 하락 (최고점 − 최종, 평균) | 0.030 | **0.000** |
| 학습 지연 평균 / 최대 | 0.36 / 0.61s | 0.36 / 0.59s |

(정답률은 만난 단어에 대한 held-out 풀 80문제 기준 argmax 정확도. 시드 16개 병렬 실행이라 지연은 단독 실행보다 길게 측정됨.)

**남은 한계 / 주의**
- 3일 차 80%는 **평균으로는 달성(0.90)**, 시드별로는 34/40. 최저 시드는 0.53으로 3일 차에 못 미친다. 4일 차에는 38/40이 80%를 넘는다.
- 이 숫자는 "정답을 아는 합성 플레이어"가 매번 올바른 시범을 보일 때의 결과다. 사람이 틀린 시범을 주거나 한 종류만 반복하면 낮아진다(의도된 동작).
- **좁게 가르치는 경우 전제가 약해졌다**: 빨간 작은 것만 가르치는 조건(이상화 하네스, 시드 20개)에서 10일 차 정답률이 증강 없이 0.53, 재추첨 증강 후 0.74. 다양하게 가르친 경우(1.00)보다는 여전히 크게 낮지만, 방해 물건 재추첨이 일부 일반화를 만들어 준다. "좁은 데이터 → 거의 못 배움"이라는 대비는 약해졌고, 색 편향 같은 구체적 실패 양상은 탐침 패널(4.6)로 따로 확인해야 한다.
- 재추첨은 정답 슬롯의 속성을 건드리지 않으며(`tests/test_learning.py`), 라벨에서 얻은 정보(정답 슬롯의 모양·재질)만 쓴다. 단어의 뜻을 따로 주입하지 않는다.
- 하이퍼파라미터는 시드 100~119로 고르고 200~239로 한 번만 확인했다(탐색 조합 약 30개).

재현:

```bash
uv run python scripts/exp_stability.py --cond session --seeds 200-239 --jobs 16 --curve \
  --grid "old default|lr=0.003,weight_decay=0,oversample=4,redraw=0" "final|"
```

---

## 11. 아이 머릿속 탐침 (DESIGN 4.6) — 구현·검증 (2026-10-26)

**구현**: `server/learning/probe.py` (+ `policy.slot_probs`, `evaluator.score` 재사용), `Session.mind(compare_day=None)`, CLI `mind [DAY]`.
- 방법: 한 속성만 다른 두 슬롯 + 다른 종류의 방해 슬롯 1개로 된 3슬롯 장면을 만들고, 해당 단어에 대해 실제 모델이 어느 쪽을 고르는지 잰다. 무작위 바탕 30개 × 두 슬롯 순서로 평균 → `pref[속성][값]` (0.5 = 무관심, 1.0 = 늘 그쪽). `pull[속성]` = 가장 센 선호의 0.5와의 차이.
- 모델을 바꾸지 않고, 고정 시드라 같은 모델이면 같은 결과다. `compare_day`를 주면 그날 스냅샷과 나란히 비교한다(전/후 막대).
- 주의: 모양·재질 대비는 카탈로그에 없는 물건(예: 납작한 도자기)을 만들 수 있다. 한 속성만 떼어 보려는 의도적 선택이며 분포에서 조금 벗어난다.
- 탐침 자체의 타당성 테스트(`tests/test_probe.py`): 색 규칙만 가르친 모델은 red 선호 > 0.8, 모양 규칙만 가르친 모델은 cylinder 선호 > 0.8이 나오고, 학습 전 모델은 무관심(pull ≈ 0.04, 노이즈 수준)이다.

**실험: 좁게 가르치면 색 편향이 보이는가** (`scripts/exp_probe.py`, 실제 Session, 새 시드 300~319 20개, 하루 6상황, 합성 플레이어, 날짜별 4단어 평균)

| 조건 | 날 | 퀴즈 정답률 전체 | 빨강·작은 것만 | 그 외 | red 선호 | color pull | shape / material pull |
|---|---|---|---|---|---|---|---|
| 다양하게 | 3 | 0.93 | 0.90 | 0.93 | **0.50** | 0.16 | 0.19 / 0.39 |
| 다양하게 | 6 | 1.00 | 1.00 | 1.00 | 0.51 | 0.15 | 0.19 / 0.41 |
| 좁게 (빨강·작은 것만) | 1 | 0.32 | 0.84 | 0.24 | **0.92** | 0.42 | 0.12 / 0.19 |
| 좁게 | 3 | 0.43 | 0.92 | 0.35 | **0.91** | 0.41 | 0.11 / 0.22 |
| 좁게 | 6 | 0.62 | 0.99 | 0.57 | 0.79 | 0.30 | 0.15 / 0.27 |
| 좁게, 재추첨 증강 끔 | 6 | 0.42 | 0.91 | 0.35 | **0.87** | 0.37 | 0.14 / 0.19 |

- **편향이 실제로 보인다.** 좁게 가르친 아이는 "빨간 것"을 0.79~0.92로 고르고(다양: 0.50 = 무관심), 단어별 가장 센 단서가 color=red인 경우가 80개 중 39개(증강 끔: 60개). 다양하게 가르친 아이는 단서가 거의 항상 material(=종류를 가리는 속성)이다. 색 단서는 단어와 무관하게 모든 단어에서 같이 나타난다.
- 정답률도 같은 이야기를 한다: 빨강·작은 것 문제는 0.84~0.99로 맞히지만, 다른 색·크기는 0.24~0.57. 맞는 이유가 "컵을 알아서"가 아니라 "빨간 걸 고르면 되니까"라는 것이 탐침에서 드러난다.
- 지난 단계의 우려("재추첨이 좁은 데이터의 대비를 약하게 만든다")에 대해: 재추첨은 편향을 지우지 않는다. 증강 있음 6일 차 red 선호 0.79 vs 없음 0.87로 조금 줄이고 정답률을 올릴 뿐(0.62 vs 0.42), 편향 자체는 남는다.
- 학습 전 노이즈 수준: 모든 pull ≈ 0.02~0.04, red 선호 0.50. 다양한 조건의 color pull 0.15~0.24는 노이즈보다 크지만 red 선호가 0.5라 특정 색에 쏠린 것이 아니라 값마다 약한 편차가 있는 정도.

**설계상 주의 (미해결)**: 현재 구현에서 플레이어가 가르치는 다양성의 손잡이는 "카드 선택/건너뛰기"뿐이고, 물건의 색·크기는 Director가 정한다. 위 "좁게" 조건은 실험에서 Director를 강제로 좁게 만든 것이다. DESIGN 1.2의 "플레이어가 무엇을 보여줄지 고른다"를 실제 플레이로 만들려면 Director가 같은 종류의 색/크기를 다르게 낸 카드를 제안하고 플레이어가 고르게 하거나, 플레이어가 물건 속성을 고르는 UI가 필요하다.
