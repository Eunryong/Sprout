/**
 * Sprout Web Frontend Application Logic
 */
const $ = (id) => document.getElementById(id);
const RES = ["solved", "partial", "nothing", "worse"];

// Every word the page itself shows, in both languages. The scene text comes from the server in the language it speaks.
const TEXT = {
  en: {
    thinking: "the child is thinking…", newChild: "New child", newChildTitle: "Start a new child. The current one is kept on disk.",
    guess: "The child’s guess:", r_solved: "solved", r_partial: "partial", r_nothing: "nothing", r_worse: "worse",
    leave: "Leave it to the child", choose: "Choose", babyBtn: "Choose", pickBtn: "Choose", tell: "Tell", show: "Show", showBtn: "Choose", sayBtn: "Choose", soloTry: "Let the child try this", again: "Raise another child", next: "Next scene", nextStep: "Next: Child's solo moment →", nextSolo: "Next: Child's solo moment →",
    step1: "1/2 · Parent demonstrates", step2: "2/2 · Child chooses alone",
    step1Badge: "1/2 · Parent demonstrates", step1SayBadge: "1/2 · Advise the child in words", step2Badge: "2/2 · Child chooses alone",
    leaveTitle: "Let the child try first",
    autoStep1Title: "Let the child try first",
    autoStep1Desc: "Step back and watch without giving an answer. The child decides on its own and learns more deeply from what happens next.",
    autoStep1Btn: "Let the child try alone",
    autoStep1SayTitle: "Let the child try what it thinks",
    autoStep1SayDesc: "Without giving separate advice, let the child try first based on what it currently knows",
    autoStep1SayBtn: "Let the child try alone",
    soloLeaveTitle: "Child chooses on its own",
    autoStep2Title: "Child chooses on its own",
    soloLeaveDesc: "Facing a new situation, the child tries an unfamiliar action on its own and learns from what happens.",
    autoStep2Desc: "Facing a new situation, the child tries an unfamiliar action on its own and learns from what happens.",
    autoStep2Btn: "🌱 Child chooses on its own →",
    babyHint: "The baby cannot choose yet. Show it what to do and it watches what comes of it.",
    hint: "Each scene has two steps from age 3: first <b>Advise the child in words</b> (pick what to tell the child, or let the child try first), then <b>Child chooses alone</b>. There is no success or failure — the child learns what came of each action and decides how to act next time.",
    hintShow: "Choose an action to do with the child. Next, the child will face a new situation alone and learn by trying something unfamiliar.",
    hintSay: "A child over 3 understands the situation and can ask in words. Choose which way to advise the child in words. In the next scene, the child will judge on its own.",
    hintSolo: "Now the child faces a new situation alone. Instead of only repeating what it already knows, watch how the child tries something unfamiliar and learns from the experience.",
    newToChild: "new to the child", leanRemembers: "The child seems drawn to this: it remembers how it went.", leanCurious: "The child seems curious about this.",
    writing: (s) => `Preparing the next choices… ${s}s`, writing0: "Preparing the next choices…",
    scene: (age, n) => `${age} · scene ${n}`,
    srcScenes: "Scenes: ", srcMock: "stand-in (mock mode)", srcDirector: "Director · ", srcTemplate: "template (the Director did not answer)", srcNone: "not asked yet",
    srcWorld: "World: ", srcWords: "Words: ", srcWordsMock: "stand-in text (mock)", srcWordsLive: (ok, n) => `${ok} of ${n} written by the model`, srcWordsFixed: "fixed text",
    grown: (age) => `The child is grown: ${age}`,
    lived: (n, a, b, c) => `${n} moments lived. You showed ${a} times, told ${b} times and left it to the child ${c} times.`,
    remembersTitle: "What the child remembers:",
    whoShow: "👀 You demonstrated it (the child watched and learned)", whoLeave: "🌱 The child chose by itself", whoSolo: "🌱 The child faced the situation and chose alone",
    whoSayFollowed: "💬 The child tried what you advised in words", whoSayIgnored: "🌱 You advised in words, but the child tried its own way",
    whoFollowed: "💬 The child tried what you advised in words", whoOther: "🌱 You advised in words, but the child tried its own way",
    research: "research: ", learnedR: (x) => `research: learned ${x}`, guessR: (r, p) => `research: guess ${r} ${p}`, curiosityR: "curiosity +",
    errScene: "Could not load the scene: ", errPlay: "That did not work: ",
    errReach: "the server is not reachable. Is `python -m server.web` still running? Reload this page after restarting it.",
    raiseNew: "A new child, in a new world", raising: (s) => `A new child is being born and growing to three… ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")} (a few minutes)`, raiseFailed: "Raising the child did not work.", other: "한국어",
    adoptTitle: "Start raising a 3-year-old child (Choose temperament)",
    adoptSub: "Babyhood (0–2 yrs) is skipped. The child starts at age 3 (36 months) with baseline 3-year-old intelligence and common sense.",
    temperaments: [
      { id: "warm", label: "🌱 Warm & Empathetic 3-year-old" },
      { id: "playful", label: "🎈 Curious & Playful 3-year-old" },
      { id: "busy", label: "👟 Independent & Bold 3-year-old" },
      { id: "strict", label: "🧩 Calm & Orderly 3-year-old" }
    ],
    restarting: "Opening the child's own world… this takes a few seconds.", cancel: "Cancel",
    adoptKid: (k) => `${k.name} · ${TEXT.en.persona[k.persona]} · ${k.scenes} moments so far${k.world && k.world.random ? ` · world #${k.world.seed}` : ""}`,
    persona: { warm: "raised by a warm, attentive parent", busy: "raised by a busy parent who often left things to the child", playful: "raised by a playful, curious parent", strict: "raised by a strict, orderly parent" },
    albumTitle: (k) => `${k.name}, three years old: you take over`,
    albumBody: (k) => `For the first three years muse-spark was the parent (${TEXT.en.persona[k.persona]}). It showed ${k.showed} times, told ${k.told} times and left it to the child ${k.left} times. Everything the child learned, it learned from what really happened.`,
    albumRemembers: "What the child remembers from those years:",
    openQaBtn: "💬 Ask the Child",
    customActionLabel: "Or type your own action as a parent",
    customActionPlaceholder: "e.g. Read a dinosaur picture book together / Wash hands playfully with bubbles",
    customActionBtn: "Do this action",
    qaTitle: "Ask the Child (Child Q&A & Growth Check)",
    qaSubtitle: "The child answers in their own voice based on the memories and habits learned so far.",
    qaCloseBtn: "✕ Return to Game",
    qaInputPlaceholder: "Ask the child anything...",
    qaSendBtn: "Ask",
    qaChildBadge: (age) => `Child (${age})`,
    qaUserBadge: "Parent",
    qaThinking: "The child is thinking...",
    cognitiveBadge: (tier, concepts, principles, persona) => persona ? `🧠 Cognitive Scale: ${tier} · Personality: ${persona} · Concepts ${concepts} · Principles ${principles}` : `🧠 Cognitive Scale: ${tier} · Concepts ${concepts} · Principles ${principles}`,
    expansionTitle: "🌱 Knowledge Expansion",
    expansionPrinciple: "Synthesized Principle",
    expansionConcepts: "Expanded Concepts",
    expansionLearned: "What Learned",
  },
  ko: {
    thinking: "아이가 생각하는 중…", newChild: "새 아이", newChildTitle: "새 아이를 시작해요. 지금 아이는 저장된 채로 남아요.",
    guess: "아이의 예상:", r_solved: "해결", r_partial: "조금", r_nothing: "변화 없음", r_worse: "더 나빠짐",
    leave: "아이에게 맡기기", choose: "선택하기", babyBtn: "선택하기", pickBtn: "선택하기", tell: "말해 주기", show: "보여 주기", showBtn: "선택하기", sayBtn: "선택하기", soloTry: "이 행동 해보기", again: "다른 아이 키우기", next: "다음 장면", nextStep: "다음: 아이가 혼자 선택하는 상황으로 →", nextSolo: "다음: 아이가 혼자 선택하는 상황으로 →",
    step1: "1/2 · 부모가 먼저 보여 주는 장면", step2: "2/2 · 아이가 스스로 선택하는 상황",
    step1Badge: "1/2 · 부모가 먼저 보여 주는 장면", step1SayBadge: "1/2 · 아이에게 말로 알려 주는 장면", step2Badge: "2/2 · 아이가 스스로 선택하는 상황",
    leaveTitle: "아이가 먼저 해보게 맡기기",
    autoStep1Title: "아이가 먼저 해보게 맡기기",
    autoStep1Desc: "부모가 먼저 답을 보여주지 않고 지켜봐요. 아이가 스스로 생각해서 행동하고, 그 결과에서 다음엔 어떻게 할지 더 깊이 배워요.",
    autoStep1Btn: "스스로 해보라고 하기",
    autoStep1SayTitle: "아이가 생각한 대로 해보라고 하기",
    autoStep1SayDesc: "별도로 알려 주지 않고 아이가 지금 알고 있는 대로 먼저 해보게 합니다",
    autoStep1SayBtn: "스스로 해보라고 하기",
    soloLeaveTitle: "아이가 스스로 선택하기",
    autoStep2Title: "아이가 스스로 선택하기",
    soloLeaveDesc: "새로운 상황에서 아이가 아직 잘 모르는 행동을 스스로 시도하고 겪으며 배웁니다",
    autoStep2Desc: "새로운 상황에서 아이가 아직 잘 모르는 행동을 스스로 시도하고 겪으며 배웁니다",
    autoStep2Btn: "🌱 아이가 스스로 선택하기 →",
    babyHint: "아기는 아직 고를 수 없어요. 할 일을 보여 주면 아기는 그 결과를 지켜봐요.",
    hint: "세 살부터는 한 장면이 두 단계로 진행돼요. 먼저 <b>아이에게 말로 알려 주는 장면</b>을 겪고, 이어서 <b>아이가 스스로 선택하는 상황</b>을 겪어요. 성공과 실패는 없어요 — 아이는 이 행동을 해서 어떤 결과가 나왔는지 보고 조심하거나 다음엔 어떻게 해 볼지 스스로 배워요.",
    hintShow: "부모가 먼저 함께할 행동을 골라 주세요. 다음 단계에서는 아이가 새로운 상황을 만나 아직 잘 모르는 것을 스스로 시도하며 배웁니다.",
    hintSay: "3살이 넘은 아이는 상황을 이해하고 말로 물어볼 수 있어요. 아이에게 어떤 방법을 말로 알려 줄지 골라 주세요. 다음 장면에서는 아이가 혼자 상황에서 스스로 판단합니다.",
    hintSolo: "이번에는 아이 혼자 새로운 상황을 마주합니다. 이미 배운 좋은 답만 고르지 않고, 아직 잘 모르는 행동을 직접 경험하며 배우는 과정을 지켜보세요.",
    newToChild: "아이에게 처음인 일", leanRemembers: "아이가 끌리는 듯해요: 어땠는지 기억하고 있어요.", leanCurious: "아이가 이걸 궁금해하는 듯해요.",
    writing: (s) => `다음 선택지를 만드는 중… ${s}초`, writing0: "다음 선택지를 만드는 중…",
    scene: (age, n) => `${age} · ${n}번째 장면`,
    srcScenes: "장면: ", srcMock: "대역 (mock 모드)", srcDirector: "감독 · ", srcTemplate: "기본 틀 (감독이 답하지 않았어요)", srcNone: "아직 묻지 않음",
    srcWorld: "세계: ", srcWords: "글: ", srcWordsMock: "대역 글 (mock)", srcWordsLive: (ok, n) => `${n}개 중 ${ok}개를 모델이 썼어요`, srcWordsFixed: "정해진 글",
    grown: (age) => `아이가 자랐어요: ${age}`,
    lived: (n, a, b, c) => `${n}번의 순간을 지났어요. 보여 준 것 ${a}번, 말해 준 것 ${b}번, 아이에게 맡긴 것 ${c}번.`,
    remembersTitle: "아이가 기억하는 것:",
    whoShow: "👀 당신이 직접 보여 줬어요 (아이가 옆에서 보고 배워요)", whoLeave: "🌱 아이가 스스로 선택했어요", whoSolo: "🌱 아이가 혼자 상황에서 스스로 선택했어요",
    whoSayFollowed: "💬 부모님이 말로 알려 준 대로 아이가 직접 해 봤어요", whoSayIgnored: "🌱 부모님이 알려 주었지만 아이가 자기 생각대로 해 봤어요",
    whoFollowed: "💬 부모님이 말로 알려 준 대로 아이가 직접 해 봤어요", whoOther: "🌱 부모님이 알려 주었지만 아이가 자기 생각대로 해 봤어요",
    research: "연구: ", learnedR: (x) => `연구: 배운 것 ${x}`, guessR: (r, p) => `연구: 예상 ${r} ${p}`, curiosityR: "호기심 +",
    errScene: "장면을 불러오지 못했어요: ", errPlay: "잘 되지 않았어요: ",
    errReach: "서버에 닿지 않아요. `python -m server.web`이 아직 켜져 있나요? 다시 켠 뒤 이 페이지를 새로고침하세요.",
    raiseNew: "새 아이 만나기 (새로 만든 세계)", raising: (s) => `새 아이가 태어나 세 살까지 자라는 중… ${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")} (몇 분 걸려요)`, raiseFailed: "아이를 키우는 데 실패했어요.", other: "English",
    adoptTitle: "어떤 성향의 3세 아이와 시작할까요?",
    adoptSub: "학습 요소가 적은 0~2세 아기 과정은 제외하고, 기초 상식과 말문을 갖춘 3세(36개월) 지능의 아이부터 바로 시작합니다.",
    temperaments: [
      { id: "warm", label: "🌱 다정하고 공감을 잘하는 3세 아이" },
      { id: "playful", label: "🎈 호기심 많고 장난기 넘치는 3세 아이" },
      { id: "busy", label: "👟 스스로 해보길 좋아하는 씩씩한 3세 아이" },
      { id: "strict", label: "🧩 차분하고 질서를 좋아하는 3세 아이" }
    ],
    restarting: "아이가 자란 세계를 여는 중입니다… 몇 초 걸려요.", cancel: "취소",
    adoptKid: (k) => `${k.name} · ${TEXT.ko.persona[k.persona]} · 지금까지 ${k.scenes}번의 순간${k.world && k.world.random ? ` · ${k.world.seed}번 세계` : ""}`,
    persona: { warm: "다정하고 세심한 부모가 키운 아이", busy: "바빠서 아이에게 자주 맡긴 부모가 키운 아이", playful: "장난기 많고 이것저것 해 본 부모가 키운 아이", strict: "엄하고 규칙을 중시한 부모가 키운 아이" },
    albumTitle: (k) => `세 살이 된 ${k.name}: 이제 당신이 이어받아요`,
    albumBody: (k) => `처음 3년은 muse-spark가 부모였어요 (${TEXT.ko.persona[k.persona]}). 보여 준 것 ${k.showed}번, 말해 준 것 ${k.told}번, 아이에게 맡긴 것 ${k.left}번. 아이가 배운 건 모두 실제로 겪은 일에서 나왔어요.`,
    albumRemembers: "그 3년에서 아이가 기억하는 것:",
    openQaBtn: "💬 아이와 대화하기 (마무리/질문)",
    customActionLabel: "또는 부모로서 직접 해주고 싶은 행동을 적어보세요",
    customActionPlaceholder: "예: 공룡 동화책을 읽어주며 주인공 이야기를 나눈다 / 거품으로 장난치며 손을 씻겨준다",
    customActionBtn: "직접 행동하기",
    qaTitle: "아이에게 질문하기 (Child Q&A & Growth Check)",
    qaSubtitle: "지금까지 겪은 경험과 학습한 기억을 바탕으로 아이가 직접 대답해요.",
    qaCloseBtn: "✕ 게임으로 돌아가기",
    qaInputPlaceholder: "아이에게 물어보고 싶은 말을 적어보세요...",
    qaSendBtn: "질문하기",
    qaChildBadge: (age) => `아이 (${age})`,
    qaUserBadge: "부모",
    qaThinking: "아이가 생각하는 중...",
    cognitiveBadge: (tier, concepts, principles, persona) => persona ? `🧠 지능 스케일: ${tier} · 성격: ${persona} · 개념 ${concepts} · 원리 ${principles}` : `🧠 지능 스케일: ${tier} · 개념 ${concepts} · 원리 ${principles}`,
    expansionTitle: "🌱 지식 확장 (Knowledge Expansion)",
    expansionPrinciple: "도출된 원리",
    expansionConcepts: "확장된 개념",
    expansionLearned: "학습한 내용",
  },
};
let LANG = "en";
try { LANG = localStorage.getItem("lang") || window.SERVER_LANG || "en"; } catch (_) { LANG = window.SERVER_LANG || "en"; }
if (!TEXT[LANG]) LANG = "en";
const t = (k, ...a) => { const v = TEXT[LANG][k]; return typeof v === "function" ? v(...a) : v; };

function applyText() {
  document.documentElement.lang = LANG;
  document.querySelectorAll("[data-t]").forEach((e) => { e.textContent = t(e.dataset.t); });
  document.querySelectorAll("[data-html]").forEach((e) => { e.innerHTML = t(e.dataset.html); });
  document.querySelectorAll("[data-title]").forEach((e) => { e.title = t(e.dataset.title); });
  document.querySelectorAll("[data-placeholder]").forEach((e) => { e.placeholder = t(e.dataset.placeholder); });
  document.querySelectorAll(".qa-chip").forEach((btn) => {
    btn.textContent = LANG === "ko" ? btn.dataset.qKo : btn.dataset.qEn;
  });
  $("langBtn").textContent = t("other");
  if (currentCognitive) {
    updateCognitiveBadge(currentCognitive);
  }
}

let busy = false;
let qaBusy = false;
let currentAgeText = "";
let currentCognitive = null;

function formatCognitiveTier(c, lang) {
  if (!c) return "";
  const scale = c.scale_b != null ? `${c.scale_b}B` : "1B";
  let tierDesc = "";
  if (c.tier_name) {
    const match = c.tier_name.match(/\((.*?)\)/);
    if (match) {
      tierDesc = match[1];
    } else {
      tierDesc = c.tier_name.replace(/^[0-9.]+B\s*·?\s*/, "");
    }
  }
  if (!tierDesc) {
    const s = c.scale_b || 1.0;
    if (s < 2.0) tierDesc = lang === "ko" ? "3세 · 단편 직관" : "Age 3 · Simple Intuition";
    else if (s < 3.5) tierDesc = lang === "ko" ? "4세 · 개념 형성" : "Age 4 · Concept Formation";
    else if (s < 5.0) tierDesc = lang === "ko" ? "5세 · 인과 연결" : "Age 5 · Causal Linking";
    else if (s < 6.2) tierDesc = lang === "ko" ? "6세 · 경험 결합" : "Age 6 · Experience Synthesis";
    else tierDesc = lang === "ko" ? "7세 · 원리 일반화" : "Age 7 · Principle Generalization";
  }
  return `${scale} (${tierDesc})`;
}

function updateCognitiveBadge(c) {
  const badge = $("cognitiveBadge");
  if (!badge) return;
  if (!c) {
    badge.hidden = true;
    return;
  }
  currentCognitive = c;
  const tier = formatCognitiveTier(c, LANG);
  const nConcepts = c.n_concepts ?? 0;
  const nPrinciples = c.n_principles ?? 0;
  const personaLabel = (c.personality && c.personality.label) ? c.personality.label : "";
  badge.textContent = t("cognitiveBadge", tier, nConcepts, nPrinciples, personaLabel);

  const recents = Array.isArray(c.recent_principles) && c.recent_principles.length > 0 ? c.recent_principles : [];
  const hops = c.reasoning_hops != null ? (LANG === "ko" ? `추론 홉: ${c.reasoning_hops}` : `Reasoning Hops: ${c.reasoning_hops}`) : "";
  const episodes = c.n_episodes != null ? (LANG === "ko" ? `에피소드: ${c.n_episodes}` : `Episodes: ${c.n_episodes}`) : "";
  const details = [hops, episodes].filter(Boolean).join(" · ");
  let title = badge.textContent;
  if (details) title += `\n(${details})`;
  if (c.personality && Array.isArray(c.personality.traits) && c.personality.traits.length > 0) {
    title += (LANG === "ko" ? `\n\n[성격 특성]:\n- ` : `\n\n[Personality Traits]:\n- `) + c.personality.traits.join("\n- ");
  }
  if (recents.length > 0) {
    title += (LANG === "ko" ? `\n\n[최근 합성된 원리]:\n- ` : `\n\n[Recent Principles]:\n- `) + recents.join("\n- ");
  }
  badge.title = title;
  badge.hidden = false;
}

function setPhase(phase) {
  const box = $("customActionBox");
  if (!box) return;
  const isScene = phase === "scene";
  box.hidden = !isScene;
  const input = $("customActionInput");
  const btn = $("customActionBtn");
  if (input) input.disabled = !isScene;
  if (btn) btn.disabled = !isScene;
}

function setBusy(on) {
  busy = on;
  $("busy").hidden = !on;
  document.querySelectorAll("button").forEach((b) => {
    if (b.id !== "qaSendBtn" && b.id !== "qaCloseBtn" && !b.classList.contains("qa-chip")) {
      b.disabled = on;
    }
  });
  if (on) {
    const customIn = $("customActionInput");
    if (customIn) customIn.disabled = true;
    const customBtn = $("customActionBtn");
    if (customBtn) customBtn.disabled = true;
  }
}
function showErr(msg) { const e = $("err"); e.textContent = msg || ""; e.hidden = !msg; }

async function api(path, body) {
  let r;
  try {
    r = await fetch(path, body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch (_) {
    throw new Error(t("errReach"));
  }
  if (!r.ok) {
    let m = r.statusText;
    try { m = (await r.json()).detail || m; } catch (_) {}
    throw new Error(m);
  }
  return r.json();
}

function showSource(info) {
  const el = $("source");
  const mock = info.mode === "mock";
  const d = info.director.source || t("srcNone");
  const live = !mock && d !== "template" && d !== t("srcNone") && info.director.source;
  const n = info.narrator;
  el.replaceChildren();
  const add = (text, cls) => { const s = document.createElement(cls ? "b" : "span"); if (cls === "warn") s.className = "warn"; s.textContent = text; el.append(s); };
  el.append(t("srcScenes"));
  if (mock) add(t("srcMock"), "warn"); else add(live ? t("srcDirector") + d : info.director.source === "template" ? t("srcTemplate") : d, live ? "b" : "warn");
  el.append(` · ${t("srcWorld")}${info.world.replace("director:", t("srcDirector"))} · ${t("srcWords")}${mock ? t("srcWordsMock") : n.calls ? t("srcWordsLive", n.ok, n.calls) : t("srcWordsFixed")}`);
  el.hidden = false;
}

function pct(x) { return Math.round(x * 100) + "%"; }

const RESEARCH = typeof location !== "undefined" && new URLSearchParams(location.search).has("research");
if (RESEARCH && typeof document !== "undefined") document.body.classList.add("research");

let CAN_ACT = true;

function optionCard(o, sc) {
  const li = document.createElement("li");
  li.className = "opt";
  const isResearch = typeof document !== "undefined" && document.body.classList.contains("research");
  const lived = isResearch && o.seen === 0 ? t("newToChild") : "";
  const lean = isResearch && (o.lean === "remembers" ? t("leanRemembers") : o.lean === "curious" ? t("leanCurious") : "");
  let research = "";
  if (isResearch) {
    const top = RES.reduce((a, b) => (o.expects[a] >= o.expects[b] ? a : b));
    const bars = RES.map((r) => `<span class="${r}" style="width:${(o.expects[r] * 100).toFixed(1)}%" title="${r} ${pct(o.expects[r])}"></span>`).join("");
    research = `<div class="bar" role="img">${bars}</div><div class="meta"><span>${t("guessR", top, pct(o.expects[top]))}</span><span>${t("curiosityR")}${o.pull.toFixed(2)}</span></div>`;
  }
  const isStep1 = sc && sc.step_total === 2 && sc.step === 1;
  const isStep2 = sc && sc.step === 2;
  const canSpeak = Boolean(sc && sc.can_act && sc.age_months >= 36);

  if (isStep2) {
    li.className = "opt thought-opt";
    const thoughtText = o.speech || (o.text.charAt(0).toUpperCase() + o.text.slice(1));
    const leanHtml = isResearch && lean ? '<p class="lean"></p>' : "";
    const memHtml = isResearch && o.memory ? '<p class="memory"></p>' : "";
    const livedHtml = isResearch && lived ? `<span>${lived}</span>` : "";
    li.innerHTML = `<h3></h3>${leanHtml}${memHtml}${research}
      <div class="foot"><div class="meta">${livedHtml}</div></div>`;
    li.querySelector("h3").textContent = thoughtText;
    if (isResearch && lean) li.querySelector(".lean").textContent = lean;
    if (isResearch && o.memory) li.querySelector(".memory").textContent = o.memory;
    return li;
  }

  const isBaby = !sc || !sc.can_act;
  const part = isStep1 ? "show" : (canSpeak ? "say" : "show");
  const btnLabel = isStep1 ? t("showBtn") : (isBaby ? t("babyBtn") : (canSpeak ? t("sayBtn") : t("choose")));
  const btns = `<button class="primary" data-part="${part}">${btnLabel}</button>`;
  const leanHtml = isResearch && lean ? '<p class="lean"></p>' : "";
  const memHtml = isResearch && o.memory ? '<p class="memory"></p>' : "";
  const livedHtml = isResearch && lived ? `<span>${lived}</span>` : "";
  const mainLine = o.text ? (o.text.charAt(0).toUpperCase() + o.text.slice(1)) : "";
  li.innerHTML = `<h3></h3>${leanHtml}${memHtml}${research}
    <div class="foot"><div class="meta">${livedHtml}</div>
    <div class="row">${btns}</div></div>`;
  li.querySelector("h3").textContent = mainLine;
  if (isResearch && lean) li.querySelector(".lean").textContent = lean;
  if (isResearch && o.memory) li.querySelector(".memory").textContent = o.memory;
  li.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => play(b.dataset.part, o.index)));
  return li;
}

function autonomousCard(sc) {
  const li = document.createElement("li");
  const isStep2 = sc && sc.step === 2;
  const canSpeak = Boolean(sc && sc.can_act && sc.age_months >= 36);
  li.className = isStep2 ? "opt solo-opt" : "opt auto-opt";
  const title = isStep2 ? t("autoStep2Title") : canSpeak ? t("autoStep1SayTitle") : t("autoStep1Title");
  const desc = isStep2 ? t("autoStep2Desc") : canSpeak ? t("autoStep1SayDesc") : t("autoStep1Desc");
  const btnLabel = isStep2 ? t("autoStep2Btn") : canSpeak ? t("autoStep1SayBtn") : t("autoStep1Btn");
  li.innerHTML = `<h3></h3><p class="auto-desc"></p>
    <div class="foot"><div class="meta"></div><div class="row"><button class="primary" data-part="leave"></button></div></div>`;
  li.querySelector("h3").textContent = title;
  li.querySelector(".auto-desc").textContent = desc;
  const btn = li.querySelector("button");
  btn.textContent = btnLabel;
  btn.addEventListener("click", () => play("leave", null));
  return li;
}

function stageBanner(sc) {
  const b = $("banner");
  b.hidden = !(sc.stage_first && sc.stage);
  if (!b.hidden) b.replaceChildren(Object.assign(document.createElement("h2"), { textContent: `${sc.stage.name} · ${sc.age}` }), Object.assign(document.createElement("p"), { textContent: sc.stage.blurb }));
}

async function showAlbum(info) {
  $("album").hidden = true;
  return;
}

async function chooseChild() {
  return new Promise((resolve) => {
    const box = $("adopt");
    const go = async (persona) => {
      box.hidden = true;
      setBusy(true);
      try {
        const r = await api("/api/new", { persona });
        if (r && r.restarting) {
          box.replaceChildren(Object.assign(document.createElement("h2"), { textContent: t("restarting") }));
          box.hidden = false;
          for (let i = 0; i < 60; i++) {
            await new Promise((ok) => setTimeout(ok, 1000));
            const up = await fetch("/api/info").then((x) => x.ok).catch(() => false);
            if (up && i > 1) { location.reload(); return; }
          }
        }
      } catch (e) { showErr(e.message); }
      setBusy(false);
      resolve(true);
    };
    const btn = (label, fn, cls) => { const b = Object.assign(document.createElement("button"), { textContent: label, className: cls || "" }); b.style.margin = "4px 6px 0 0"; b.addEventListener("click", fn); return b; };
    const items = (TEXT[LANG] || TEXT.ko).temperaments || [];
    box.replaceChildren(
      Object.assign(document.createElement("h2"), { textContent: t("adoptTitle") }),
      Object.assign(document.createElement("p"), { textContent: t("adoptSub") }),
      ...items.map((item) => btn(item.label, () => go(item.id), "primary")),
      btn(t("cancel"), () => { box.hidden = true; resolve(false); }, "quiet")
    );
    box.hidden = false;
  });
}

async function showEnd() {
  const s = await api("/api/summary");
  const life = $("life");
  life.replaceChildren();
  const add = (tag, text) => { const e = document.createElement(tag); e.textContent = text; life.appendChild(e); return e; };
  add("h2", t("grown", s.age));
  add("p", t("lived", s.scenes, s.took_part.showed, s.took_part.told, s.took_part.left));
  add("p", s.stages.map((x) => `${x.name}: ${x.scenes}`).join("  ·  "));
  add("p", t("remembersTitle"));
  const ul = document.createElement("ul");
  s.moments.forEach((m) => ul.appendChild(Object.assign(document.createElement("li"), { textContent: m.charAt(0).toUpperCase() + m.slice(1) })));
  life.appendChild(ul);
  $("story").hidden = true; $("banner").hidden = true;
  $("choose").hidden = true; $("after").hidden = true; $("end").hidden = false;
  setPhase("done");
  $("qaCard").hidden = false;
}

async function loadScene(opts = {}) {
  showErr("");
  setBusy(true);
  const keep = Boolean(opts && opts.keepScreen);
  if (!keep) lastResult = null;
  let tick = null;
  if (!keep) {
    $("story").hidden = false; $("end").hidden = true;
    const t0 = Date.now();
    tick = setInterval(() => { $("story").textContent = t("writing", Math.round((Date.now() - t0) / 1000)); }, 500);
    $("story").textContent = t("writing0");
    $("choose").hidden = true; $("after").hidden = true;
  }
  try {
    const [sc, info] = await Promise.all([api("/api/scene"), api("/api/info")]);
    currentAgeText = sc.age;
    const cog = sc.cognitive || sc.cognitive_status || (info && info.cognitive);
    if (cog) updateCognitiveBadge(cog);
    if (Array.isArray(info.qa_history)) renderQaHistory(info.qa_history);
    try {
      if (sessionStorage.getItem("qaOpen") === "1" && info.qa_history && info.qa_history.length > 0) $("qaCard").hidden = false;
    } catch (_) {}
    $("sceneNo").textContent = sc.ended ? "" : t("scene", sc.age, sc.number);
    const chip = $("stageChip"); chip.hidden = !sc.stage; if (sc.stage) chip.textContent = sc.stage.name;
    const stepChip = $("stepChip");
    stepChip.hidden = sc.ended || !(sc.step_total && sc.step_total > 1);
    if (!stepChip.hidden) stepChip.textContent = sc.step === 2 ? t("step2Badge") : (sc.age_months >= 36 ? t("step1SayBadge") : t("step1Badge"));
    const p = $("premise"); p.textContent = info.premise; p.hidden = !info.premise;
    if (RESEARCH) showSource(info);
    await showAlbum(info);
    if (sc.ended) { if (tick) clearInterval(tick); await showEnd(); setBusy(false); return; }
    CAN_ACT = sc.can_act;
    stageBanner(sc);
    $("story").textContent = sc.narration;
    const cards = sc.options.map((o) => optionCard(o, sc));
    if (CAN_ACT && sc.step !== 2) cards.push(autonomousCard(sc));
    const ul = $("opts"); ul.replaceChildren(...cards);
    $("legend").hidden = !RESEARCH;
    $("leave").hidden = true; $("hint").hidden = !CAN_ACT; $("babyHint").hidden = CAN_ACT;
    if (CAN_ACT) {
      if (sc.step_total === 2 && sc.step === 1) $("hint").innerHTML = sc.age_months >= 36 ? t("hintSay") : t("hintShow");
      else if (sc.step === 2) $("hint").innerHTML = t("hintSolo");
      else $("hint").innerHTML = sc.age_months >= 36 ? t("hintSay") : t("hint");
    }
    $("choose").hidden = false; $("after").hidden = true;
    setPhase("scene");
    if (sc.step === 2) {
      if (tick) { clearInterval(tick); tick = null; }
      await play("leave", null, { autoStep2: true });
    }
  } catch (e) { showErr(t("errScene") + e.message); }
  if (tick) clearInterval(tick);
  setBusy(false);
}

function resultCard(r) {
  const who = r.part === "show" ? t("whoShow")
    : (r.part === "solo" || r.step === 2) ? t("whoSolo")
    : r.part === "say" ? (r.followed ? t("whoSayFollowed") : t("whoSayIgnored"))
    : r.part === "leave" ? t("whoLeave")
    : r.followed ? t("whoFollowed") : t("whoOther");
  const el = document.createElement("div");
  const add = (cls, text) => { const p = document.createElement("p"); if (cls) p.className = cls; p.textContent = text; el.appendChild(p); };
  add("who", who);
  if (r.child_said) add("child-said", `💬 ${r.child_said}`);
  add("", r.narration || r.happened || "");
  if (r.world_says) add("world", r.world_says);
  if (r.explanation && RESEARCH) add("surprise", t("research") + r.explanation);
  if (r.remembers) add("learned", r.remembers);
  else if (!RESEARCH && r.learned) add("learned", r.learned);
  if (RESEARCH) add("learned", t("learnedR", r.learned));

  // Knowledge Expansion box (synthesized_principle or expanded_concepts)
  const synthPrinciple = r.synthesized_principle || (r.turn && r.turn.synthesized_principle);
  const expConcepts = r.expanded_concepts || r.new_concepts || (r.turn && (r.turn.expanded_concepts || r.turn.new_concepts));
  const hasSynth = Boolean(synthPrinciple && (typeof synthPrinciple === "string" ? synthPrinciple.trim() : true));
  const hasExp = Boolean(Array.isArray(expConcepts) ? expConcepts.length > 0 : expConcepts);

  if (hasSynth || hasExp) {
    const box = document.createElement("div");
    box.className = "knowledge-expansion";

    const title = document.createElement("div");
    title.className = "expansion-title";
    title.textContent = t("expansionTitle");
    box.appendChild(title);

    if (hasSynth) {
      const prinText = typeof synthPrinciple === "object" ? (synthPrinciple.principle || synthPrinciple.text || JSON.stringify(synthPrinciple)) : String(synthPrinciple);
      const row = document.createElement("div");
      row.className = "expansion-item";
      const lbl = document.createElement("span");
      lbl.className = "expansion-label";
      lbl.textContent = `${t("expansionPrinciple")}:`;
      const val = document.createElement("span");
      val.className = "expansion-text";
      val.textContent = ` ${prinText}`;
      row.appendChild(lbl);
      row.appendChild(val);
      box.appendChild(row);
    }

    if (hasExp) {
      const conceptList = (Array.isArray(expConcepts) ? expConcepts : [expConcepts])
        .map((c) => (typeof c === "object" ? (c.name || c.concept || JSON.stringify(c)) : String(c).trim()))
        .filter(Boolean);
      if (conceptList.length > 0) {
        const row = document.createElement("div");
        row.className = "expansion-item";
        const label = document.createElement("span");
        label.className = "expansion-label";
        label.textContent = `${t("expansionConcepts")}: `;
        row.appendChild(label);

        const tags = document.createElement("span");
        tags.className = "concept-tags";
        conceptList.forEach((c) => {
          const tag = document.createElement("span");
          tag.className = "concept-tag";
          tag.textContent = c;
          tags.appendChild(tag);
        });
        row.appendChild(tags);
        box.appendChild(row);
      }
    }

    const whatLearned = r.what_learned || (r.turn && r.turn.what_learned) || r.remembers || r.learned;
    if (whatLearned) {
      const row = document.createElement("div");
      row.className = "expansion-item";
      const lbl = document.createElement("span");
      lbl.className = "expansion-label";
      lbl.textContent = `${t("expansionLearned")}:`;
      const val = document.createElement("span");
      val.className = "expansion-text";
      val.textContent = ` ${whatLearned}`;
      row.appendChild(lbl);
      row.appendChild(val);
      box.appendChild(row);
    }

    el.appendChild(box);
  }

  return el;
}

let lastResult = null;

async function play(partOrIndex, optionOrCustom, opts = {}) {
  let part = "say";
  let option = 0;
  let custom_text = null;
  if (typeof partOrIndex === "number") {
    option = partOrIndex;
    if (typeof optionOrCustom === "string") {
      custom_text = optionOrCustom;
    }
  } else {
    part = partOrIndex;
    option = optionOrCustom;
    if (opts && opts.custom_text) custom_text = opts.custom_text;
  }
  if (busy && !opts.autoStep2) return;
  showErr("");
  setBusy(true);
  setPhase("outcome");
  try {
    const payload = {
      part: part,
      option: option ?? null,
      option_index: typeof option === "number" ? option : 0,
    };
    if (custom_text) {
      payload.custom_text = custom_text;
    }
    const r = await api("/api/play", payload);
    lastResult = r;
    const cog = r.cognitive || r.cognitive_status;
    if (cog) updateCognitiveBadge(cog);
    $("result").replaceChildren(...resultCard(r).children);
    if (opts.autoStep2 || r.step === 2) {
      $("choose").hidden = false;
      const optsEl = $("opts");
      if (optsEl && r.acted != null && optsEl.children[r.acted]) {
        optsEl.children[r.acted].classList.add("chosen-opt");
      }
      $("next").textContent = t("next");
    } else {
      $("choose").hidden = true;
      $("next").textContent = r.has_next_step ? t("nextStep") : t("next");
    }
    $("after").hidden = false;
    setBusy(false);
    $("next").focus();
  } catch (e) { setBusy(false); showErr(t("errPlay") + e.message); }
}

async function chooseLang(lang) {
  LANG = lang;
  try { localStorage.setItem("lang", lang); } catch (_) {}
  applyText();
  setBusy(true);
  try { await api("/api/lang", { lang }); } catch (e) { showErr(e.message); }
  setBusy(false);
}

function clearInputCleanly(input) {
  if (!input) return;
  const hadFocus = document.activeElement === input;
  input.blur();
  input.value = "";
  if (hadFocus) input.focus();
  setTimeout(() => { input.value = ""; }, 0);
  requestAnimationFrame(() => { input.value = ""; });
}

function readAndClearInput(input) {
  if (!input) return "";
  const hadFocus = document.activeElement === input;
  input.blur();
  const text = input.value.trim();
  input.value = "";
  if (hadFocus) input.focus();
  setTimeout(() => { input.value = ""; }, 0);
  requestAnimationFrame(() => { input.value = ""; });
  return text;
}

function doCustomAction(customText) {
  const input = $("customActionInput");
  let text = typeof customText === "string" ? customText.trim() : "";
  if (text) {
    clearInputCleanly(input);
  } else {
    text = readAndClearInput(input);
  }
  if (!text) return;
  play(0, text);
}

function openQa() {
  const card = $("qaCard");
  card.hidden = false;
  try { sessionStorage.setItem("qaOpen", "1"); } catch (_) {}
  card.scrollIntoView({ behavior: "smooth", block: "start" });
  const input = $("qaInput");
  if (input) input.focus();
}

function renderQaHistory(history) {
  if (qaBusy) return;
  const msgs = $("qaMessages");
  if (!msgs || !Array.isArray(history)) return;
  msgs.replaceChildren();
  for (const item of history) {
    if (!item) continue;
    const userMsg = document.createElement("div");
    userMsg.className = "qa-msg user";
    userMsg.innerHTML = `<span class="badge">${t("qaUserBadge")}</span><div class="qa-bubble-text"></div>`;
    userMsg.querySelector(".qa-bubble-text").textContent = item.question || "";
    msgs.appendChild(userMsg);

    const childMsg = document.createElement("div");
    childMsg.className = "qa-msg child";
    let ageBadge = item.age || currentAgeText || (LANG === "ko" ? "성장 중" : "growing");
    if (currentCognitive && currentCognitive.personality && currentCognitive.personality.label) {
      ageBadge += ` · ${currentCognitive.personality.label}`;
    }
    childMsg.innerHTML = `<span class="badge">${t("qaChildBadge", ageBadge)}</span><div class="qa-bubble-text"></div>`;
    childMsg.querySelector(".qa-bubble-text").textContent = item.answer || "";
    msgs.appendChild(childMsg);
  }
  msgs.scrollTop = msgs.scrollHeight;
}

async function sendQa(qText) {
  if (qaBusy) return;
  const input = $("qaInput");
  let text = typeof qText === "string" ? qText.trim() : "";
  if (text) {
    clearInputCleanly(input);
  } else {
    text = readAndClearInput(input);
  }
  if (!text) return;
  qaBusy = true;
  const sendBtn = $("qaSendBtn");
  if (sendBtn) sendBtn.disabled = true;

  const msgs = $("qaMessages");
  const userMsg = document.createElement("div");
  userMsg.className = "qa-msg user";
  userMsg.innerHTML = `<span class="badge">${t("qaUserBadge")}</span><div class="qa-bubble-text"></div>`;
  userMsg.querySelector(".qa-bubble-text").textContent = text;
  msgs.appendChild(userMsg);

  const pendingMsg = document.createElement("div");
  pendingMsg.className = "qa-msg child";
  let pendingAge = currentAgeText || (LANG === "ko" ? "성장 중" : "growing");
  if (currentCognitive && currentCognitive.personality && currentCognitive.personality.label) {
    pendingAge += ` · ${currentCognitive.personality.label}`;
  }
  pendingMsg.innerHTML = `<span class="badge">${t("qaChildBadge", pendingAge)}</span><div class="qa-bubble-text" style="font-style:italic">${t("qaThinking")}</div>`;
  msgs.appendChild(pendingMsg);
  msgs.scrollTop = msgs.scrollHeight;

  try {
    const res = await api("/api/ask", { question: text });
    let ageLabel = res.age || currentAgeText || (LANG === "ko" ? "성장 중" : "growing");
    if (currentCognitive && currentCognitive.personality && currentCognitive.personality.label) {
      ageLabel += ` · ${currentCognitive.personality.label}`;
    }
    pendingMsg.querySelector(".badge").textContent = t("qaChildBadge", ageLabel);
    const bodyEl = pendingMsg.querySelector(".qa-bubble-text");
    bodyEl.style.fontStyle = "normal";
    bodyEl.textContent = res.answer || "...";
  } catch (err) {
    pendingMsg.querySelector(".qa-bubble-text").textContent = t("errPlay") + err.message;
  } finally {
    qaBusy = false;
    if (sendBtn) sendBtn.disabled = false;
    msgs.scrollTop = msgs.scrollHeight;
  }
}

function initEventHandlers() {
  if ($("leave")) $("leave").addEventListener("click", () => play("leave", null));
  if ($("next")) $("next").addEventListener("click", () => loadScene());
  if ($("again")) $("again").addEventListener("click", () => $("newGame").click());
  if ($("langBtn")) {
    $("langBtn").addEventListener("click", async () => {
      if (busy) return;
      const resting = !$("after").hidden;
      await chooseLang(LANG === "ko" ? "en" : "ko");
      if (!resting) {
        await loadScene({ keepScreen: true });
      } else if (lastResult) {
        $("result").replaceChildren(...resultCard(lastResult).children);
        $("next").textContent = lastResult.has_next_step ? t("nextStep") : t("next");
      }
    });
  }
  if ($("newGame")) {
    $("newGame").addEventListener("click", async () => {
      if (busy) return;
      let went = false;
      try { went = await chooseChild(); } catch (e) { showErr(e.message); }
      if (went) {
        lastResult = null;
        currentCognitive = null;
        updateCognitiveBadge(null);
        if ($("qaMessages")) $("qaMessages").replaceChildren();
        if ($("qaCard")) $("qaCard").hidden = true;
        try { sessionStorage.removeItem("qaOpen"); } catch (_) {}
        loadScene();
      }
    });
  }
  if ($("customActionBtn")) $("customActionBtn").addEventListener("click", () => doCustomAction());
  if ($("customActionInput")) {
    $("customActionInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (e.isComposing || e.keyCode === 229) {
          const val = readAndClearInput(e.target);
          if (val) doCustomAction(val);
          return;
        }
        doCustomAction();
      }
    });
  }
  if ($("openQaBtn")) $("openQaBtn").addEventListener("click", openQa);
  if ($("summaryQaBtn")) $("summaryQaBtn").addEventListener("click", openQa);
  if ($("qaCloseBtn")) {
    $("qaCloseBtn").addEventListener("click", () => {
      $("qaCard").hidden = true;
      try { sessionStorage.setItem("qaOpen", "0"); } catch (_) {}
      const target = !$("choose").hidden ? $("choose") : !$("after").hidden ? $("after") : $("story");
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }
  document.querySelectorAll(".qa-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = LANG === "ko" ? btn.dataset.qKo : btn.dataset.qEn;
      const input = $("qaInput");
      if (input) input.value = q;
      sendQa(q);
    });
  });
  if ($("qaSendBtn")) $("qaSendBtn").addEventListener("click", () => sendQa());
  if ($("qaInput")) {
    $("qaInput").addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        if (qaBusy) return;
        if (e.isComposing || e.keyCode === 229) {
          const val = readAndClearInput(e.target);
          if (val) sendQa(val);
          return;
        }
        sendQa();
      }
    });
  }
}

if (typeof window !== "undefined") {
  window.formatCognitiveTier = formatCognitiveTier;
  window.updateCognitiveBadge = updateCognitiveBadge;
  window.resultCard = resultCard;
}
