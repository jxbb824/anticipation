/*
  Standalone UI logic.
  - Uses fixed seeded shuffle to partition 40 questions into 8 sets of 5.
  - Options within each question are shuffled per-load.
  - Exports CSV at the end; no uploads.
  - Audio files are local in ./audio/q_xxx/.
*/

// Minimal seeded PRNG (Mulberry32)
function mulberry32(seed) {
  let t = seed >>> 0;
  return function () {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle(array, rng) {
  const a = array.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// Build global 40-question order with fixed seed
const GLOBAL_SEED = 123456;
const rngGlobal = mulberry32(GLOBAL_SEED);
const GLOBAL_ORDER = seededShuffle(Array.from({ length: 40 }, (_, i) => i), rngGlobal);

// Partition into 8 sets of 5
const QUESTION_SETS = Array.from({ length: 8 }, (_, s) => GLOBAL_ORDER.slice(s * 5, s * 5 + 5));

// Quality question configuration (global index among 0..39)
const QUALITY_QUESTION_GLOBAL_INDEX = 29; // configurable marker question

// State
let participantSetIdx = 0; // 0..7
let participantId = null;  // original participant code
// questions is an array of objects: { q: number, isQuality: boolean }
let questions = [];
// answers is an array aligned with questions: answers[i] = { seedShuffleOrder: number[6], scores: (number|null)[6] }
let answers = [];
let currentIdx = 0;        // 0..(len-1)

const el = (id) => document.getElementById(id);

function optionRowHtml(optionId) {
  return `
    <div class="option-item">
      <div><strong>Option ${optionId + 1}</strong></div>
      <audio id="optAudio_${optionId}" controls preload="metadata"></audio>
      <div class="scores">
        <label>Score:</label>
        ${[1,2,3,4,5].map(v => `
          <label><input type="radio" name="score_${optionId}" value="${v}"> ${v}</label>
        `).join('')}
      </div>
    </div>
  `;
}

function loadQuestion(idx) {
  const qEntry = questions[idx];
  const q = qEntry.q;
  el('progressText').textContent = `Question ${idx + 1} / ${questions.length}`;

  // Test audio
  const testAudio = el('testAudio');
  const testPath = `./audio/q_${String(q).padStart(3, '0')}/test.mp3`;
  testAudio.src = testPath;

  // Options shuffled each time we first encounter the question
  if (!answers[idx]) {
    const rng = mulberry32(1000 + q); // stable per question
    answers[idx] = {
      seedShuffleOrder: seededShuffle([0,1,2,3,4,5], rng),
      scores: [null, null, null, null, null, null],
    };
  }

  const container = el('optionsContainer');
  container.innerHTML = '';

  const order = answers[idx].seedShuffleOrder;
  order.forEach((optPos, renderedIdx) => {
    // renderedIdx is 0..5 visual slot; optPos is original 0..5
    container.insertAdjacentHTML('beforeend', optionRowHtml(renderedIdx));
    const audioEl = el(`optAudio_${renderedIdx}`);
    if (qEntry.isQuality && optPos === 0) {
      // Replace rank-1 option with reference (identical to test)
      audioEl.src = testPath;
    } else {
      audioEl.src = `./audio/q_${String(q).padStart(3, '0')}/opt_${optPos}.mp3`;
    }

    const prevScore = answers[idx].scores[optPos];
    if (prevScore) {
      const radios = document.getElementsByName(`score_${renderedIdx}`);
      for (const r of radios) {
        if (r.value === String(prevScore)) r.checked = true;
      }
    }

    const radios = document.getElementsByName(`score_${renderedIdx}`);
    for (const r of radios) {
      r.addEventListener('change', (e) => {
        const val = parseInt(e.target.value, 10);
        answers[idx].scores[optPos] = val;
      });
    }
  });
}

function validateQuestion(idx) {
  const scores = answers[idx]?.scores || [];
  return scores.filter((s) => typeof s === 'number').length === 6;
}

function exportCsv() {
  const rows = [];
  rows.push(["participant_id", "global_question", "option_index", "score", ""]);
  for (let i = 0; i < questions.length; i++) {
    const qEntry = questions[i];
    const q = qEntry.q;
    const order = answers[i].seedShuffleOrder;
    for (let j = 0; j < 6; j++) {
      const optPos = order[j];
      const score = answers[i].scores[optPos];
      const isQualityFlag = qEntry.isQuality ? 1 : 0;
      rows.push([participantId, q, optPos, score, isQualityFlag]);
    }
  }
  const csv = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `ratings_participant_${participantId}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

el('startBtn').addEventListener('click', () => {
  const code = parseInt(el('participantCode').value, 10);
  if (Number.isNaN(code)) {
    alert('Please enter a participant code (integer).');
    return;
  }
  participantSetIdx = ((code % 8) + 8) % 8; // normalized mod
  participantId = code;
  // Prepare questions later after consent + instruction
  const baseFive = QUESTION_SETS[participantSetIdx].slice(); // stash for later
  window.__baseFive = baseFive;
  el('setup').classList.add('hidden');
  el('consent').classList.remove('hidden');
  el('declined').classList.add('hidden');
  el('instruction').classList.add('hidden');
});
// Consent flow
el('consentDecline').addEventListener('click', () => {
  el('consent').classList.add('hidden');
  el('declined').classList.remove('hidden');
});

el('consentAgree').addEventListener('click', () => {
  el('consent').classList.add('hidden');
  el('instruction').classList.remove('hidden');
});

// Instruction -> Begin
el('instructionContinue').addEventListener('click', () => {
  const baseFive = window.__baseFive || [];
  // Choose quality question as the next item after this set in GLOBAL_ORDER (wrap-around)
  const qualityIdxInGlobal = (participantSetIdx * 5 + 5) % GLOBAL_ORDER.length;
  const whichQ = GLOBAL_ORDER[qualityIdxInGlobal];
  // Insert one quality question at a deterministic position per participant
  const rngQ = mulberry32(777 + participantId);
  const insertAt = Math.floor(rngQ() * (baseFive.length + 1)); // position 0..5
  questions = baseFive.map(q => ({ q, isQuality: false }));
  questions.splice(insertAt, 0, { q: whichQ, isQuality: true }); // now 6 questions
  answers = new Array(questions.length);
  currentIdx = 0;
  el('instruction').classList.add('hidden');
  el('quiz').classList.remove('hidden');
  el('finish').classList.add('hidden');
  loadQuestion(currentIdx);
});

el('prevBtn').addEventListener('click', () => {
  if (currentIdx > 0) {
    currentIdx -= 1;
    loadQuestion(currentIdx);
    el('validationMsg').textContent = '';
  }
});

el('nextBtn').addEventListener('click', () => {
  if (!validateQuestion(currentIdx)) {
    el('validationMsg').textContent = 'Please rate all 6 options before proceeding.';
    el('validationMsg').classList.add('danger');
    return;
  }
  el('validationMsg').textContent = '';
  el('validationMsg').classList.remove('danger');
  if (currentIdx < questions.length - 1) {
    currentIdx += 1;
    loadQuestion(currentIdx);
  } else {
    el('quiz').classList.add('hidden');
    el('finish').classList.remove('hidden');
  }
});

el('downloadBtn').addEventListener('click', () => {
  exportCsv();
});


