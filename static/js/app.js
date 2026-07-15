const chatLog = document.getElementById('chat-log');
const chatForm = document.getElementById('chat-form');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

const buildBtn = document.getElementById('build-btn');
const buildBtnLabel = document.getElementById('build-btn-label');
const buildStatus = document.getElementById('build-status');
const tickerInput = document.getElementById('ticker-input');
const trackedList = document.getElementById('tracked-list');
const tickerTrack = document.getElementById('ticker-track');

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');

let history = [];

function addMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `msg msg-${role}`;

  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  meta.textContent = role === 'user' ? 'YOU' : role === 'bot' ? 'LEDGER' : role.toUpperCase();

  const body = document.createElement('div');
  body.className = 'msg-body';
  body.textContent = text;

  wrap.appendChild(meta);
  wrap.appendChild(body);
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return body;
}

function renderTracked(tickers) {
  trackedList.innerHTML = '';
  if (!tickers || tickers.length === 0) {
    trackedList.innerHTML = '<li class="tracked-empty">nothing tracked yet</li>';
    tickerTrack.innerHTML = '<span class="ticker-empty">no tickers loaded yet — build your knowledge base above</span>';
    return;
  }
  tickers.forEach(t => {
    const li = document.createElement('li');
    li.innerHTML = `<span>${t}</span><span style="color: var(--teal)">●</span>`;
    trackedList.appendChild(li);
  });

  const items = tickers.map(t => `<span class="ticker-item">${t}</span>`).join('');
  tickerTrack.innerHTML = items + items;
}

async function refreshTracked() {
  try {
    const res = await fetch('/api/tickers');
    const data = await res.json();
    renderTracked(data.tickers);
  } catch (e) {
    renderTracked([]);
  }
}

async function checkStatus() {
  try {
    const res = await fetch('/api/tickers');
    if (res.ok) {
      statusDot.classList.add('online');
      statusText.textContent = 'connected';
    } else {
      throw new Error('bad response');
    }
  } catch (e) {
    statusDot.classList.add('offline');
    statusText.textContent = 'server unreachable';
  }
}

buildBtn.addEventListener('click', async () => {
  const raw = tickerInput.value.trim();
  if (!raw) {
    buildStatus.textContent = 'enter at least one ticker';
    buildStatus.className = 'build-status err';
    return;
  }
  buildBtn.disabled = true;
  buildBtnLabel.textContent = 'building…';
  buildStatus.textContent = 'fetching data and embedding — this can take a minute…';
  buildStatus.className = 'build-status';

  try {
    const res = await fetch('/api/build', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ tickers: raw }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'build failed');

    buildStatus.textContent = `stored ${data.documents} documents for ${data.tickers.join(', ')}`;
    buildStatus.className = 'build-status ok';
    await refreshTracked();
  } catch (e) {
    buildStatus.textContent = `error: ${e.message}`;
    buildStatus.className = 'build-status err';
  } finally {
    buildBtn.disabled = false;
    buildBtnLabel.textContent = 'build knowledge base';
  }
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const question = chatInput.value.trim();
  if (!question) return;

  addMessage('user', question);
  chatInput.value = '';
  sendBtn.disabled = true;

  const thinkingBody = addMessage('bot', 'thinking…');

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, history }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || 'request failed');

    thinkingBody.textContent = data.answer;
    history.push({ role: 'user', content: question });
    history.push({ role: 'assistant', content: data.answer });
    if (history.length > 20) history = history.slice(-20);
  } catch (e) {
    thinkingBody.parentElement.className = 'msg msg-error';
    thinkingBody.parentElement.querySelector('.msg-meta').textContent = 'ERROR';
    thinkingBody.textContent = e.message;
  } finally {
    sendBtn.disabled = false;
  }
});

checkStatus();
refreshTracked();

