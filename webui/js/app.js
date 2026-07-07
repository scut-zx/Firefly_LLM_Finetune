/* ============================================
   流萤角色助手 - 前端逻辑
   改编自 reference_code zzz-yixuan-webui/js/app.js
   ============================================ */

// ============================================
// Configuration
// ============================================
const API_BASE = '';
const MODEL_NAME = 'firefly-assistant';

// State
let isGenerating = false;
let useRAG = true;
let enableValidation = true;

// DOM Elements
const messagesEl = document.getElementById('messages');
const inputEl = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const ragToggle = document.getElementById('ragToggle');
const validationToggle = document.getElementById('validationToggle');

// ============================================
// Firefly Particle Canvas
// ============================================
class FireflyParticles {
    constructor() {
        this.canvas = document.getElementById('fireflyCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.particles = [];
        this.maxParticles = 35;
        this.resize();
        this.init();
        this.animate();

        window.addEventListener('resize', () => this.resize());
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    init() {
        for (let i = 0; i < this.maxParticles; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                radius: Math.random() * 2.5 + 0.8,
                alpha: Math.random() * 0.6 + 0.1,
                speedX: (Math.random() - 0.5) * 0.4,
                speedY: (Math.random() - 0.5) * 0.4,
                pulse: Math.random() * Math.PI * 2,
                pulseSpeed: Math.random() * 0.02 + 0.005,
                color: Math.random() < 0.7
                    ? { r: 126, g: 200, b: 123 }  // Green firefly
                    : { r: 245, g: 176, b: 65 },   // Amber firefly
            });
        }
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        for (const p of this.particles) {
            // Update position
            p.x += p.speedX;
            p.y += p.speedY;
            p.pulse += p.pulseSpeed;

            // Wrap around edges
            if (p.x < -10) p.x = this.canvas.width + 10;
            if (p.x > this.canvas.width + 10) p.x = -10;
            if (p.y < -10) p.y = this.canvas.height + 10;
            if (p.y > this.canvas.height + 10) p.y = -10;

            // Pulse alpha
            const currentAlpha = p.alpha * (0.5 + 0.5 * Math.sin(p.pulse));

            // Draw glow
            const gradient = this.ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.radius * 4);
            gradient.addColorStop(0, `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${currentAlpha})`);
            gradient.addColorStop(0.5, `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${currentAlpha * 0.3})`);
            gradient.addColorStop(1, `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, 0)`);

            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.radius * 4, 0, Math.PI * 2);
            this.ctx.fillStyle = gradient;
            this.ctx.fill();

            // Draw core
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = `rgba(${p.color.r}, ${p.color.g}, ${p.color.b}, ${currentAlpha + 0.3})`;
            this.ctx.fill();
        }

        requestAnimationFrame(() => this.animate());
    }
}

// ============================================
// Initialization
// ============================================
function init() {
    // Start particle animation
    new FireflyParticles();

    // Event bindings
    sendBtn.addEventListener('click', sendMessage);
    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    inputEl.addEventListener('input', autoResize);

    // Toggle switches
    ragToggle.addEventListener('click', () => toggleSwitch(ragToggle, 'rag'));
    validationToggle.addEventListener('click', () => toggleSwitch(validationToggle, 'validation'));

    const ragToggleMobile = document.getElementById('ragToggleMobile');
    const validationToggleMobile = document.getElementById('validationToggleMobile');
    if (ragToggleMobile) {
        ragToggleMobile.addEventListener('click', () => toggleSwitch(ragToggleMobile, 'rag'));
    }
    if (validationToggleMobile) {
        validationToggleMobile.addEventListener('click', () => toggleSwitch(validationToggleMobile, 'validation'));
    }

    // Quick questions
    document.querySelectorAll('.quick-question').forEach(btn => {
        btn.addEventListener('click', () => {
            inputEl.value = btn.textContent.trim();
            sendMessage();
            closeDrawer('right');
        });
    });

    // Drawer overlay
    document.getElementById('drawerOverlay').addEventListener('click', () => {
        closeDrawer('left');
        closeDrawer('right');
    });

    console.log('✅ 流萤角色助手已启动');
    console.log('API 地址:', API_BASE || '同源');
    console.log('模型:', MODEL_NAME);

    // Initial greeting
    addMessage('bot', '嗨…又见面啦。叫我流萤就好。今天想去哪儿？一起走走吧。');
}

// ============================================
// Send Message
// ============================================
async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text || isGenerating) return;

    addMessage('user', text);
    inputEl.value = '';
    inputEl.style.height = 'auto';

    isGenerating = true;
    sendBtn.disabled = true;
    const thinkingEl = addThinking();

    try {
        const response = await callAPI(text);
        thinkingEl.remove();

        if (response) {
            addMessage('bot', response);
        } else {
            addMessage('bot', '…嗯，我一时不知道该说什么。可以再说一遍吗？');
        }
    } catch (error) {
        thinkingEl.remove();
        console.error('API error:', error);

        let errorMsg = '⚠️ 请求失败';
        if (error.message.includes('503') || error.message.includes('Connect')) {
            errorMsg = '服务暂时不可用…请确保后端和 vLLM 已启动。';
        } else if (error.message.includes('502')) {
            errorMsg = 'vLLM 返回了错误…请检查模型是否正确加载。';
        }
        addMessage('bot', errorMsg);
    } finally {
        isGenerating = false;
        sendBtn.disabled = false;
    }
}

// ============================================
// API Call
// ============================================
async function callAPI(question) {
    const url = `${API_BASE}/v1/chat/completions`;

    const requestBody = {
        model: MODEL_NAME,
        messages: [
            {
                role: 'user',
                content: question
            }
        ],
        max_tokens: 512,
        temperature: 0.7,
        top_p: 0.9,
        top_k: 50,
        use_rag: useRAG,
        enable_validation: enableValidation,
    };

    console.log('发送请求:', url, requestBody);

    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`HTTP ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    console.log('API 响应:', data);

    try {
        let content = data.choices[0].message.content;

        // Filter <think> tags
        content = content.replace(/<think>.*?<\/think>\s*/gs, '').trim();

        return content;
    } catch (e) {
        console.error('解析响应失败:', e, data);
        throw new Error('响应解析失败');
    }
}

// ============================================
// UI Helpers
// ============================================
function addMessage(role, content) {
    const msgEl = document.createElement('div');
    msgEl.className = `message ${role}`;

    const avatarText = role === 'bot' ? '萤' : '你';
    const avatarClass = role === 'bot' ? 'bot' : 'user';

    msgEl.innerHTML = `
        <div class="message-avatar ${avatarClass}">${avatarText}</div>
        <div class="message-content">${escapeHtml(content)}</div>
    `;

    messagesEl.appendChild(msgEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addThinking() {
    const msgEl = document.createElement('div');
    msgEl.className = 'message bot';
    msgEl.innerHTML = `
        <div class="message-avatar bot">萤</div>
        <div class="message-content">
            <div class="thinking">
                <span>流萤正在思考</span>
                <div class="thinking-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
    `;
    messagesEl.appendChild(msgEl);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return msgEl;
}

function toggleSwitch(el, type) {
    el.classList.toggle('active');
    if (type === 'rag') {
        useRAG = el.classList.contains('active');
        // Sync mobile
        const other = document.getElementById('ragToggleMobile');
        if (other && other !== el) {
            other.classList.toggle('active', useRAG);
        }
    }
    if (type === 'validation') {
        enableValidation = el.classList.contains('active');
        const other = document.getElementById('validationToggleMobile');
        if (other && other !== el) {
            other.classList.toggle('active', enableValidation);
        }
    }
    if (type === 'rag') {
        document.getElementById('ragToggleMobile')?.classList.toggle('active', useRAG);
    }
}

function autoResize() {
    inputEl.style.height = 'auto';
    inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ============================================
// Drawer Controls
// ============================================
function openDrawer(side) {
    const drawer = side === 'left'
        ? document.getElementById('drawerLeft')
        : document.getElementById('drawerRight');
    const overlay = document.getElementById('drawerOverlay');
    drawer.classList.add('active');
    overlay.classList.add('active');
}

function closeDrawer(side) {
    const drawer = side === 'left'
        ? document.getElementById('drawerLeft')
        : document.getElementById('drawerRight');
    const overlay = document.getElementById('drawerOverlay');
    drawer.classList.remove('active');
    overlay.classList.remove('active');
}

function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
    inputEl.focus();
}

// ============================================
// Launch
// ============================================
init();
