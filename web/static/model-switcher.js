/*
 * Model Switcher Module
 * Provides dropdown selector + reload button for toggling LLM models via backend API.
 */

(() => {
    const API_BASE = 'http://127.0.0.1:11434/v1/llm';

    const modelSelect = document.getElementById('modelSelect');
    const applyBtn = document.getElementById('applyModelBtn');
    if (!modelSelect || !applyBtn) return;

    const statusDot = document.getElementById('statusDot');
    const statusText = document.getElementById('statusText');

    async function fetchModels() {
        try {
            const res = await fetch(`${API_BASE}/models`);
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            populateDropdown(data.models, data.active_model);
        } catch (err) {
            console.error('Failed to fetch models:', err);
        }
    }

    function populateDropdown(models, active) {
        modelSelect.innerHTML = '';
        Object.entries(models).forEach(([key, cfg]) => {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = cfg.label || key;
            if (key === active) opt.selected = true;
            modelSelect.appendChild(opt);
        });
    }

    async function switchModel() {
        const selected = modelSelect.value;
        if (!selected) return;
        applyBtn.disabled = true;
        updateStatus('Starting…', false);
        try {
            const res = await fetch(`${API_BASE}/switch`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ model_key: selected })
            });
            const data = await res.json();
            if (!data.success) {
                throw new Error(data.error || 'Unknown error');
            }
            // Begin polling until phase==running or error
            pollUntilReady();
        } catch (err) {
            console.error('Switch failed:', err);
            updateStatus('Error', false);
            alert(`Model switch failed: ${err.message}`);
            applyBtn.disabled = false;
        }
    }

    async function pollUntilReady() {
        let attempts = 0;
        const maxAttempts = 40; // ~80 seconds
        const interval = 2000;
        while (attempts < maxAttempts) {
            try {
                const res = await fetch(`${API_BASE}/models`);
                const data = await res.json();
                if (data.phase === 'running') {
                    updateStatus(`Active: ${data.active_model}`, true);
                    applyBtn.disabled = false;
                    fetchModels(); // refresh dropdown
                    return;
                }
                if (data.phase === 'error') {
                    updateStatus('Error', false);
                    const log = await (await fetch(`${API_BASE}/log_tail?lines=100`)).text();
                    console.error('Model start error log:\n', log);
                    alert(`Model start error. Check console for details.`);
                    applyBtn.disabled = false;
                    return;
                }
                // else still starting/stopping
                updateStatus(`${data.phase}…`, false);
            } catch (e) {
                console.warn('Polling error', e);
            }
            await new Promise(r => setTimeout(r, interval));
            attempts += 1;
        }
        updateStatus('Timeout', false);
        applyBtn.disabled = false;
    }

    function updateStatus(text, connected) {
        if (statusText) statusText.textContent = text;
        if (statusDot) {
            statusDot.classList.toggle('connected', connected);
            statusDot.classList.toggle('disconnected', !connected);
        }
    }

    applyBtn.addEventListener('click', switchModel);

    // Initial load
    fetchModels();
    // Periodic refresh to stay in sync if model switched elsewhere
    setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/models`);
            if (!res.ok) return;
            const data = await res.json();
            populateDropdown(data.models, data.active_model);
            updateStatus(`Active: ${data.active_model}`, data.phase === 'running');
        } catch {}
    }, 15000);
})(); 