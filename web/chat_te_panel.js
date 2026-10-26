import { app } from "../../scripts/app.js";

// ============================================================
// Chat TE sidebar panel
//
// Talks to this same custom node's own /v1/chat/completions and
// /chat_te/* routes (see py/routes.py). No separate LLM process --
// the backend is whatever Qwen2.5-VL CLIP is currently registered
// (loaded from here, or connected via the QwenChatBridgeLoader node).
// ============================================================

const STYLE_ID = "chat-te-panel-style";

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ID;
    style.textContent = `
.chat-te-root { display: flex; flex-direction: column; height: 100%; min-height: 0; font-size: 13px; color: var(--fg-color, #ddd); overflow: hidden; box-sizing: border-box; }
.chat-te-header { display: flex; align-items: center; gap: 4px; padding: 6px 10px; border-bottom: 1px solid var(--border-color, #444); flex-shrink: 0; }
.chat-te-header-title { font-weight: 600; flex: 1; }
.chat-te-model-dot { font-size: 10px; color: #777; cursor: default; }
.chat-te-model-dot.ok { color: #6c6; }
.chat-te-model-dot.err { color: #e66; }
.chat-te-icon-btn {
    background: transparent; border: 1px solid transparent; color: var(--fg-color, #ddd);
    border-radius: 4px; cursor: pointer; padding: 2px 6px; font-size: 14px; line-height: 1.4;
}
.chat-te-icon-btn:hover { background: var(--comfy-input-bg, #222); border-color: var(--border-color, #444); }
.chat-te-icon-btn.active { background: var(--comfy-input-bg, #222); border-color: var(--border-color, #444); }

.chat-te-settings { padding: 8px 10px; border-bottom: 1px solid var(--border-color, #444); flex-shrink: 0; max-height: 45%; overflow-y: auto; box-sizing: border-box; }
.chat-te-row { display: flex; gap: 6px; align-items: center; margin-bottom: 6px; }
.chat-te-row:last-child { margin-bottom: 0; }
.chat-te-root select, .chat-te-root input[type=text], .chat-te-root input[type=number], .chat-te-root textarea, .chat-te-root button {
    background: var(--comfy-input-bg, #222); color: var(--input-text, #ddd);
    border: 1px solid var(--border-color, #444); border-radius: 4px; padding: 4px 6px; font-size: 12px;
}
.chat-te-root select { flex: 1; min-width: 0; }
.chat-te-root input[type=number] { width: 70px; }
.chat-te-root button { cursor: pointer; white-space: nowrap; }
.chat-te-root button:hover { filter: brightness(1.2); }
.chat-te-root button:disabled { opacity: 0.5; cursor: default; }
.chat-te-inline-label { display: flex; align-items: center; gap: 4px; font-size: 12px; cursor: default; }
.chat-te-inline-label input[type=checkbox] { cursor: pointer; }
.chat-te-status { font-size: 11px; opacity: 0.8; }
.chat-te-status.ok { color: #6c6; }
.chat-te-status.err { color: #e66; }
.chat-te-unload-btn { margin-left: auto; font-size: 11px; padding: 2px 6px; }

.chat-te-chat { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.chat-te-messages { flex: 1; min-height: 0; overflow-y: auto; padding: 8px 10px; display: flex; flex-direction: column; gap: 8px; }
.chat-te-msg { max-width: 92%; padding: 6px 9px; border-radius: 8px; white-space: pre-wrap; word-break: break-word; line-height: 1.4; }
.chat-te-msg-user { align-self: flex-end; background: var(--comfy-menu-bg, #3a3f5c); }
.chat-te-msg-assistant { align-self: flex-start; background: var(--comfy-input-bg, #2a2a2a); }
.chat-te-msg img { max-width: 100%; border-radius: 6px; margin-top: 4px; display: block; }
.chat-te-msg-actions { margin-top: 4px; display: flex; gap: 6px; }
.chat-te-msg-actions button { font-size: 11px; padding: 2px 6px; }

.chat-te-input-area { border-top: 1px solid var(--border-color, #444); padding: 8px 10px; flex-shrink: 0; box-sizing: border-box; }
.chat-te-attach-preview { display: flex; align-items: center; gap: 6px; margin-bottom: 6px; flex-wrap: wrap; }
.chat-te-attach-preview img { width: 40px; height: 40px; object-fit: cover; border-radius: 4px; }
.chat-te-dropzone { border: 1px dashed var(--border-color, #555); border-radius: 4px; padding: 6px; text-align: center; font-size: 11px; opacity: 0.75; cursor: pointer; margin-bottom: 6px; }
.chat-te-dropzone.drag-over { opacity: 1; border-color: #6c9; }
.chat-te-root textarea.chat-te-textinput { width: 100%; resize: vertical; min-height: 42px; box-sizing: border-box; }
.chat-te-sysprompt { width: 100%; box-sizing: border-box; min-height: 32px; resize: vertical; }
.chat-te-send-row { display: flex; align-items: center; gap: 6px; margin-top: 6px; }
.chat-te-chatstatus { margin-left: auto; font-size: 11px; opacity: 0.85; text-align: right; max-width: 55%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.chat-te-chatstatus.ok { color: #6c6; }
.chat-te-chatstatus.err { color: #e66; }
`;
    document.head.appendChild(style);
}

function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

// Stages a dropped/picked file into ComfyUI's own input/ folder via the
// stock /upload/image endpoint (the same one the built-in LoadImage node's
// upload button uses) so a graph node can reference it by filename -- no
// second drop/browse needed to get the same image into the workflow.
async function uploadImageToComfy(file) {
    const body = new FormData();
    body.append("image", file, file.name);
    const res = await fetch("/upload/image", { method: "POST", body });
    if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data?.error || `HTTP ${res.status}`);
    }
    return res.json(); // {name, subfolder, type}
}

// Qwen2.5-VL / Qwen3-VL are fed a video as several still frames (one vision
// block per frame), so the sampling happens here in the browser: seek a
// <video> element to evenly spaced timestamps and grab each frame via canvas.
// No server-side decoder (ffmpeg / av) needed, and the existing multi-image
// chat path handles the rest. Frames are downscaled because every frame costs
// a full image's worth of vision tokens.
async function extractVideoFrames(file, count, maxSide = 512) {
    const url = URL.createObjectURL(file);
    const video = document.createElement("video");
    video.muted = true;
    video.preload = "auto";
    video.src = url;
    try {
        await new Promise((resolve, reject) => {
            video.onloadedmetadata = resolve;
            video.onerror = () => reject(new Error("could not decode this video (unsupported codec?)"));
        });
        const duration = video.duration;
        if (!Number.isFinite(duration) || duration <= 0 || !video.videoWidth) {
            throw new Error("video has no readable duration / size");
        }
        const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
        canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
        const ctx = canvas.getContext("2d");
        const frames = [];
        for (let i = 0; i < count; i++) {
            // Centre of each of `count` equal slices, so the first/last frame
            // aren't the (often black) very first/last instants.
            const t = Math.min(duration - 0.001, (duration * (i + 0.5)) / count);
            await new Promise((resolve, reject) => {
                video.onseeked = resolve;
                video.onerror = () => reject(new Error("seek failed"));
                video.currentTime = t;
            });
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            frames.push(canvas.toDataURL("image/jpeg", 0.85));
        }
        return frames;
    } finally {
        URL.revokeObjectURL(url);
    }
}

async function callChatTE(messages, { maxTokens = 512, temperature = 0.7, enableThinking = false } = {}) {
    const res = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            model: "qwen2.5-vl-chat-te",
            messages,
            max_tokens: maxTokens,
            temperature,
            enable_thinking: enableThinking,
            stream: false,
        }),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
        throw new Error(data?.error?.message || data?.error || `HTTP ${res.status}`);
    }
    return data.choices?.[0]?.message || {};
}

// Best-effort: find a widget matching `predicate` on the currently selected
// node first, falling back to any node in the graph. Shared by "Send to
// encoder" (text widgets) and "Send to workflow" (image widgets).
function findWidget(predicate) {
    const graph = app.graph;
    if (!graph) return null;

    let selected = [];
    try {
        const sel = app.canvas?.selected_nodes;
        if (sel) selected = Object.values(sel);
    } catch (e) { /* ignore */ }

    const pools = [selected, graph._nodes || []];
    for (const pool of pools) {
        for (const node of pool) {
            const widget = node?.widgets?.find(predicate);
            if (widget) return widget;
        }
    }
    return null;
}

function findPromptWidget() {
    const names = ["user_prompt", "text", "prompt"];
    return findWidget((w) => names.includes(w.name) && typeof w.value === "string");
}

// LoadImage and LoadImage-alike nodes expose their file picker as a combo
// widget literally named "image".
function findImageWidget() {
    return findWidget((w) => w.name === "image");
}

function buildPanel(container) {
    ensureStyles();

    // Some ComfyUI sidebar-tab hosts don't force a bounded height onto the
    // container they hand us, which lets `.chat-te-root` grow past the
    // panel's visible area instead of scrolling internally -- the input
    // row then drifts out of view as the chat grows, and the overflow can
    // visually bleed into whatever sits below (e.g. another sidebar tab).
    container.style.height = "100%";
    container.style.minHeight = "0";
    container.style.overflow = "hidden";
    container.style.boxSizing = "border-box";

    container.innerHTML = `
        <div class="chat-te-root">
            <div class="chat-te-header">
                <span class="chat-te-header-title">Chat TE</span>
                <span id="chat-te-model-dot" class="chat-te-model-dot" title="Not loaded">&#9679;</span>
                <button id="chat-te-settings-btn" class="chat-te-icon-btn" title="Model / system prompt settings">&#9881;</button>
            </div>

            <div id="chat-te-settings" class="chat-te-settings" style="display:none">
                <div class="chat-te-row">
                    <select id="chat-te-model-select"><option>...</option></select>
                    <button id="chat-te-refresh-btn" title="Refresh list">&#8635;</button>
                    <button id="chat-te-load-btn">Load</button>
                </div>
                <div class="chat-te-row">
                    <span id="chat-te-status" class="chat-te-status">Not loaded</span>
                    <button id="chat-te-unload-btn" class="chat-te-unload-btn" title="Free VRAM via ComfyUI's standard /free endpoint. This affects every model ComfyUI currently has loaded, not just this one -- and the next chat message will automatically reload this one, with the usual reload delay.">Unload</button>
                </div>
                <div class="chat-te-row">
                    <textarea id="chat-te-sysprompt" class="chat-te-sysprompt" placeholder="System prompt (optional)"></textarea>
                </div>
                <div class="chat-te-row">
                    <label class="chat-te-inline-label" title="Qwen3-VL / Qwen3.5 only -- ignored by Qwen2.5-VL models. Leave this on unless you specifically want to suppress the model's reasoning preamble; turning it off has caused broken output on at least one model (Qwen3-VL-4B).">
                        <input id="chat-te-thinking" type="checkbox" checked>
                        Thinking mode
                    </label>
                </div>
                <div class="chat-te-row">
                    <label class="chat-te-inline-label" for="chat-te-max-tokens">Max tokens</label>
                    <input id="chat-te-max-tokens" type="number" min="1" max="8192" step="1" value="512">
                </div>
                <div class="chat-te-row">
                    <label class="chat-te-inline-label" for="chat-te-video-frames" title="Frames sampled evenly from an attached video; each frame costs as many vision tokens as one image">Video frames</label>
                    <input id="chat-te-video-frames" type="number" min="1" max="32" step="1" value="8">
                </div>
            </div>

            <div class="chat-te-chat">
                <div id="chat-te-messages" class="chat-te-messages"></div>
                <div class="chat-te-input-area">
                    <div id="chat-te-dropzone" class="chat-te-dropzone">Drop / click to attach an image or video</div>
                    <input id="chat-te-file-input" type="file" accept="image/*,video/*" style="display:none">
                    <div id="chat-te-attach-preview" class="chat-te-attach-preview" style="display:none">
                        <img id="chat-te-attach-thumb">
                        <span id="chat-te-attach-label">attached</span>
                        <button id="chat-te-attach-sendwf" title="Write this image's filename into the selected node's image widget (e.g. LoadImage)" disabled>Send to workflow</button>
                        <button id="chat-te-attach-clear">&times;</button>
                    </div>
                    <textarea id="chat-te-textinput" class="chat-te-textinput" placeholder="Message... (Enter to send, Shift+Enter for newline)"></textarea>
                    <div class="chat-te-send-row">
                        <button id="chat-te-send-btn">Send</button>
                        <button id="chat-te-clear-btn">Clear</button>
                        <span id="chat-te-chatstatus" class="chat-te-chatstatus"></span>
                    </div>
                </div>
            </div>
        </div>
    `;

    const settingsBtn = container.querySelector("#chat-te-settings-btn");
    const settingsView = container.querySelector("#chat-te-settings");
    const modelDot = container.querySelector("#chat-te-model-dot");
    const modelSelect = container.querySelector("#chat-te-model-select");
    const refreshBtn = container.querySelector("#chat-te-refresh-btn");
    const loadBtn = container.querySelector("#chat-te-load-btn");
    const unloadBtn = container.querySelector("#chat-te-unload-btn");
    const statusEl = container.querySelector("#chat-te-status");
    const sysPromptEl = container.querySelector("#chat-te-sysprompt");
    const thinkingEl = container.querySelector("#chat-te-thinking");
    const maxTokensEl = container.querySelector("#chat-te-max-tokens");
    const videoFramesEl = container.querySelector("#chat-te-video-frames");
    const messagesEl = container.querySelector("#chat-te-messages");
    const dropzone = container.querySelector("#chat-te-dropzone");
    const fileInput = container.querySelector("#chat-te-file-input");
    const attachPreview = container.querySelector("#chat-te-attach-preview");
    const attachThumb = container.querySelector("#chat-te-attach-thumb");
    const attachLabel = container.querySelector("#chat-te-attach-label");
    const attachSendWfBtn = container.querySelector("#chat-te-attach-sendwf");
    const attachClear = container.querySelector("#chat-te-attach-clear");
    const textInput = container.querySelector("#chat-te-textinput");
    const sendBtn = container.querySelector("#chat-te-send-btn");
    const clearBtn = container.querySelector("#chat-te-clear-btn");
    const chatStatusEl = container.querySelector("#chat-te-chatstatus");

    let history = []; // {role, text, imageDataUrls?}
    let attachedImages = []; // data URLs (one for an image, N sampled frames for a video), used for the chat preview + vision payload
    let attachedImageUpload = null; // {name, subfolder, type} once staged into ComfyUI's input/ folder

    // The system prompt textarea otherwise only lives in this closure's DOM
    // and is lost the moment the sidebar tab unmounts or ComfyUI reloads --
    // persist it across those via localStorage. Wrapped in try/catch since
    // localStorage can throw (private browsing, disabled storage, etc.);
    // losing persistence there is fine, breaking the panel over it is not.
    const SYSTEM_PROMPT_STORAGE_KEY = "chat-te-system-prompt";
    try {
        const savedSystemPrompt = localStorage.getItem(SYSTEM_PROMPT_STORAGE_KEY);
        if (savedSystemPrompt) sysPromptEl.value = savedSystemPrompt;
    } catch (e) { /* ignore */ }
    sysPromptEl.addEventListener("input", () => {
        try {
            localStorage.setItem(SYSTEM_PROMPT_STORAGE_KEY, sysPromptEl.value);
        } catch (e) { /* ignore */ }
    });

    // Persistent status (model load state) -- lives in the settings panel
    // and mirrors onto the small header dot so it's visible even collapsed.
    function setStatus(text, cls) {
        statusEl.textContent = text;
        statusEl.className = `chat-te-status ${cls || ""}`;
        modelDot.className = `chat-te-model-dot ${cls || ""}`;
        modelDot.title = text;
    }

    // Transient status (Thinking..., errors, "sent to workflow"...) -- lives
    // bottom-right of the input row so it stays visible without scrolling
    // back up as the conversation grows.
    function setChatStatus(text, cls) {
        chatStatusEl.textContent = text;
        chatStatusEl.className = `chat-te-chatstatus ${cls || ""}`;
    }

    function setSettingsOpen(open) {
        settingsView.style.display = open ? "block" : "none";
        settingsBtn.classList.toggle("active", open);
    }

    settingsBtn.addEventListener("click", () => {
        setSettingsOpen(settingsView.style.display === "none");
    });

    // Builds <option> nodes via DOM properties (value/textContent) rather
    // than an innerHTML template string -- the entries here come from a
    // server-side folder listing, which normally isn't attacker-influenced,
    // but this avoids depending on that: a filename containing `<`/`"` can't
    // break the markup or inject anything this way.
    function setSelectOptions(selectEl, entries) {
        selectEl.replaceChildren(...entries.map(({ value, label }) => {
            const opt = document.createElement("option");
            opt.value = value;
            opt.textContent = label;
            return opt;
        }));
    }

    async function refreshTextEncoders() {
        try {
            const res = await fetch("/chat_te/text_encoders");
            const data = await res.json();
            const names = data.text_encoders || [];
            setSelectOptions(modelSelect, names.length
                ? names.map((n) => ({ value: n, label: n }))
                : [{ value: "", label: "(no Qwen text encoder found in models/text_encoders)" }]);
        } catch (e) {
            setSelectOptions(modelSelect, [{ value: "", label: "(failed to list text encoders)" }]);
        }
    }

    async function refreshStatus() {
        try {
            const res = await fetch("/chat_te/status");
            const data = await res.json();
            if (data.loaded) {
                setStatus(`Loaded: ${data.model_name} (${data.source})`, "ok");
            } else {
                setStatus("Not loaded", "");
            }
        } catch (e) {
            setStatus("Status check failed", "err");
        }
    }

    refreshBtn.addEventListener("click", refreshTextEncoders);
    refreshTextEncoders();
    refreshStatus();

    loadBtn.addEventListener("click", async () => {
        const modelName = modelSelect.value;
        if (!modelName) return;
        loadBtn.disabled = true;
        setStatus(`Loading ${modelName}...`, "");
        try {
            const res = await fetch("/chat_te/load", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ model_name: modelName }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
            setStatus(`Loaded: ${data.model_name}`, "ok");
        } catch (e) {
            setStatus(`Load failed: ${e.message}`, "err");
        } finally {
            loadBtn.disabled = false;
        }
    });

    // ComfyUI's own standard endpoint (server.py's POST /free), not
    // something Chat TE implements itself. Frees VRAM for every model
    // ComfyUI currently has loaded, chat backend included -- not just this
    // one -- and doesn't clear STATE.clip on the Python side, so the panel's
    // "Loaded: ..." status is left alone; the next chat message reloads it
    // automatically (with the same reload delay a workflow's own model
    // eviction would cause).
    unloadBtn.addEventListener("click", async () => {
        unloadBtn.disabled = true;
        try {
            const res = await fetch("/free", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ unload_models: true }),
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            setChatStatus("VRAM freed (reloads automatically on next message)", "ok");
        } catch (e) {
            setChatStatus(`Unload failed: ${e.message}`, "err");
        } finally {
            unloadBtn.disabled = false;
        }
    });

    function setAttachedImages(dataUrls) {
        attachedImages = dataUrls || [];
        if (attachedImages.length) {
            attachThumb.src = attachedImages[0];
            attachPreview.style.display = "flex";
        } else {
            attachPreview.style.display = "none";
            attachedImageUpload = null;
            attachSendWfBtn.disabled = true;
            attachLabel.textContent = "attached";
        }
    }

    // One drop/pick does both jobs: the data URL feeds the chat (vision)
    // request, and the same file is staged into ComfyUI's input/ folder so
    // "Send to workflow" can wire it into a graph node without asking the
    // user to attach it a second time.
    async function attachVideo(file) {
        const count = Math.min(32, Math.max(1, parseInt(videoFramesEl.value, 10) || 8));
        setAttachedImages(null);
        setChatStatus(`Sampling ${count} frames from ${file.name}...`, "");
        try {
            const frames = await extractVideoFrames(file, count);
            setAttachedImages(frames);
            attachLabel.textContent = `video: ${frames.length} frames`;
            setChatStatus("", "");
        } catch (e) {
            setChatStatus(`Video load failed: ${e.message}`, "err");
        }
        // The stock /upload/image + LoadImage widget route is image-only, so
        // "Send to workflow" stays disabled for videos.
    }

    async function attachFile(file) {
        if (file.type.startsWith("video/")) return attachVideo(file);
        setAttachedImages([await fileToDataUrl(file)]);
        attachedImageUpload = null;
        attachSendWfBtn.disabled = true;
        attachLabel.textContent = "uploading...";
        try {
            attachedImageUpload = await uploadImageToComfy(file);
            attachSendWfBtn.disabled = false;
            attachLabel.textContent = "attached";
        } catch (e) {
            attachLabel.textContent = "attached (workflow upload failed)";
            setChatStatus(`Image upload failed: ${e.message}`, "err");
        }
    }

    attachSendWfBtn.addEventListener("click", () => {
        if (!attachedImageUpload) return;
        const widget = findImageWidget();
        if (!widget) {
            setChatStatus("No image widget found (select a LoadImage-type node)", "err");
            return;
        }
        const filename = attachedImageUpload.subfolder
            ? `${attachedImageUpload.subfolder}/${attachedImageUpload.name}`
            : attachedImageUpload.name;
        if (widget.options?.values && !widget.options.values.includes(filename)) {
            widget.options.values.unshift(filename);
        }
        widget.value = filename;
        widget.callback?.(widget.value);
        app.graph.setDirtyCanvas(true, true);
        setChatStatus(`Sent "${filename}" to workflow`, "ok");
    });

    attachClear.addEventListener("click", () => setAttachedImages(null));
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", async (e) => {
        const file = e.target.files?.[0];
        if (file) await attachFile(file);
    });
    dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
    dropzone.addEventListener("drop", async (e) => {
        e.preventDefault();
        dropzone.classList.remove("drag-over");
        const file = e.dataTransfer.files?.[0];
        if (file && (file.type.startsWith("image/") || file.type.startsWith("video/"))) await attachFile(file);
    });

    function appendBubble(role, text, imageDataUrls) {
        const div = document.createElement("div");
        div.className = `chat-te-msg chat-te-msg-${role}`;
        if (text) div.appendChild(document.createTextNode(text));
        for (const url of imageDataUrls || []) {
            const img = document.createElement("img");
            img.src = url;
            div.appendChild(img);
        }
        if (role === "assistant" && text) {
            const actions = document.createElement("div");
            actions.className = "chat-te-msg-actions";
            const sendToEncoderBtn = document.createElement("button");
            sendToEncoderBtn.textContent = "Send to encoder";
            sendToEncoderBtn.addEventListener("click", () => {
                const widget = findPromptWidget();
                if (!widget) {
                    setChatStatus("No prompt widget found (select a text-encoder node)", "err");
                    return;
                }
                widget.value = text;
                widget.callback?.(widget.value);
                app.graph.setDirtyCanvas(true, true);
            });
            actions.appendChild(sendToEncoderBtn);
            div.appendChild(actions);
        }
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
    }

    async function sendMessage() {
        const text = textInput.value.trim();
        if (!text) return;

        const userMsg = { role: "user", text, imageDataUrls: attachedImages };
        history.push(userMsg);
        appendBubble("user", text, attachedImages);

        textInput.value = "";
        // Deliberately not clearing the attachment here: the preview and its
        // "Send to workflow" button stay up after sending so there's time to
        // wire the image into a graph node once the chat about it is done --
        // only the explicit clear ("x") button or attaching a new image
        // resets it (see setAttachedImages / attachClear below).

        sendBtn.disabled = true;
        setChatStatus("Thinking...", "");

        try {
            const wireMessages = history.map((m) => {
                if (m.role === "user" && m.imageDataUrls?.length) {
                    return {
                        role: "user",
                        content: [
                            { type: "text", text: m.text },
                            ...m.imageDataUrls.map((url) => ({ type: "image_url", image_url: { url } })),
                        ],
                    };
                }
                return { role: m.role, content: m.text };
            });

            const sysPrompt = sysPromptEl.value.trim();
            const fullMessages = sysPrompt ? [{ role: "system", content: sysPrompt }, ...wireMessages] : wireMessages;

            const maxTokens = Math.max(1, parseInt(maxTokensEl.value, 10) || 512);
            const reply = await callChatTE(fullMessages, { maxTokens, enableThinking: thinkingEl.checked });
            const replyText = reply.content || "";
            history.push({ role: "assistant", text: replyText });
            appendBubble("assistant", replyText);
            setChatStatus("", "");
        } catch (e) {
            history.pop();
            messagesEl.removeChild(messagesEl.lastChild);
            textInput.value = text;
            setChatStatus(`Error: ${e.message}`, "err");
        } finally {
            sendBtn.disabled = false;
        }
    }

    sendBtn.addEventListener("click", sendMessage);
    textInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    clearBtn.addEventListener("click", () => {
        history = [];
        messagesEl.innerHTML = "";
        setChatStatus("", "");
    });
}

app.registerExtension({
    name: "chat_te.panel",
    async setup() {
        app.extensionManager.registerSidebarTab({
            id: "chat-te-sidebar",
            title: "Chat TE",
            icon: "pi pi-comments",
            type: "custom",
            render: (container) => buildPanel(container),
        });
    },
});
