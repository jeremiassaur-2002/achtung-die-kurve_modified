// ============================================================================
//  ai_bot.js - runs a trained PPO policy (exported to ONNX by
//  ai/core/export/export_onnx.py) as an in-browser opponent, using onnxruntime-web.
//
//  Setup:
//    1. Train + export a model (see ai/colab/Train_Achtung_Kurve_AI.ipynb). By
//       default ai/core/export/export_onnx.py also writes a `model_data.js` sidecar
//       (the .onnx bytes base64-embedded as `const AI_MODEL_BASE64 = "..."`).
//       Copy BOTH next to index.html, and add:
//         <script src="model_data.js" defer></script>
//       ...somewhere before you call addAI() (order relative to ai_bot.js
//       itself doesn't matter, since addAI only runs on user interaction).
//       This is what lets the game work by just double-clicking index.html
//       (file://) - onnxruntime-web loads the model via fetch() otherwise,
//       which browsers block for local files under file://.
//    2. Add this script tag to index.html, AFTER script.js:
//         <script src="ai_bot.js" defer></script>
//    3. From the start screen, tick a player's "KI" checkbox - or from the
//       console: addAI('fred'). Without a model_data.js, addAI(name, url) falls
//       back to fetching the .onnx by URL, which needs a local server (e.g.
//       `python -m http.server`), not file://.
//
//  Interface: identical shape to the old bot.js - a player with `isAI = true`
//  gets `aiThink(name)` called once per simulation tick (see script.js's
//  `if (players[player].isAI) aiThink(player)` hook), which sets turnL/turnR.
//  The observation (RGB frame stack + metadata vector) mirrors
//  ai/core/env/observation.py exactly, so any exported checkpoint drops in as-is.
//
//  ONNX Runtime Web itself isn't bundled here (no build step in this project,
//  same as Google Fonts being loaded via a CDN link in index.html) - it's
//  fetched from a CDN lazily, the first time addAI() is actually called.
// ============================================================================

const ORT_CDN_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"

// must match ai/core/env/observation.py's ObsConfig defaults used for training/export
const AI_DEFAULT_OBS_RESOLUTION = 96
const AI_DEFAULT_FRAME_STACK = 4
const AI_MAX_PLAYERS = 6 // ai/core/config/game_constants.py: gc.MAX_PLAYERS

const _aiState = {} // name -> { session, frames: Uint8Array[], obsResolution, frameStack, busy }
let _ortLoadPromise = null

function _ensureOrtLoaded() {
    if (window.ort) return Promise.resolve()
    if (_ortLoadPromise) return _ortLoadPromise
    _ortLoadPromise = new Promise((resolve, reject) => {
        const script = document.createElement("script")
        script.src = ORT_CDN_URL
        script.onload = () => resolve()
        script.onerror = () => reject(new Error(`could not load onnxruntime-web from ${ORT_CDN_URL}`))
        document.head.appendChild(script)
    })
    return _ortLoadPromise
}

// captures the live composite (items -> trails -> dots/border/heads, same
// bottom-to-top order as style.css's z-index stack) at the given resolution
function _captureFrame(size) {
    if (!_captureFrame._canvas) {
        _captureFrame._canvas = document.createElement("canvas")
        _captureFrame._ctx = _captureFrame._canvas.getContext("2d", { willReadFrequently: true })
    }
    const canvas = _captureFrame._canvas
    const ctx = _captureFrame._ctx
    if (canvas.width !== size || canvas.height !== size) {
        canvas.width = size
        canvas.height = size
    }
    ctx.fillStyle = "#000000"
    ctx.fillRect(0, 0, size, size)
    ctx.drawImage(powerupVisualCanvas, 0, 0, h, h, 0, 0, size, size)
    ctx.drawImage(trailsHitboxCanvas, 0, 0, h, h, 0, 0, size, size)
    ctx.drawImage(dotsCanvas, 0, 0, h, h, 0, 0, size, size)
    return ctx.getImageData(0, 0, size, size).data // RGBA Uint8ClampedArray
}

// RGBA HWC -> RGB CHW uint8, dropping alpha - matches ai/core/env/renderer.to_chw()
function _rgbaToChw(rgba, size) {
    const chw = new Uint8Array(3 * size * size)
    const plane = size * size
    for (let i = 0; i < plane; i++) {
        chw[i] = rgba[i * 4]
        chw[plane + i] = rgba[i * 4 + 1]
        chw[2 * plane + i] = rgba[i * 4 + 2]
    }
    return chw
}

// matches ai/core/env/observation.py's _build_vector exactly - same order, same normalization
function _buildVector(name) {
    const p = players[name]
    const half = h / 2

    let aliveOthers = 0
    for (const other in players) {
        if (other !== name && players[other].ready && players[other].alive) aliveOthers++
    }

    const base = [
        p.powerup.speed,
        p.powerup.size,
        Math.cos(p.dir),
        Math.sin(p.dir),
        (p.x / h) * 2 - 1,
        (p.y / h) * 2 - 1,
        Math.min(p.x, h - p.x, p.y, h - p.y) / half,
        p.powerup.reverse ? 1 : 0,
        p.powerup.invisible ? 1 : 0,
        p.powerup.side ? 1 : 0,
        p.powerup.ghost ? 1 : 0,
        p.powerup.freeze ? 1 : 0,
        p.powerup.sineStart !== null ? 1 : 0,
        aliveOthers / (AI_MAX_PLAYERS - 1),
        achtung.fieldInset / half,
        achtung.sides !== 0 ? 1 : 0,
    ]
    const onehot = new Array(AI_MAX_PLAYERS).fill(0)
    onehot[Object.keys(players).indexOf(name)] = 1
    return Float32Array.from(base.concat(onehot))
}

async function aiThink(name) {
    const p = players[name]
    const state = _aiState[name]
    if (!p.isAI || !p.alive || !state || !state.session || state.busy) return
    state.busy = true
    try {
        const rgba = _captureFrame(state.obsResolution)
        const chw = _rgbaToChw(rgba, state.obsResolution)

        if (state.frames.length === 0) {
            for (let i = 0; i < state.frameStack; i++) state.frames.push(chw)
        } else {
            state.frames.push(chw)
            if (state.frames.length > state.frameStack) state.frames.shift()
        }
        const stacked = new Uint8Array(3 * state.frameStack * state.obsResolution * state.obsResolution)
        state.frames.forEach((frame, i) => stacked.set(frame, i * frame.length))

        const imageTensor = new ort.Tensor("uint8", stacked, [1, 3 * state.frameStack, state.obsResolution, state.obsResolution])
        const vectorTensor = new ort.Tensor("float32", _buildVector(name), [1, AI_MAX_PLAYERS + 16])

        const output = await state.session.run({ image: imageTensor, vector: vectorTensor })
        const logits = output.action_logits.data
        let best = 0
        for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i

        // action encoding matches ai/core/env/engine.py: 0 = turn left, 1 = straight, 2 = turn right
        p.turnL = best === 0
        p.turnR = best === 2
    } catch (err) {
        console.error(`aiThink(${name}) failed:`, err)
    } finally {
        state.busy = false
    }
}

function _base64ToBytes(base64) {
    const binary = atob(base64)
    const bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
    return bytes
}

// ============================================================================
//  AI player management (mirrors the old bot.js's addBot/removeBot shape)
// ============================================================================
async function addAI(name, modelUrl = "model.onnx", obsResolution = AI_DEFAULT_OBS_RESOLUTION, frameStack = AI_DEFAULT_FRAME_STACK) {
    if (!players[name]) return console.warn("Kein Spieler namens", name)
    await _ensureOrtLoaded()

    // prefer the embedded model_data.js bytes (works under file://); only fetch
    // by URL if that sidecar wasn't loaded (needs a local server, not file://)
    const modelSource = typeof AI_MODEL_BASE64 !== "undefined" ? _base64ToBytes(AI_MODEL_BASE64) : modelUrl
    const session = await ort.InferenceSession.create(modelSource, { executionProviders: ["wasm"] })
    _aiState[name] = { session, frames: [], obsResolution, frameStack, busy: false }

    players[name].isAI = true
    players[name].active = true
    players[name].ready = true
    players[name].keyL = false
    players[name].keyR = false

    const wrap = document.querySelector(`.player_wrapper.${name}`)
    if (wrap) {
        wrap.classList.add("focus")
        const lt = wrap.querySelector(".key_wrapper_left .key_text")
        const rt = wrap.querySelector(".key_wrapper_right .key_text")
        if (lt) lt.textContent = "AI"
        if (rt) rt.textContent = ""
    }
    const sourceLabel = typeof AI_MODEL_BASE64 !== "undefined" ? "embedded model_data.js" : modelUrl
    console.log(`${name} ist jetzt eine KI (${sourceLabel}).`)
}

function removeAI(name) {
    if (!players[name]) return
    players[name].isAI = false
    players[name].ready = false
    players[name].active = false
    delete _aiState[name]
    const wrap = document.querySelector(`.player_wrapper.${name}`)
    if (wrap) {
        wrap.classList.remove("focus")
        const lt = wrap.querySelector(".key_wrapper_left .key_text")
        const rt = wrap.querySelector(".key_wrapper_right .key_text")
        if (lt) lt.textContent = ""
        if (rt) rt.textContent = ""
    }
}
