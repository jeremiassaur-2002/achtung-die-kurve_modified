// ============================================================================
//  bot.js  v2  -  spielstarker, offensiver KI-Gegner fuer "Achtung, die Kurve!"
//  (reine Regel-Logik, KEIN ML-Modell)
//
//  Drop-in: in index.html NACH script.js einbinden:
//     <script src="script.js" defer></script>
//     <script src="bot.js" defer></script>
//
//  Und in script.js EINE Zeile in der draw()-Schleife ergaenzen, direkt nach
//     if (!players[player].ready) continue
//  einfuegen:
//     if (players[player].isBot && players[player].alive) botThink(player)
//
//  Was v2 besser macht als der reine Ueberlebens-Bot:
//   - schaut weiter voraus, KLEBT ABER NICHT MEHR AM RAND (zieht ins Feld)
//   - faehrt gezielt auf sinnvolle Powerups zu (r_* gegen dich zuerst)
//   - VERFOLGT dich und schneidet dir den Weg ab (Abschneidepunkt vor dir)
//   - Anti-Spiral-Bremse: dreht sich nicht in die eigene Schleife
//   - Sicherheit bleibt harte Grenze: opfert sich nie fuer Aggression/Powerup
// ============================================================================

// ---- Schwierigkeitsgrade ----------------------------------------------------
//  lookahead    : Frames Vorausschau (mehr = klueger)
//  react        : Frames, die eine Entscheidung gehalten wird (mehr = traeger)
//  turnCommit   : wie lange eine simulierte Kurve gehalten wird, bevor "gerade"
//                 gerechnet wird - 30 laesst den Bot seine eigene Schleife
//                 rechtzeitig erkennen (per Test ermittelt, nicht aendern ohne Not)
//  straightBias : Vorliebe fuers Geradeausfahren (mehr = ruhiger)
//  noise        : Chance auf einen Zufalls-Patzer (mehr = dummer/menschlicher)
//  powerup      : wie stark Powerups angesteuert werden
//  aggression   : wie stark dir der Weg abgeschnitten wird (das "Jagd"-Verhalten)
//  wallAvoid    : wie stark der Rand gemieden wird (gegen langweiliges Rand-Kleben)
const BOT_DIFFICULTIES = {
    easy:   { lookahead: 35,  react: 4, turnCommit: 30, straightBias: 8, noise: 0.10, powerup: 0.3, aggression: 0.3, wallAvoid: 0.3 },
    medium: { lookahead: 60,  react: 2, turnCommit: 30, straightBias: 5, noise: 0.03, powerup: 0.7, aggression: 0.8, wallAvoid: 0.5 },
    hard:   { lookahead: 90,  react: 1, turnCommit: 30, straightBias: 3, noise: 0.00, powerup: 1.0, aggression: 1.2, wallAvoid: 0.6 },
    hunter: { lookahead: 110, react: 1, turnCommit: 30, straightBias: 2, noise: 0.00, powerup: 1.3, aggression: 1.8, wallAvoid: 0.5 },
}

let BOT_LEVEL = "medium" // easy | medium | hard | hunter

// Powerup-Wert aus Bot-Sicht: r_* schaden DIR (top), g_* helfen dem Bot, b_/o_ Utility
const BOT_PU_VALUE = {
    r_reverse: 1.5, r_thick: 1.4, r_robot: 1.4, r_fast: 1.2, r_slow: 1.1,
    g_invisible: 1.2, g_thin: 1.1, g_slow: 1.0, g_side: 1.0, g_robot: 0.9, g_fast: 0.8,
    b_clear: 1.0, b_sides: 0.9, b_more: 0.8, o_random: 0.9,
}

// ---- Schnappschuss des Spur-Canvas pro Frame (schnell) ----------------------
let _botFieldData = null, _botFieldFrame = -1
function botSnapshotField() {
    if (_botFieldFrame === tFrame && _botFieldData) return _botFieldData
    _botFieldData = ctxTH.getImageData(0, 0, h, h).data
    _botFieldFrame = tFrame
    return _botFieldData
}
function botTrailAt(data, x, y) {
    if (x < 0 || y < 0 || x >= h || y >= h) return false
    return data[(y * h + x) * 4 + 3] === 255
}
const _botDist = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by)

// ---- Kernentscheidung: -1 = "dir kleiner", 0 = geradeaus, +1 = "dir groesser"
function botDecide(name) {
    const p = players[name]
    const cfg = BOT_DIFFICULTIES[BOT_LEVEL]
    const data = botSnapshotField()

    const robot = p.powerup.robot !== 0
    const sides = achtung.sides !== 0 || p.powerup.side !== 0
    const mv = moveSpeed * p.powerup.speed
    const half = hitboxSize * p.powerup.size
    const step = robot ? r2d(90) : turnSpeed / Math.pow(p.powerup.size, 0.3)
    const skip = Math.max(2, Math.ceil(half / mv) + 1)
    const H = cfg.lookahead
    const commit = cfg.turnCommit
    const b = borderWidth + half
    const margin = h * 0.14

    // simuliert: Kurve fuer "commit" Frames halten (Roboter: EIN 90-Grad-Knick),
    // danach gerade - und gibt ueberlebte Frames + Endpunkt zurueck
    function sim(a) {
        let sx = p.x, sy = p.y, sd = p.dir, survived = H + 1, ex = sx, ey = sy
        for (let f = 1; f <= H; f++) {
            if (robot) { if (f === 1) sd += a * step } else if (f <= commit) sd += a * step
            sx += Math.cos(sd) * mv; sy += Math.sin(sd) * mv
            if (sides) {
                if (sx < 0) sx += h; else if (sx > h) sx -= h
                if (sy < 0) sy += h; else if (sy > h) sy -= h
            } else if (sx < b || sx > h - b || sy < b || sy > h - b) { survived = f; break }
            if (f > skip && botTrailAt(data, Math.round(sx), Math.round(sy))) { survived = f; break }
            ex = sx; ey = sy
        }
        return { survived, ex, ey }
    }

    // bestes Powerup (naechstes, mit Wert gewichtet)
    let bestPU = null
    if (cfg.powerup > 0 && achtung.powerupsOnScreen.length) {
        let bestD = Infinity
        for (const pu of achtung.powerupsOnScreen) {
            if (!pu) continue
            const d = _botDist(p.x, p.y, pu.xPos, pu.yPos)
            if (d < bestD) { bestD = d; bestPU = { x: pu.xPos, y: pu.yPos, d, value: BOT_PU_VALUE[pu.pow] || 1 } }
        }
    }

    // naechster lebender menschlicher Gegner (fuer's Abschneiden)
    let opp = null
    if (cfg.aggression > 0) {
        let bestD = Infinity
        for (const q in players) {
            if (q === name || players[q].isBot || !players[q].ready || !players[q].alive) continue
            const d = _botDist(p.x, p.y, players[q].x, players[q].y)
            if (d < bestD) { bestD = d; opp = { x: players[q].x, y: players[q].y, dir: players[q].dir, d } }
        }
    }

    // Ausrichtung 0..1: 1 = Aktion zeigt genau aufs Ziel
    function aim(a, tx, ty) {
        const desired = Math.atan2(ty - p.y, tx - p.x)
        const testDir = p.dir + a * step * 6
        const diff = Math.abs(Math.atan2(Math.sin(desired - testDir), Math.cos(desired - testDir)))
        return 1 - diff / Math.PI
    }

    let chosen = 0, bestScore = -Infinity
    for (const a of [-1, 0, 1]) {
        const { survived, ex, ey } = sim(a)
        let score
        if (survived <= H) {
            // unsicher: nur ueberlebte Frames zaehlen (immer schlechter als jede sichere Aktion)
            score = survived
        } else {
            score = 1e6 // sichere Basis - dominiert jede unsichere Aktion
            if (a === 0) score += cfg.straightBias
            // Rand meiden (kein Kleben)
            const bd = Math.min(ex - b, h - b - ex, ey - b, h - b - ey)
            if (bd < margin) score -= cfg.wallAvoid * (1 - bd / margin) * 8
            // Anti-Spiral: zu langes Dauerdrehen in dieselbe Richtung bremsen
            if (a !== 0 && a === (p._botLastTurn || 0) && (p._botStreak || 0) > 18) score -= ((p._botStreak || 0) - 18) * 0.6
            // Powerup ansteuern (nur in Reichweite)
            if (bestPU && bestPU.d < h * 0.6) score += cfg.powerup * bestPU.value * aim(a, bestPU.x, bestPU.y) * 8
            // Gegner abschneiden: auf einen Punkt VOR ihm zusteuern
            if (opp && opp.d < h * 0.65) {
                const lead = Math.min(opp.d * 0.85, mv * H * 0.8)
                const lx = opp.x + Math.cos(opp.dir) * lead, ly = opp.y + Math.sin(opp.dir) * lead
                score += cfg.aggression * aim(a, lx, ly) * 12
            }
        }
        score += Math.random() * 0.001
        if (score > bestScore) { bestScore = score; chosen = a }
    }
    if (Math.random() < cfg.noise) chosen = [-1, 0, 1][Math.floor(Math.random() * 3)]
    return chosen
}

// ---- setzt turnL/turnR passend, beachtet reverse + robot-cooldown + streak ---
function botThink(name) {
    const p = players[name]
    if (!p.isBot || !p.alive) return
    const cfg = BOT_DIFFICULTIES[BOT_LEVEL]
    const robot = p.powerup.robot !== 0

    if (p._botHold === undefined) p._botHold = 0

    if (p._botHold > 0) {
        p._botHold-- // Entscheidung committen (Tasten bleiben gesetzt)
    } else {
        const action = botDecide(name)
        p._botAction = action
        p.turnL = false
        p.turnR = false
        if (action !== 0) {
            const rev = p.powerup.reverse !== 0
            const wantDecrease = action < 0
            if (wantDecrease) rev ? (p.turnR = true) : (p.turnL = true)
            else rev ? (p.turnL = true) : (p.turnR = true)
            p._botHold = robot ? Math.max(3, cfg.react) : cfg.react
        } else {
            p._botHold = cfg.react
        }
    }

    // Drehserie pro Frame mitzaehlen (fuer die Anti-Spiral-Bremse)
    const act = p._botAction || 0
    if (act !== 0 && act === (p._botLastTurn || 0)) p._botStreak = (p._botStreak || 0) + 1
    else p._botStreak = 0
    p._botLastTurn = act
}

// ============================================================================
//  Bot-Verwaltung
// ============================================================================
function addBot(name, level) {
    if (!players[name]) return console.warn("Kein Spieler namens", name)
    if (level) BOT_LEVEL = level
    players[name].isBot = true
    players[name].active = true
    players[name].ready = true
    players[name].keyL = false
    players[name].keyR = false
    const wrap = document.querySelector(`.player_wrapper.${name}`)
    if (wrap) {
        wrap.classList.add("focus")
        const lt = wrap.querySelector(".key_wrapper_left .key_text")
        const rt = wrap.querySelector(".key_wrapper_right .key_text")
        if (lt) lt.textContent = "BOT"
        if (rt) rt.textContent = BOT_LEVEL.toUpperCase()
    }
    console.log(`${name} ist jetzt ein Bot (${BOT_LEVEL}).`)
}

function removeBot(name) {
    if (!players[name]) return
    players[name].isBot = false
    players[name].ready = false
    players[name].active = false
    const wrap = document.querySelector(`.player_wrapper.${name}`)
    if (wrap) {
        wrap.classList.remove("focus")
        const lt = wrap.querySelector(".key_wrapper_left .key_text")
        const rt = wrap.querySelector(".key_wrapper_right .key_text")
        if (lt) lt.textContent = ""
        if (rt) rt.textContent = ""
    }
}

// Schnellwechsel per Konsole:  botLevel('hunter')
function botLevel(level) {
    if (BOT_DIFFICULTIES[level]) BOT_LEVEL = level
    console.log("Bot-Level:", BOT_LEVEL)
}

// ============================================================================
//  Startseiten-UI: Haekchen "Bots" + Anzahl (neben Arcade/Classic)
// ============================================================================

function clearBots() {
    for (const name in players) {
        if (players[name].isBot) removeBot(name)
    }
}

// belegt so viele noch unbelegte Spieler-Slots mit Bots wie eingestellt;
// wird bei jeder Aenderung der Checkbox/Anzahl neu berechnet (idempotent)
function applyBotSelection() {
    const toggle = document.querySelector("#bot_toggle")
    const countSelect = document.querySelector("#bot_count")
    if (!toggle || !countSelect) return

    countSelect.disabled = !toggle.checked
    clearBots()
    if (!toggle.checked) return

    const count = parseInt(countSelect.value, 10) || 0
    const freeNames = Object.keys(players).filter((name) => !players[name].ready)
    freeNames.slice(0, count).forEach((name) => addBot(name))
}

// die Checkbox/Auswahl existiert seit dem Powerup-Menue nicht mehr auf der Startseite -
// Bots bleiben ueber die Konsole nutzbar (addBot('fred'), botLevel('hunter'), ...)
document.querySelector("#bot_toggle")?.addEventListener("change", applyBotSelection)
document.querySelector("#bot_count")?.addEventListener("change", applyBotSelection)
