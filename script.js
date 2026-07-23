// players
let players = {
    fred: { powerup: { toClear: [] }, score: 0, ready: false },
    greenlee: { powerup: { toClear: [] }, score: 0, ready: false },
    pinkney: { powerup: { toClear: [] }, score: 0, ready: false },
    bluebell: { powerup: { toClear: [] }, score: 0, ready: false },
    willem: { powerup: { toClear: [] }, score: 0, ready: false },
    greydon: { powerup: { toClear: [] }, score: 0, ready: false },
}

let achtung = {
    gamemode: 1, //  0 = arcade, 1 = classic
    startScreen: true, // are we on the start screen?
    gameRunning: false, // are we playing?
    gameEnded: true, // are noone alive?
    winner: false, // do we have a winner?
    sides: 0, // can all players go out of screen and come out the other side
    clearSides: [], // to clear timeouts if leftover time from last round
    fieldInset: 0, // extra px inset from the border on all sides (b_shrink/b_grow), can go negative
    clearInset: [], // to clear fieldInset timeouts if leftover time from last round
    playing: [], // who's playing
    powerups: [
        "g_slow",
        "g_fast",
        "g_thin",
        "g_robot",
        "g_side",
        "g_invisible",
        "g_sine",
        "g_ghost",
        "r_slow",
        "r_fast",
        "r_thick",
        "r_robot",
        "r_reverse",
        "r_sine",
        "r_swap",
        "r_freeze",
        "b_clear",
        "b_more",
        "b_sides",
        "b_shrink",
        "b_grow",
        "o_random",
    ],
    powerupsOnScreen: [], // what powerups are on screen now
}

// variables
let canvasID,
    UIcanvas = document.querySelector("#ui_canvas"), // ui canvas
    dotsCanvas = document.querySelector("#dots_canvas"), // canvas for player dots
    trailsHitboxCanvas = document.querySelector("#trails_hitbox_canvas"), // canvas for player trails
    powerupVisualCanvas = document.querySelector("#powerup_visual_canvas"), // canvas for powerup icons
    powerupHitboxCanvas = document.querySelector("#powerup_hitbox_canvas"), // canvas for powerup hitboxes
    ctxUI = UIcanvas.getContext("2d"),
    ctxDO = dotsCanvas.getContext("2d"),
    ctxTH = trailsHitboxCanvas.getContext("2d"),
    ctxPV = powerupVisualCanvas.getContext("2d"),
    ctxPH = powerupHitboxCanvas.getContext("2d"),
    yellow = getComputedStyle(document.documentElement).getPropertyValue(`--yellow`), // colors
    green = getComputedStyle(document.documentElement).getPropertyValue(`--greenlee`),
    greent = getComputedStyle(document.documentElement).getPropertyValue(`--greenlee-t`),
    red = getComputedStyle(document.documentElement).getPropertyValue(`--fred`),
    redt = getComputedStyle(document.documentElement).getPropertyValue(`--fred-t`),
    blue = getComputedStyle(document.documentElement).getPropertyValue(`--blue`),
    bluet = getComputedStyle(document.documentElement).getPropertyValue(`--blue-t`),
    tFrame = 0, // cur frame in draw
    powerupProb = 0.005, // in percent
    bridgeProb = 0.005, // in percent
    bridgeSize = 10, // in frames
    turnSpeed = 0.06, // in radians per frame
    sineTurnDuration = 240, // in frames; duration of g_sine/r_sine steering-intensity wave (4s @ 60 steps/sec)
    w,
    h,
    w100th,
    h100th,
    moveSpeed,
    playerSize,
    hitboxSize,
    borderWidth,
    iconSize // to be set in newSize()

// when resizing
window.addEventListener("resize", newSize)

function newSize() {
    // update canvas sizes and variable sizes to fit new size
    const dpr = Math.min(window.devicePixelRatio, 2)
    w = Math.round(UIcanvas.getBoundingClientRect().width * dpr)
    h = Math.round(UIcanvas.getBoundingClientRect().height * dpr)
    UIcanvas.width = w
    UIcanvas.height = h
    dotsCanvas.width = w
    dotsCanvas.height = h
    trailsHitboxCanvas.width = w
    trailsHitboxCanvas.height = h
    powerupVisualCanvas.width = w
    powerupVisualCanvas.height = h
    powerupHitboxCanvas.width = w
    powerupHitboxCanvas.height = h
    w100th = w / 100 // 1 percent of canvas width
    h100th = h / 100 // 1 percent of canvas height
    moveSpeed = w100th * 0.18 // in pixels per frame
    playerSize = w100th * 0.7 // in pixels
    hitboxSize = playerSize / 1.8 // in pixels
    borderWidth = w100th / 2 // in pixels
    iconSize = w100th * 2 // in pixels

    init() // restart
}

function init() {
    achtung.powerupsOnScreen = [] // clear powerups on screen
    for (let i = 0; i < achtung.clearSides.length; i++) clearTimeout(achtung.clearSides[i]) // clear timeouts if sides powerup leftover time from last round
    achtung.clearSides = []
    achtung.sides = 0 // reset sides
    for (let i = 0; i < achtung.clearInset.length; i++) clearTimeout(achtung.clearInset[i]) // clear timeouts if shrink/grow leftover time from last round
    achtung.clearInset = []
    achtung.fieldInset = 0 // reset field size

    for (const player in players) {
        // clear timeout if powerup leftover time from last round
        for (let i = 0; i < players[player].powerup.toClear.length; i++) {
            clearTimeout(players[player].powerup.toClear[i])
        }

        // reset players object to default values before starting a new round
        players[player].x = 0
        players[player].y = 0
        players[player].dir = 0
        players[player].turnL = false
        players[player].turnR = false
        players[player].color = getComputedStyle(document.documentElement).getPropertyValue(`--${player}`) // colors from css :root object
        const rgbMatch = players[player].color.match(/rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)/i)
        players[player].colorRGB = rgbMatch ? [+rgbMatch[1], +rgbMatch[2], +rgbMatch[3]] : [255, 255, 0] // for g_ghost's own-trail check
        players[player].alive = true
        players[player].winner = false
        players[player].bridge = false
        players[player].bridgeFrame = 0
        players[player].powerup = {} // contains powerup values
        players[player].powerup.size = 1
        players[player].powerup.robot = 0
        players[player].powerup.reverse = 0
        players[player].powerup.speed = 1
        players[player].powerup.invisible = 0
        players[player].powerup.side = 0
        players[player].powerup.ghost = 0
        players[player].powerup.freeze = 0
        players[player].powerup.powerupArray = []
        players[player].powerup.toClear = [] // to clear timeout at the end of rounds if leftover time
        players[player].powerup.sineStart = null // frame the sine steering wave started, null if inactive
    }

    // clear everything
    ctxTH.clearRect(0, 0, w, h)
    ctxUI.clearRect(0, 0, w, h)
    ctxDO.clearRect(0, 0, w, h)
    ctxPH.clearRect(0, 0, w, h)
    ctxPV.clearRect(0, 0, w, h)

    // draw yellow border
    ctxDO.lineWidth = borderWidth
    ctxDO.strokeStyle = yellow
    ctxDO.strokeRect(borderWidth / 2, borderWidth / 2, h - borderWidth, h - borderWidth)

    calcRandomStartPos() // calc random start positions
    calcRandomStartDir() // calc random start directions
    drawGameUI() // draw ui
    drawStart() // draw players at start so they can see where they're going
}

document.addEventListener("keydown", (e) => {
    // update players turning to true if turning
    for (const player in players) {
        if (!players[player].alive) continue // if player not alive; skip
        if (e.code == players[player].keyL) {
            players[player].turnL = true
        }
        if (e.code == players[player].keyR) {
            players[player].turnR = true
        }
    }

    // if keydown == escape, go to start page
    if (e.code == "Escape") {
        achtung.startScreen = true
        achtung.gameEnded = true
        achtung.gameRunning = false
        startPage.style.display = "block"
        window.cancelAnimationFrame(canvasID)
        for (const player in players) {
            players[player].score = 0
        }
        init()
    }
})
document.addEventListener("keyup", (e) => {
    // update players turning to false when keyup
    for (const player in players) {
        if (!players[player].alive) continue // if player not alive; skip
        if (e.code == players[player].keyL) {
            players[player].turnL = false
        }
        if (e.code == players[player].keyR) {
            players[player].turnR = false
        }
    }
})
document.addEventListener("keypress", (e) => {
    if (e.code == "Space") {
        pressSpace()
    }
})

function pressSpace() {
    if (achtung.startScreen) applyBotSelection() // fill any unclaimed slots with bots per the checkbox/count

    let playingC = 0
    for (const player in players) {
        if (players[player].ready) playingC++
    }

    if (achtung.startScreen) {
        if (playingC >= 2) {
            // start game
            for (const player in players) {
                if (players[player].active && !players[player].ready) {
                    resetPlayer(document.querySelector(`.player_wrapper.${player}`))
                }
            }
            achtung.startScreen = false
            achtung.gameEnded = true
            startPage.style.display = "none"
            init()
        }
    }

    if (!achtung.gameEnded) {
        if (!achtung.gameRunning) {
            // resume game
            achtung.gameRunning = true
            window.requestAnimationFrame(draw)
        } else {
            // pause game
            achtung.gameRunning = false
            window.cancelAnimationFrame(canvasID)
        }
    } else {
        // restart game
        if (achtung.winner) {
            achtung.winner = false
            achtung.startScreen = true
            achtung.gameEnded = true
            achtung.gameRunning = false
            startPage.style.display = "block"
            window.cancelAnimationFrame(canvasID)
            for (const player in players) {
                players[player].score = 0
            }
            init()
        } else {
            if (playingC >= 2) {
                achtung.gameEnded = false
                achtung.gameRunning = false
                init()
            }
        }
    }
}

// draw start position so players know where they're going
function drawStart() {
    for (const player in players) {
        if (!players[player].ready) continue

        // draw player dot
        ctxDO.fillStyle = yellow
        ctxDO.beginPath()
        ctxDO.arc(players[player].x, players[player].y, (playerSize / 2 - 0.1) * players[player].powerup.size, 0, r2d(360), true)
        ctxDO.fill()

        // draw player trail
        ctxTH.fillStyle = players[player].color
        ctxTH.save()
        ctxTH.translate(players[player].x, players[player].y)
        ctxTH.rotate(players[player].dir - r2d(270))
        ctxTH.fillRect(
            (-playerSize / 2) * players[player].powerup.size,
            0,
            playerSize * players[player].powerup.size,
            playerSize * 2 * players[player].powerup.size
        )
        ctxTH.restore()
    }
}

// main loop
// the game logic is tuned for 60 steps per second, but requestAnimationFrame fires at the
// display refresh rate (60/120/144 Hz...) - on fast monitors the whole game ran up to 2.4x
// too fast. accumulate real time and only advance the simulation at 60 steps per second
let lastFrameTime = 0,
    frameAcc = 0
const frameStep = 1000 / 60

function draw(now = performance.now()) {
    canvasID = window.requestAnimationFrame(draw) // to pause: cancelAnimationFrame(CanvasID)

    let dt = now - lastFrameTime
    lastFrameTime = now
    if (dt < 0 || dt > 100) dt = frameStep // first frame, or resuming after pause/tab switch
    frameAcc += dt
    if (frameAcc < frameStep) return // display is faster than 60 Hz; wait for the next frame
    frameAcc -= frameStep
    if (frameAcc > frameStep) frameAcc = frameStep // slow machine; don't pile up debt

    tFrame++ // increment tFrame

    // clear
    ctxTH.clearRect(h, 0, w - h, h)
    ctxDO.clearRect(0, 0, w, h)
    ctxDO.fillStyle = "#000000"
    ctxDO.fillRect(h, 0, w - h, h)

    // draw yellow border - inset by achtung.fieldInset so b_shrink/b_grow are visible, not just deadly
    const inset = achtung.fieldInset
    ctxDO.lineWidth = borderWidth
    ctxDO.strokeStyle = "#000000"
    ctxDO.strokeRect(borderWidth / 2 + inset, borderWidth / 2 + inset, h - borderWidth - inset * 2, h - borderWidth - inset * 2)
    if (achtung.sides != 0) ctxDO.strokeStyle = `rgba(255, 255, 0, ${Math.abs((tFrame % 40) - 20) / 20})`
    else ctxDO.strokeStyle = yellow // if sides, border flickers
    ctxDO.strokeRect(borderWidth / 2 + inset, borderWidth / 2 + inset, h - borderWidth - inset * 2, h - borderWidth - inset * 2)

    // spawn new powerup if arcade mode and math.random() < powerup probability
    if (achtung.gamemode == 1) if (Math.random() < powerupProb) powerupSpawner()

    // loop through players and draw them
    for (const player in players) {
        if (!players[player].ready) continue // continue loop if player not playing

        // player pos
        let prevprevPosX = players[player].x - mathCos(players[player].dir) * moveSpeed * players[player].powerup.speed,
            prevprevPosY = players[player].y - mathSin(players[player].dir) * moveSpeed * players[player].powerup.speed,
            prevPosX = players[player].x,
            prevPosY = players[player].y,
            nextPosX = players[player].x + mathCos(players[player].dir) * moveSpeed * players[player].powerup.speed,
            nextPosY = players[player].y + mathSin(players[player].dir) * moveSpeed * players[player].powerup.speed

        // draw player dot
        if (players[player].powerup.reverse == 0) {
            if (players[player].powerup.side == 0) {
                ctxDO.fillStyle = yellow
            } else ctxDO.fillStyle = `rgba(255, 255, 0, ${Math.abs((tFrame % 40) - 20) / 20})` // flicker dot if side powerup
        } else {
            if (players[player].powerup.side == 0) {
                ctxDO.fillStyle = blue
            } else ctxDO.fillStyle = `rgba(0, 0, 255, ${Math.abs((tFrame % 40) - 20) / 20})` // flicker dot if side powerup
        }
        if (players[player].powerup.robot == 0) {
            // draw dot if normal
            ctxDO.beginPath()
            ctxDO.arc(nextPosX, nextPosY, (playerSize / 2) * players[player].powerup.size, 0, r2d(360), true)
            ctxDO.fill()
        } else {
            // draw square if robot
            ctxDO.save()
            ctxDO.translate(nextPosX, nextPosY)
            ctxDO.rotate(players[player].dir - r2d(270))
            ctxDO.fillRect(
                (-playerSize / 2) * players[player].powerup.size,
                (-playerSize / 2) * players[player].powerup.size,
                playerSize * players[player].powerup.size,
                playerSize * players[player].powerup.size
            )
            ctxDO.restore()
        }

        if (!players[player].alive) continue // continue if player not alive (drawing dot is above, so player dot will still be drawn even if dead)

        if (players[player].isBot) botThink(player) // let bot.js set turnL/turnR for this frame

        // update player turning
        // powerup.sineStart (g_sine/r_sine) scales turn intensity by a sine wave between
        // 0.5x and 2x over sineTurnDuration frames - it only scales magnitude, never the sign,
        // so turnL/turnR (and reverse) always steer the same way, keys are never swapped
        let turnFactor = 1
        if (players[player].powerup.sineStart !== null) {
            const sineElapsed = tFrame - players[player].powerup.sineStart
            if (sineElapsed < sineTurnDuration) {
                turnFactor = Math.pow(2, Math.sin((sineElapsed / sineTurnDuration) * Math.PI * 2))
            } else {
                players[player].powerup.sineStart = null
            }
        }

        if (players[player].powerup.freeze == 0) {
            // r_freeze blocks steering entirely for its duration - held keys just resume once it ends
            if (players[player].powerup.robot == 0) {
                // if normal
                if (players[player].turnL) {
                    if (players[player].powerup.reverse == 0) players[player].dir -= (turnSpeed * turnFactor) / Math.pow(players[player].powerup.size, 0.3)
                    else players[player].dir += (turnSpeed * turnFactor) / Math.pow(players[player].powerup.size, 0.3)
                }
                if (players[player].turnR) {
                    if (players[player].powerup.reverse == 0) players[player].dir += (turnSpeed * turnFactor) / Math.pow(players[player].powerup.size, 0.3)
                    else players[player].dir -= (turnSpeed * turnFactor) / Math.pow(players[player].powerup.size, 0.3)
                }
            } else {
                // if robot
                if (players[player].turnL) {
                    players[player].turnL = false
                    if (players[player].powerup.reverse == 0) players[player].dir -= r2d(90) * turnFactor
                    else players[player].dir += r2d(90) * turnFactor
                }
                if (players[player].turnR) {
                    players[player].turnR = false
                    if (players[player].powerup.reverse == 0) players[player].dir += r2d(90) * turnFactor
                    else players[player].dir -= r2d(90) * turnFactor
                }
            }
        }

        // update player position
        prevPosX = players[player].x
        prevPosY = players[player].y
        players[player].x = nextPosX
        players[player].y = nextPosY

        // check for player inside playing field
        if (achtung.sides != 0 || players[player].powerup.side != 0) {
            // player has side powerup or achtung.sides, players can move out of canvas.
            // shift ALL reference points by the same offset so the trail continues seamlessly
            // on the other side; hard-setting only one coordinate drew a perpendicular blob
            // of trail on the entry edge which instantly killed the wrapping player (issue #2)
            if (players[player].x < 0) {
                players[player].x += h
                prevPosX += h
                prevprevPosX += h
            }
            if (players[player].x > h) {
                players[player].x -= h
                prevPosX -= h
                prevprevPosX -= h
            }
            if (players[player].y < 0) {
                players[player].y += h
                prevPosY += h
                prevprevPosY += h
            }
            if (players[player].y > h) {
                players[player].y -= h
                prevPosY -= h
                prevprevPosY -= h
            }
        } else {
            if (
                // if not, player dead - achtung.fieldInset (b_shrink/b_grow) moves this boundary for everyone
                players[player].x < borderWidth + hitboxSize + achtung.fieldInset ||
                players[player].x > h - borderWidth - hitboxSize - achtung.fieldInset ||
                players[player].y < borderWidth + hitboxSize + achtung.fieldInset ||
                players[player].y > h - borderWidth - hitboxSize - achtung.fieldInset
            ) {
                givePoints(players[player])
                continue
            }
        }

        // insert bridge
        if (!players[player].bridge) {
            // if not already bridge
            if (Math.random() < bridgeProb) {
                // if math.random() less than prob for bridge
                players[player].bridge = true
            }
            players[player].bridgeFrame = tFrame // what frame did bridge start
        }
        if (players[player].bridgeFrame < tFrame - (bridgeSize / players[player].powerup.speed) * players[player].powerup.size) {
            // stop bridge when bridgeSize frame has passed
            players[player].bridge = false
        }

        // draw player trail; don't draw if bridge or invisible
        if (!players[player].bridge && players[player].powerup.invisible == 0) {
            ctxTH.strokeStyle = players[player].color
            ctxTH.lineWidth = playerSize * players[player].powerup.size
            ctxTH.beginPath()
            if (players[player].powerup.robot != 0) {
                ctxTH.lineCap = "round"
                ctxTH.moveTo(prevPosX, prevPosY)
            } else {
                ctxTH.lineCap = "butt"
                ctxTH.moveTo(prevprevPosX, prevprevPosY)
            }
            ctxTH.lineTo(players[player].x, players[player].y)
            ctxTH.stroke()
        }

        // check collision
        // the robot trail is drawn with lineCap "round", so it sticks out lineWidth/2 past the
        // player position - that leaves the front sample less than 1px of clearance and rounding
        // makes it randomly hit the carrier's own fresh trail. add 2px clearance while robot is
        // active so the square powerup can't kill its own carrier (issue #1)
        const frontDist = hitboxSize * players[player].powerup.size + (players[player].powerup.robot != 0 ? 2 : 0)
        const pxFront = Math.round(players[player].x + mathCos(players[player].dir) * frontDist)
        const pyFront = Math.round(players[player].y + mathSin(players[player].dir) * frontDist)
        const pxFront2 = Math.round(players[player].x + mathCos(players[player].dir))
        const pyFront2 = Math.round(players[player].y + mathSin(players[player].dir))
        const pxLeft = Math.round(players[player].x + mathCos(players[player].dir - r2d(55)) * hitboxSize * players[player].powerup.size)
        const pyLeft = Math.round(players[player].y + mathSin(players[player].dir - r2d(55)) * hitboxSize * players[player].powerup.size)
        const pxRight = Math.round(players[player].x + mathCos(players[player].dir + r2d(55)) * hitboxSize * players[player].powerup.size)
        const pyRight = Math.round(players[player].y + mathSin(players[player].dir + r2d(55)) * hitboxSize * players[player].powerup.size)

        const imgDataFrontTH = ctxTH.getImageData(pxFront, pyFront, 1, 1).data
        const imgDataFrontPH = ctxPH.getImageData(pxFront, pyFront, 1, 1).data
        const imgDataFront2TH = ctxTH.getImageData(pxFront2, pyFront2, 1, 1).data
        const imgDataFront2PH = ctxPH.getImageData(pxFront2, pyFront2, 1, 1).data
        const imgDataLeftTH = ctxTH.getImageData(pxLeft, pyLeft, 1, 1).data
        const imgDataLeftPH = ctxPH.getImageData(pxLeft, pyLeft, 1, 1).data
        const imgDataRightTH = ctxTH.getImageData(pxRight, pyRight, 1, 1).data
        const imgDataRightPH = ctxPH.getImageData(pxRight, pyRight, 1, 1).data

        // uncomment to visualize hitbox
        // ctxDO.fillStyle = "#ffffff"
        // ctxDO.fillRect(pxFront, pyFront, 1, 1)
        // ctxDO.fillRect(pxFront2, pyFront2, 1, 1)
        // ctxDO.fillRect(pxLeft, pyLeft, 1, 1)
        // ctxDO.fillRect(pxRight, pyRight, 1, 1)

        // check collision for every powerup on screen
        for (let i = 0; i < achtung.powerupsOnScreen.length; i++) {
            if (
                (imgDataFrontPH[2] == i * 3 + 1 && imgDataFrontPH[1] == i * 3 + 2 && imgDataFrontPH[0] == i * 3 + 3) ||
                (imgDataLeftPH[2] == i * 3 + 1 && imgDataLeftPH[1] == i * 3 + 2 && imgDataLeftPH[0] == i * 3 + 3) ||
                (imgDataRightPH[2] == i * 3 + 1 && imgDataRightPH[1] == i * 3 + 2 && imgDataRightPH[0] == i * 3 + 3)
            ) {
                let powName = achtung.powerupsOnScreen[i].pow
                players[player].powerup.powerupArray.push(powName)

                // remove powerup from screen
                achtung.powerupsOnScreen.splice(i, 1)

                // do powerup
                doPowerups(player, players[player].powerup.powerupArray.length - 1)

                // draw powerup
                powerupDraw()
            }
        }

        if (!players[player].bridge) {
            // don't check collision if making bridge
            if (players[player].powerup.invisible == 0) {
                // don't check if invisible
                // g_ghost: a trail pixel only counts as a hit if it's NOT this player's own color -
                // still dies to walls and everyone else's trail, just not their own
                const own = players[player].colorRGB
                const isOwnTrail = (d) => players[player].powerup.ghost != 0 && d[3] == 255 && d[0] == own[0] && d[1] == own[1] && d[2] == own[2]
                if (players[player].powerup.robot == 0) {
                    // check alpha value of pixels front, front2, left, right
                    if (
                        (imgDataFrontTH[3] == 255 && !isOwnTrail(imgDataFrontTH)) ||
                        (imgDataFront2TH[3] == 255 && !isOwnTrail(imgDataFront2TH)) ||
                        (imgDataLeftTH[3] == 255 && !isOwnTrail(imgDataLeftTH)) ||
                        (imgDataRightTH[3] == 255 && !isOwnTrail(imgDataRightTH))
                    ) {
                        givePoints(players[player])
                        continue
                    }
                } else {
                    if (imgDataFrontTH[3] == 255 && !isOwnTrail(imgDataFrontTH)) {
                        // if robot only check alpha value of front
                        givePoints(players[player])
                        continue
                    }
                }
            }
        }
    }

    // drawGameUI()
    checkGameState()
}

// check game stats
function checkGameState() {
    // how many are alive?
    let alive = 0
    for (const player in players) {
        if (players[player].alive && players[player].ready) {
            alive++
        }
    }

    // if all dead
    if (alive <= 1) {
        // IMPORTANT - change back to 1 -------------------------------------------------------------------------------------------------------------------------------
        window.cancelAnimationFrame(canvasID)
        achtung.gameEnded = true
    }

    // did someone win?
    if (achtung.gameEnded) {
        if (achtung.scoreArray[achtung.scoreArray.length - 1][1] >= achtung.pointGoal) {
            if (achtung.scoreArray[achtung.scoreArray.length - 1][1] - achtung.scoreArray[achtung.scoreArray.length - 2][1] > 1) {
                let p = achtung.scoreArray[achtung.scoreArray.length - 1][0]
                // console.log(p + " wins the game")
                achtung.winner = true

                // draw winner screen
                for (const player in players) {
                    if (player == p) {
                        ctxUI.fillStyle = players[player].color.replace("rgb", "rgba").replace(")", ", 0.3)")
                        ctxUI.fillRect(20 * h100th, 32 * h100th, h - 40 * h100th, h - 64 * h100th)

                        ctxUI.lineWidth = borderWidth
                        ctxUI.strokeStyle = players[player].color
                        ctxUI.strokeRect(20 * h100th, 32 * h100th, h - 40 * h100th, h - 64 * h100th)

                        ctxUI.textBaseline = "middle"
                        ctxUI.fillStyle = players[player].color
                        ctxUI.textAlign = "center"
                        ctxUI.font = `${w100th * 6}px 'Sarabun'`
                        ctxUI.fillText("Konec hry", h / 2, h / 2 - h100th * 5) // the legendary "konec hry"
                        ctxUI.font = `${w100th * 4}px 'Sarabun'`
                        ctxUI.fillText(`${capitalize(player)} wins!`, h / 2, h / 2 + h100th * 5)
                    }
                }
            } else {
                // two players are within 1 point; continue playing
                // console.log("play on")
            }
        }
    }
}

// updates points for players
function givePoints(p) {
    p.alive = false
    for (const player in players) {
        if (!players[player].ready) continue
        if (p != players[player] && players[player].alive) {
            players[player].score++
            drawGameUI()
        }
    }
}

// draws game ui
const drawGameUI = () => {
    ctxUI.textBaseline = "alphabetic"
    ctxUI.clearRect(h, 0, w - h, h)
    ctxUI.fillStyle = "#000000"
    ctxUI.fillRect(h, 0, w - h, h)

    // sort players
    achtung.scoreArray = []
    for (const player in players) {
        if (!players[player].ready) continue
        achtung.scoreArray.push([player, players[player].score])
    }
    achtung.scoreArray.sort((a, b) => a[1] - b[1])

    // draw top text
    achtung.pointGoal = (achtung.scoreArray.length - 1) * 10 // // // // // // // change back to * 10 -------------------------------------------------------------------------------------
    let UIcenter = +h + (w - h) / 2
    ctxUI.fillStyle = "#FFFFFF"
    ctxUI.textAlign = "center"
    ctxUI.letterSpacing = `${w100th * 0.06}px`
    ctxUI.font = `${w100th * 3}px 'Sarabun'`
    ctxUI.fillText("Race to", UIcenter, w100th * 5)
    ctxUI.font = `${w100th * 12}px 'Sarabun'`
    ctxUI.fillText(achtung.pointGoal, UIcenter, w100th * 15)
    ctxUI.font = `${w100th * 2}px 'Sarabun'`
    ctxUI.fillText("2 point difference", UIcenter, w100th * 19)

    // draw player names and score
    ctxUI.font = `${w100th * 3}px 'Sarabun'`
    let playerYOffset = w100th * 32

    for (let i = achtung.scoreArray.length - 1; i >= 0; i--) {
        let p = achtung.scoreArray[i][0]
        ctxUI.fillStyle = players[p].color
        ctxUI.textAlign = "start"
        ctxUI.fillText(capitalize(p), +h + w100th * 2, playerYOffset)
        ctxUI.textAlign = "end"
        ctxUI.fillText(players[p].score, +w - w100th * 2, playerYOffset)
        playerYOffset += w100th * 5
    }

    // draw space to continue text
    ctxUI.fillStyle = "#FFFFFF"
    ctxUI.textAlign = "center"
    ctxUI.font = `${w100th * 2}px 'Sarabun'`
    ctxUI.fillText("SPACE to play", UIcenter, +h - w100th * 6)
    ctxUI.fillText("ESCAPE to quit", UIcenter, +h - w100th * 3)
}

// executes powerups
function doPowerups(puPlayer, index) {
    let gTimeout = 8000
    let rTimeout = 5000
    let freezeTimeout = 1000 // r_freeze is stronger than r_reverse, so it stays much shorter
    let powName = players[puPlayer].powerup.powerupArray[index]

    // powerup starts
    if (powName == "o_random") {
        const pool = achtung.powerups.filter((p) => achtung.enabledPowerups.has(p) && p != "o_random")
        if (pool.length) powName = pool[Math.floor(Math.random() * pool.length)]
    }
    if (powName == "g_slow") {
        players[puPlayer].powerup.speed *= 0.5
        players[puPlayer].powerup.toClear.push(setTimeout(() => (players[puPlayer].powerup.speed *= 2), gTimeout))
    }
    if (powName == "g_fast") {
        players[puPlayer].powerup.speed *= 2
        players[puPlayer].powerup.toClear.push(setTimeout(() => (players[puPlayer].powerup.speed *= 0.5), gTimeout))
    }
    if (powName == "g_thin") {
        players[puPlayer].powerup.size *= 0.5
        players[puPlayer].powerup.toClear.push(setTimeout(() => (players[puPlayer].powerup.size *= 2), gTimeout))
    }
    if (powName == "g_robot") {
        players[puPlayer].powerup.robot++
        players[puPlayer].powerup.toClear.push(setTimeout(() => players[puPlayer].powerup.robot--, gTimeout))
    }
    if (powName == "g_side") {
        players[puPlayer].powerup.side++
        players[puPlayer].powerup.toClear.push(setTimeout(() => players[puPlayer].powerup.side--, gTimeout))
    }
    if (powName == "g_invisible") {
        players[puPlayer].powerup.invisible++
        players[puPlayer].powerup.toClear.push(setTimeout(() => players[puPlayer].powerup.invisible--, gTimeout))
    }
    if (powName == "g_sine") {
        players[puPlayer].powerup.sineStart = tFrame
    }
    if (powName == "g_ghost") {
        players[puPlayer].powerup.ghost++
        players[puPlayer].powerup.toClear.push(setTimeout(() => players[puPlayer].powerup.ghost--, gTimeout))
    }
    if (powName == "r_slow") {
        for (const otherPlayers in players) {
            if (otherPlayers != puPlayer) {
                players[otherPlayers].powerup.speed *= 0.5
                players[otherPlayers].powerup.toClear.push(setTimeout(() => (players[otherPlayers].powerup.speed *= 2), rTimeout))
            }
        }
    }
    if (powName == "r_fast") {
        for (const otherPlayers in players) {
            if (otherPlayers != puPlayer) {
                players[otherPlayers].powerup.speed *= 2
                players[otherPlayers].powerup.toClear.push(setTimeout(() => (players[otherPlayers].powerup.speed *= 0.5), rTimeout))
            }
        }
    }
    if (powName == "r_thick") {
        for (const otherPlayers in players) {
            if (otherPlayers != puPlayer) {
                players[otherPlayers].powerup.size *= 2
                players[otherPlayers].powerup.toClear.push(setTimeout(() => (players[otherPlayers].powerup.size *= 0.5), rTimeout))
            }
        }
    }
    if (powName == "r_robot") {
        for (const otherPlayers in players) {
            if (otherPlayers != puPlayer) {
                players[otherPlayers].powerup.robot++
                players[otherPlayers].powerup.toClear.push(setTimeout(() => players[otherPlayers].powerup.robot--, rTimeout))
            }
        }
    }
    if (powName == "r_reverse") {
        for (const otherPlayers in players) {
            if (otherPlayers != puPlayer) {
                players[otherPlayers].powerup.reverse++
                players[otherPlayers].powerup.toClear.push(setTimeout(() => players[otherPlayers].powerup.reverse--, rTimeout))
            }
        }
    }
    if (powName == "r_sine") {
        for (const otherPlayers in players) {
            if (otherPlayers != puPlayer) {
                players[otherPlayers].powerup.sineStart = tFrame
            }
        }
    }
    if (powName == "r_swap") {
        // teleports puPlayer and the nearest other living player to swap places (facing direction stays each player's own)
        let target = null,
            bestD = Infinity
        for (const q in players) {
            if (q == puPlayer || !players[q].ready || !players[q].alive) continue
            const d = Math.hypot(players[q].x - players[puPlayer].x, players[q].y - players[puPlayer].y)
            if (d < bestD) {
                bestD = d
                target = q
            }
        }
        if (target) {
            const tx = players[target].x,
                ty = players[target].y
            players[target].x = players[puPlayer].x
            players[target].y = players[puPlayer].y
            players[puPlayer].x = tx
            players[puPlayer].y = ty
        }
    }
    if (powName == "r_freeze") {
        for (const otherPlayers in players) {
            if (otherPlayers != puPlayer) {
                players[otherPlayers].powerup.freeze++
                players[otherPlayers].powerup.toClear.push(setTimeout(() => players[otherPlayers].powerup.freeze--, freezeTimeout))
            }
        }
    }
    if (powName == "b_clear") {
        ctxTH.clearRect(0, 0, h, h)
    }
    if (powName == "b_more") {
        setTimeout(powerupSpawner, 100)
        setTimeout(powerupSpawner, 200)
        setTimeout(powerupSpawner, 300)
    }
    if (powName == "b_sides") {
        achtung.sides++
        achtung.clearSides.push(setTimeout(() => achtung.sides--, gTimeout))
    }
    if (powName == "b_shrink") {
        achtung.fieldInset = Math.min(achtung.fieldInset + h * 0.08, h * 0.3)
        achtung.clearInset.push(setTimeout(() => (achtung.fieldInset -= h * 0.08), gTimeout))
    }
    if (powName == "b_grow") {
        achtung.fieldInset = Math.max(achtung.fieldInset - h * 0.05, -hitboxSize * 0.8)
        achtung.clearInset.push(setTimeout(() => (achtung.fieldInset += h * 0.05), gTimeout))
    }
}

// updates the achtung object with data of a new powerup
function powerupSpawner() {
    if (achtung.powerupsOnScreen.length > 30) return
    const pool = achtung.powerups.filter((p) => achtung.enabledPowerups.has(p))
    if (!pool.length) return
    let newPow = Math.floor(Math.random() * pool.length),
        spawnX = Math.floor(Math.random() * h),
        spawnY = Math.floor(Math.random() * h),
        powup = pool[newPow]
    // powup = "r_reverse" //  apklsdjalskdjalksdjlaksjdlakfjlæanæoæiuanweifupnaweifunaewæfnakdnfkalsjdfnklajsdfnkaljdnfklajnsdfklajnsdfkajdsnfkajdsnflakjdnf

    achtung.powerupsOnScreen[achtung.powerupsOnScreen.length] = {}
    achtung.powerupsOnScreen[achtung.powerupsOnScreen.length - 1].pow = powup
    achtung.powerupsOnScreen[achtung.powerupsOnScreen.length - 1].xPos = spawnX
    achtung.powerupsOnScreen[achtung.powerupsOnScreen.length - 1].yPos = spawnY

    // powerupIndex++

    powerupDraw()
}

// draws powerups to canvas
function powerupDraw() {
    ctxPV.clearRect(0, 0, w, h)
    ctxPH.clearRect(0, 0, w, h)

    for (let i = 0; i < achtung.powerupsOnScreen.length; i++) {
        if (achtung.powerupsOnScreen[i] == 0) continue

        let pow = achtung.powerupsOnScreen[i].pow,
            spawnX = achtung.powerupsOnScreen[i].xPos,
            spawnY = achtung.powerupsOnScreen[i].yPos

        // draw hitbox
        ctxPH.fillStyle = `rgba(${i * 3 + 3}, ${i * 3 + 2}, ${i * 3 + 1}, 1)`
        ctxPH.beginPath()
        ctxPH.arc(spawnX, spawnY, iconSize, 0, r2d(360), false)
        ctxPH.fill()

        ctxPV.save()
        ctxPV.translate(spawnX, spawnY)
        drawPowerupBadge(ctxPV, pow, iconSize)
        ctxPV.restore()
    }
}

// calc random start direction
function calcRandomStartDir() {
    for (const player in players) {
        players[player].dir = round100(Math.random() * Math.PI * 2)
    }
}

// calc random start position x and y
function calcRandomStartPos() {
    for (const player in players) {
        players[player].x = map(calcRandomInt(h), 0, h, borderWidth * 10, h - borderWidth * 10) // map to avoid instant death
        players[player].y = map(calcRandomInt(h), 0, h, borderWidth * 10, h - borderWidth * 10) // map to avoid instant death
    }
}

// capitalize string
const capitalize = (s) => (typeof s === "string" ? s.charAt(0).toUpperCase() + s.slice(1) : "")

// returns n rounded to .00
const round100 = (n) => Math.round(n * 100) / 100

// returns n round mathCos and mathSin
const mathCos = (n) => round100(Math.cos(n))
const mathSin = (n) => round100(Math.sin(n))

// returns pixel index for alpha value in raw pixel data string
const getAlphaIndexForCoord = (x, y, width) => y * (width * 4) + x * 4 + 3

// returns radians from degree input
const r2d = (deg) => ((Math.PI * 2) / 360) * deg

// returns random int from 0 to n
const calcRandomInt = (int) => Math.floor(Math.random() * int)

// returns n mapped from start1-stop1 to start2-stop2
const map = (n, start1, stop1, start2, stop2) => ((n - start1) / (stop1 - start1)) * (stop2 - start2) + start2

newSize() //  calc initial values
init() //  start init
