// ============================================================================
//  Powerup-Menue: zweite Ansicht der Startseite (kein Popup) zum An-/Abwaehlen
//  einzelner Powerups - "Menü" blendet #start_main_view aus und #powerup_page
//  ein, "Zurück" macht es rueckgaengig
// ============================================================================

const POWERUP_INFO = {
    g_slow: { label: "Langsam", group: "self" },
    g_fast: { label: "Schnell", group: "self" },
    g_thin: { label: "Dünn", group: "self" },
    g_robot: { label: "Roboter", group: "self" },
    g_side: { label: "Seitenwechsel", group: "self" },
    g_invisible: { label: "Unsichtbar", group: "self" },
    g_sine: { label: "Sinus-Lenkung", group: "self" },
    g_ghost: { label: "Geist", group: "self" },
    r_slow: { label: "Langsam", group: "enemy" },
    r_fast: { label: "Schnell", group: "enemy" },
    r_thick: { label: "Dick", group: "enemy" },
    r_robot: { label: "Roboter", group: "enemy" },
    r_reverse: { label: "Umkehren", group: "enemy" },
    r_sine: { label: "Sinus-Lenkung", group: "enemy" },
    r_swap: { label: "Platz tauschen", group: "enemy" },
    r_freeze: { label: "Einfrieren", group: "enemy" },
    b_clear: { label: "Spur löschen", group: "global" },
    b_more: { label: "Mehr Powerups", group: "global" },
    b_sides: { label: "Ränder offen", group: "global" },
    b_shrink: { label: "Feld verkleinern", group: "global" },
    b_grow: { label: "Feld vergrößern", group: "global" },
    o_random: { label: "Zufall", group: "random" },
}

const POWERUP_GROUPS = [
    { id: "self", label: "Für dich" },
    { id: "enemy", label: "Gegen Gegner" },
    { id: "global", label: "Global" },
    { id: "random", label: "Zufall" },
]

achtung.enabledPowerups = new Set(achtung.powerups) // alle Powerups starten aktiv

function buildPowerupRow(id) {
    const row = document.createElement("section")
    row.classList.add("powerup_row")

    const iconCell = document.createElement("div")
    const canvas = document.createElement("canvas")
    canvas.width = 54
    canvas.height = 54
    const ctx = canvas.getContext("2d")
    ctx.translate(27, 27)
    drawPowerupBadge(ctx, id, 23)
    iconCell.append(canvas)
    row.append(iconCell)

    const nameCell = document.createElement("p")
    nameCell.textContent = POWERUP_INFO[id].label
    row.append(nameCell)

    const activeCell = document.createElement("div")
    activeCell.classList.add("text_center")
    const checkbox = document.createElement("input")
    checkbox.type = "checkbox"
    checkbox.checked = true
    checkbox.addEventListener("change", () => {
        if (checkbox.checked) achtung.enabledPowerups.add(id)
        else achtung.enabledPowerups.delete(id)
    })
    activeCell.append(checkbox)
    row.append(activeCell)

    return row
}

function buildPowerupPage() {
    const page = document.createElement("div")
    page.id = "powerup_page"
    page.classList.add("hidden")

    const backButton = document.createElement("button")
    backButton.id = "powerup_back_button"
    backButton.textContent = "← Zurück"
    backButton.addEventListener("click", () => {
        page.classList.add("hidden")
        document.querySelector("#start_main_view").classList.remove("hidden")
    })
    page.append(backButton)

    const list = document.createElement("div")
    list.id = "powerup_list"

    const headerRow = document.createElement("section")
    headerRow.classList.add("powerup_row_start")
    const hIcon = document.createElement("p")
    const hName = document.createElement("p")
    hName.textContent = "Powerup"
    const hActive = document.createElement("p")
    hActive.textContent = "Aktiv"
    hActive.classList.add("text_center")
    headerRow.append(hIcon, hName, hActive)
    list.append(headerRow)

    for (const group of POWERUP_GROUPS) {
        const ids = achtung.powerups.filter((id) => POWERUP_INFO[id]?.group === group.id)
        if (!ids.length) continue

        const heading = document.createElement("p")
        heading.textContent = group.label
        heading.classList.add("powerup_group_heading", `powerup_group_${group.id}`)
        list.append(heading)

        for (const id of ids) list.append(buildPowerupRow(id))
    }

    page.append(list)
    startPage.append(page)
    return page
}

const powerupPage = buildPowerupPage()
document.querySelector("#powerup_menu_button")?.addEventListener("click", () => {
    document.querySelector("#start_main_view").classList.add("hidden")
    powerupPage.classList.remove("hidden")
})
