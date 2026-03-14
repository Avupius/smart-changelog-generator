// ─── Modal-Funktionen ─────────────────────────────────────────────────────
function openEvalModal() {
    document.getElementById("eval-modal").classList.remove("hidden");
    document.body.style.overflow = "hidden";
}

function closeEvalModal() {
    document.getElementById("eval-modal").classList.add("hidden");
    document.body.style.overflow = "";
}

// Schließe Modal bei Escape-Taste
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeEvalModal();
});

// Metadaten für alle 6 Kategorien: Labels, Farben und Balken-Farben
const CATEGORY_META = {
    feature: { label: "✨ Feature", color: "text-emerald-400", bg: "bg-emerald-500/10 border-emerald-500/30", bar: "#10b981" },
    bugfix: { label: "🐛 Bugfix", color: "text-red-400", bg: "bg-red-500/10 border-red-500/30", bar: "#ef4444" },
    documentation: { label: "📖 Docs", color: "text-blue-400", bg: "bg-blue-500/10 border-blue-500/30", bar: "#3b82f6" },
    refactor: { label: "♻️ Refactor", color: "text-amber-400", bg: "bg-amber-500/10 border-amber-500/30", bar: "#f59e0b" },
    test: { label: "🧪 Test", color: "text-violet-400", bg: "bg-violet-500/10 border-violet-500/30", bar: "#8b5cf6" },
    chore: { label: "🔧 Chore", color: "text-slate-400", bg: "bg-slate-500/10 border-slate-500/30", bar: "#64748b" },
};
const CATS = ["feature", "bugfix", "documentation", "refactor", "test", "chore"];

function wait(ms) { return new Promise(r => setTimeout(r, ms)); }

// ─── Changelog-Seite ──────────────────────────────────────────────────────

let currentMarkdown = "";
let currentMode = "n";

// Konfiguration der Generierungs-Schritte
const STEP_CONFIG = [
    { id: "step-fetch", icon: "cloud_download", label: "Fetching commits from GitHub" },
    { id: "step-classify", icon: "psychology", label: "Classifying commits with NLP model" },
    { id: "step-summarize", icon: "summarize", label: "Summarizing with GPT-4o-mini" },
    { id: "step-render", icon: "markdown", label: "Rendering Markdown changelog" },
];
const STEP_DELAYS = [4000, 8000, 10000, 3000];
let stepTimer = null, currentStep = 0;

function initChangelog() {
    // Initialisiere die Changelog-Seite
    const container = document.getElementById("page-content");
    if (container && container.children.length === 0) {
        const tpl = document.getElementById("tpl-changelog");
        if (tpl) container.appendChild(tpl.content.cloneNode(true));
    }
    renderProgressSteps(-1);
}

function renderProgressSteps(activeIdx) {
    // Render die Fortschritts-Schritte mit aktuellem Status
    const el = document.getElementById("progress-steps");
    if (!el) return;
    el.innerHTML = STEP_CONFIG.map((s, i) => {
        const done = i < activeIdx;
        const active = i === activeIdx;
        const idle = activeIdx === -1 || i > activeIdx;
        const iconBg = done ? "bg-emerald-500/20 text-emerald-400"
            : active ? "bg-primary/20 text-primary"
                : "bg-slate-800 text-slate-600";
        const icon = done ? "check_circle" : s.icon;
        const pulse = active ? "animate-pulse" : "";
        const op = idle && activeIdx !== -1 ? "opacity-40" : idle ? "opacity-40" : "";
        const textC = done ? "text-emerald-400" : active ? "text-white" : "text-slate-500";
        return `<div class="flex items-center gap-4 ${op}" id="${s.id}">
      <div class="flex-shrink-0 w-9 h-9 rounded-full ${iconBg} ${pulse} flex items-center justify-center">
        <span class="material-symbols-outlined text-lg">${icon}</span>
      </div>
      <span class="text-sm font-medium ${textC}">${s.label}</span>
    </div>`;
    }).join("");
}

function startProgressSteps() {
    // Starte die Schritt-Animation mit Verzögerungen
    currentStep = 0;
    renderProgressSteps(0);
    const tick = () => {
        currentStep++;
        if (currentStep < STEP_CONFIG.length) {
            renderProgressSteps(currentStep);
            stepTimer = setTimeout(tick, STEP_DELAYS[currentStep]);
        }
    };
    stepTimer = setTimeout(tick, STEP_DELAYS[0]);
}

function finishProgressSteps() {
    // Beende die Schritt-Animation
    clearTimeout(stepTimer);
    renderProgressSteps(STEP_CONFIG.length);
}

function setMode(mode) {
    // Wechsle zwischen "Anzahl" und "Datumsbereich" Modi
    currentMode = mode;
    const isN = mode === "n";
    document.getElementById("n-commits-input").classList.toggle("hidden", !isN);
    document.getElementById("date-range-inputs").classList.toggle("hidden", isN);
    const activeC = "bg-slate-700 text-primary py-2 px-2 rounded-md text-xs font-bold flex items-center justify-center gap-1.5 transition-all";
    const inactiveC = "text-slate-400 py-2 px-2 rounded-md text-xs font-medium flex items-center justify-center gap-1.5 hover:bg-slate-700/50 transition-colors";
    document.getElementById("btn-mode-n").className = isN ? activeC : inactiveC;
    document.getElementById("btn-mode-date").className = isN ? inactiveC : activeC;
}

// Regex zur Validierung von GitHub URLs
const GITHUB_URL_RE = /^https?:\/\/github\.com\/[^/]+\/[^/]+\/?$/;
function validateRepoUrl(input) {
    // Validiere GitHub-URL-Format und zeige optisches Feedback
    const val = input.value.trim();
    const status = document.getElementById("url-status");
    const hint = document.getElementById("url-hint");
    if (!val) { status.textContent = ""; hint.classList.add("hidden"); return; }
    if (GITHUB_URL_RE.test(val)) {
        status.textContent = "✓";
        status.className = "absolute right-2.5 top-1/2 -translate-y-1/2 text-xs font-bold pointer-events-none text-emerald-400";
        hint.classList.add("hidden");
    } else {
        status.textContent = "✗";
        status.className = "absolute right-2.5 top-1/2 -translate-y-1/2 text-xs font-bold pointer-events-none text-red-400";
        hint.classList.remove("hidden");
    }
}

async function generateChangelog() {
    // Generiere Changelog durch API-Aufruf
    const repoUrl = document.getElementById("repo-url").value.trim();
    if (!repoUrl) { showError("Please enter a GitHub repository URL."); return; }
    if (!GITHUB_URL_RE.test(repoUrl)) { showError("Please enter a valid GitHub URL (e.g. https://github.com/owner/repo)."); return; }
    const payload = buildPayload();
    if (!payload) return;

    hideError();
    // Verstecke alte Ergebnisse und zeige Loading-Screen
    hide("empty-state"); hide("result-area"); show("loading-overlay");
    startProgressSteps();

    // Deaktiviere Button während der Generierung
    const btn = document.getElementById("generate-btn");
    btn.disabled = true;
    btn.innerHTML = `<span class="material-symbols-outlined text-sm spin">autorenew</span> Generating…`;

    try {
        // Rufe die API auf
        const res = await fetch("/api/changelog", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || `Server error: ${res.status}`);
        finishProgressSteps();
        await wait(600);
        hide("loading-overlay");
        renderResult(data);
    } catch (err) {
        hide("loading-overlay"); show("empty-state");
        showError(err.message);
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-outlined text-sm">rocket_launch</span> Generate Changelog`;
    }
}

function buildPayload() {
    // Baue die Payload für die API basierend auf Modal-Eingaben
    const payload = {
        repo_url: document.getElementById("repo-url").value.trim(),
        github_token: document.getElementById("github-token").value.trim() || null,
        output_language: document.getElementById("language").value,
    };
    // Verschiedene Modi: nach Anzahl oder Datumsbereich
    if (currentMode === "date") {
        const from = document.getElementById("date-from").value;
        const to = document.getElementById("date-to").value;
        if (!from && !to) { showError("Please select at least one date."); return null; }
        if (from) payload.date_from = from;
        if (to) payload.date_to = to;
    } else {
        const n = parseInt(document.getElementById("last-n").value, 10);
        if (!n || n < 1) { showError("Please enter a valid number of commits."); return null; }
        payload.last_n_commits = n;
    }
    return payload;
}

function renderResult(data) {
    // Render die Changelog-Ergebnisse
    currentMarkdown = data.markdown;
    document.getElementById("stats-bar").textContent =
        `${data.commit_count} commits · ${data.categories_found?.length || 0} categories · ${data.generation_time_seconds}s`;
    
    // Zeige Statistiken an
    document.getElementById("category-badges").innerHTML = (data.categories_found || []).map(cat => {
        const m = CATEGORY_META[cat.toLowerCase()] || { label: cat, color: "text-slate-400", bg: "bg-slate-500/10 border-slate-500/30" };
        return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold border ${m.bg} ${m.color}">${m.label}</span>`;
    }).join("");

    // Render Markdown-Vorschau (HTML)
    document.getElementById("markdown-output").innerHTML = marked.parse(data.markdown);
    // Zeige Rohtexte-Version
    document.getElementById("raw-output").textContent = data.markdown;

    // Erstelle Download-Link
    const blob = new Blob([data.markdown], { type: "text/markdown" });
    document.getElementById("download-btn").href = URL.createObjectURL(blob);

    show("result-area");
}

function showTab(tab) {
    // Wechsle zwischen Vorschau und Rohtexte-Tabs
    const isPreview = tab === "rendered";
    document.getElementById("tab-rendered").classList.toggle("hidden", !isPreview);
    document.getElementById("tab-raw").classList.toggle("hidden", isPreview);
    const aC = "px-2.5 py-1.5 text-xs font-semibold bg-primary/10 text-primary border border-primary/20 rounded flex items-center gap-1 transition-colors";
    const iC = "px-2.5 py-1.5 text-xs font-semibold bg-slate-800 hover:bg-slate-700 rounded flex items-center gap-1 transition-colors text-slate-300";
    document.getElementById("btn-tab-preview").className = isPreview ? aC : iC;
    document.getElementById("btn-tab-raw").className = isPreview ? iC : aC;
}

async function copyMarkdown() {
    // Kopiere das Markdown in die Zwischenablage
    if (!currentMarkdown) return;
    try { await navigator.clipboard.writeText(currentMarkdown); }
    catch { // Fallback für ältere Browser 
            const el = document.createElement("textarea");
            el.value = currentMarkdown; 
            document.body.appendChild(el); 
            el.select(); 
            document.execCommand("copy"); 
            document.body.removeChild(el); 
        }
    // Zeige "Kopiert!"-Bestätigung
    const label = document.getElementById("copy-label");
    label.textContent = "Copied!";
    setTimeout(() => (label.textContent = "Copy"), 1500);
}

function showError(msg) {
    // Zeige Fehlermeldung an
    const b = document.getElementById("error-box");
    if (!b) return;
    b.textContent = "⚠ " + msg;
    b.classList.remove("hidden");
}
function hideError() { 
    // Verstecke Fehlermeldung
    document.getElementById("error-box")?.classList.add("hidden"); 
}

// ─── Evaluierungs-Seite ───────────────────────────────────────────────────
// Seite ist bereit
function initEvaluation(){}

async function runEvaluation() {
    // Starte die Modell-Evaluierung über die API
    const btn = document.getElementById("eval-btn");
    // Deaktiviere Button und zeige Loading-State
    btn.disabled = true;
    btn.innerHTML = `<span class="material-symbols-outlined text-sm spin">autorenew</span> Running…`;
    document.getElementById("error-box-eval").classList.add("hidden");
    document.getElementById("eval-results").classList.add("hidden");

    try {
        const res = await fetch("/api/evaluate", { method: "POST" });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Evaluation failed.");
        renderEvaluation(data);
    } catch (err) {
        const b = document.getElementById("error-box-eval");
        b.textContent = "⚠ " + err.message;
        b.classList.remove("hidden");
    } finally {
        btn.disabled = false;
        btn.innerHTML = `<span class="material-symbols-outlined text-sm">play_circle</span> Run Evaluation`;
    }
}

function renderEvaluation(d) {
    // Render alle Evaluierungs-Metriken und Visualisierungen
    const fmt = v => (v * 100).toFixed(1) + "%";

    // Metriken-Definitionen
    const metricDefs = [
        { label: "Accuracy", value: fmt(d.accuracy) },
        { label: "F1 Macro", value: fmt(d.f1_macro) },
        { label: "F1 Weighted", value: fmt(d.f1_weighted) },
        { label: "Precision Macro", value: fmt(d.precision_macro) },
        { label: "Recall Macro", value: fmt(d.recall_macro) },
        { label: "Test Samples", value: d.total_samples },
    ];
    document.getElementById("metrics-grid").innerHTML = metricDefs.map(m => `
    <div class="bg-slate-800/60 border border-slate-700/50 rounded-lg p-4 text-center">
      <div class="text-2xl font-bold text-primary mb-1">${m.value}</div>
      <div class="text-xs text-slate-500 leading-tight">${m.label}</div>
    </div>`).join("");

    // Per-Klasse Tabelle
    document.getElementById("per-class-tbody").innerHTML = CATS.filter(c => d.per_class?.[c]).map(c => {
        const pc = d.per_class[c];
        const m = CATEGORY_META[c];
        const f1C = pc.f1 >= 0.8 ? "text-emerald-400" : pc.f1 >= 0.65 ? "text-amber-400" : "text-red-400";
        return `<tr class="border-b border-slate-800 last:border-0 hover:bg-slate-800/30 transition-colors">
      <td class="py-2.5 px-4"><span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold border ${m.bg} ${m.color}">${m.label}</span></td>
      <td class="py-2.5 px-4 text-right text-xs text-slate-300">${(pc.precision * 100).toFixed(1)}%</td>
      <td class="py-2.5 px-4 text-right text-xs text-slate-300">${(pc.recall * 100).toFixed(1)}%</td>
      <td class="py-2.5 px-4 text-right text-xs font-bold ${f1C}">${(pc.f1 * 100).toFixed(1)}%</td>
      <td class="py-2.5 px-4 text-right text-xs text-slate-500">${pc.support}</td>
    </tr>`;
    }).join("");

    // F1-Score Balken-Diagramm mit Animation
    document.getElementById("f1-bars").innerHTML = CATS.filter(c => d.per_class?.[c]).map(c => {
        const pc = d.per_class[c];
        const m = CATEGORY_META[c];
        const pct = (pc.f1 * 100).toFixed(1);
        return `<div class="flex items-center gap-3">
      <span class="text-xs w-24 flex-shrink-0 ${m.color} font-medium">${m.label}</span>
      <div class="flex-1 bg-slate-800 rounded-full h-2 overflow-hidden">
        <div class="bar-fill h-full rounded-full" style="width:0%;background:${m.bar}" data-w="${pct}%"></div>
      </div>
      <span class="text-xs text-slate-400 w-10 text-right">${pct}%</span>
    </div>`;
    }).join("");
    // Animiere die Balken
    requestAnimationFrame(() => {
        document.querySelectorAll(".bar-fill").forEach(el => { el.style.width = el.dataset.w; });
    });

    // Confusion Matrix (6×6 Tabelle)
    if (d.confusion_matrix) {
        const cm = d.confusion_matrix, labels = d.categories || CATS;
        const maxVal = Math.max(...cm.flat());
        let html = `<table class="text-xs border-collapse">`;
        html += `<tr><td class="p-1.5"></td>${labels.map(l => `<td class="p-1.5 text-center text-slate-500 font-semibold">${l.slice(0, 5)}</td>`).join("")}</tr>`;
        cm.forEach((row, i) => {
            html += `<tr><td class="p-1.5 text-right text-slate-500 font-semibold pr-3">${labels[i].slice(0, 5)}</td>`;
            row.forEach((val, j) => {
                const correct = i === j;
                const intensity = maxVal > 0 ? val / maxVal : 0;
                const bg = correct
                    ? `rgba(60,175,246,${(0.1 + intensity * 0.7).toFixed(2)})`
                    : val > 0 ? `rgba(239,68,68,${(0.05 + intensity * 0.4).toFixed(2)})` : "transparent";
                const tc = intensity > 0.5 ? "text-white" : correct ? "text-blue-300" : "text-slate-400";
                html += `<td class="p-1.5 text-center font-mono ${tc} rounded" style="background:${bg};min-width:2.5rem">${val}</td>`;
            });
            html += `</tr>`;
        });
        html += `</table>`;
        document.getElementById("confusion-matrix").innerHTML = html;
    }

    // Scrolle zum Ergebnis-Bereich
    document.getElementById("eval-results").classList.remove("hidden");
    document.getElementById("eval-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ─── Hilfsfunktionen ──────────────────────────────────────────────────────
function show(id) { document.getElementById(id)?.classList.remove("hidden"); }
function hide(id) { document.getElementById(id)?.classList.add("hidden"); }

// ─── Initialisierung ──────────────────────────────────────────────────────
initChangelog();