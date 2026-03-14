// ── State ─────────────────────────────────────────────────────────────────────
let currentMarkdown = "";

// ── Mode Toggle ───────────────────────────────────────────────────────────────
document.querySelectorAll('input[name="mode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    const isDate = document.getElementById("mode-date").checked;
    document.getElementById("date-range-inputs").classList.toggle("hidden", !isDate);
    document.getElementById("n-commits-input").classList.toggle("hidden", isDate);
  });
});

// ── Tab Switch ────────────────────────────────────────────────────────────────
function showTab(tab) {
  document.getElementById("tab-rendered").classList.toggle("hidden", tab !== "rendered");
  document.getElementById("tab-raw").classList.toggle("hidden", tab !== "raw");
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent.toLowerCase().includes(tab === "rendered" ? "preview" : "raw"));
  });
}

// ── Generate Changelog ────────────────────────────────────────────────────────
async function generateChangelog() {
  const repoUrl = document.getElementById("repo-url").value.trim();
  if (!repoUrl) {
    showError("Please enter a GitHub repository URL.");
    return;
  }

  const payload = buildPayload();
  if (!payload) return;

  hideError();
  showSection("loading-card");
  hideSection("result-card");

  const btn = document.getElementById("generate-btn");
  btn.disabled = true;

  try {
    const response = await fetch("/api/changelog", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || `Server error: ${response.status}`);
    }

    renderResult(data);
  } catch (err) {
    showError(err.message);
    hideSection("loading-card");
  } finally {
    btn.disabled = false;
    hideSection("loading-card");
  }
}

function buildPayload() {
  const repoUrl = document.getElementById("repo-url").value.trim();
  const token = document.getElementById("github-token").value.trim() || null;
  const language = document.getElementById("language").value;
  const isDate = document.getElementById("mode-date").checked;

  const payload = {
    repo_url: repoUrl,
    github_token: token,
    output_language: language,
  };

  if (isDate) {
    const from = document.getElementById("date-from").value;
    const to = document.getElementById("date-to").value;
    if (!from && !to) {
      showError("Please select at least one date (From or To).");
      return null;
    }
    if (from) payload.date_from = from;
    if (to) payload.date_to = to;
  } else {
    const n = parseInt(document.getElementById("last-n").value, 10);
    if (!n || n < 1) {
      showError("Please enter a valid number of commits (>= 1).");
      return null;
    }
    payload.last_n_commits = n;
  }

  return payload;
}

function renderResult(data) {
  currentMarkdown = data.markdown;

  // Stats bar
  const categories = (data.categories_found || []).join(", ");
  document.getElementById("stats-bar").innerHTML =
    `<strong>${data.commit_count}</strong> commits &nbsp;|&nbsp; ` +
    `<strong>${data.categories_found?.length || 0}</strong> categories &nbsp;|&nbsp; ` +
    `${categories} &nbsp;|&nbsp; ` +
    `<strong>${data.generation_time_seconds}s</strong>`;

  // Rendered markdown
  document.getElementById("markdown-output").innerHTML = marked.parse(data.markdown);

  // Raw markdown
  document.getElementById("raw-output").textContent = data.markdown;

  // Download link
  const blob = new Blob([data.markdown], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const dlBtn = document.getElementById("download-btn");
  dlBtn.href = url;
  dlBtn.download = "CHANGELOG.md";

  showSection("result-card");
  document.getElementById("result-card").scrollIntoView({ behavior: "smooth" });
}

// ── Copy Markdown ─────────────────────────────────────────────────────────────
async function copyMarkdown() {
  if (!currentMarkdown) return;
  try {
    await navigator.clipboard.writeText(currentMarkdown);
    const btn = document.querySelector('[onclick="copyMarkdown()"]');
    const original = btn.textContent;
    btn.textContent = "Copied!";
    setTimeout(() => (btn.textContent = original), 1500);
  } catch {
    // fallback
    const el = document.createElement("textarea");
    el.value = currentMarkdown;
    document.body.appendChild(el);
    el.select();
    document.execCommand("copy");
    document.body.removeChild(el);
  }
}

// ── Evaluation ────────────────────────────────────────────────────────────────
async function runEvaluation() {
  const btn = document.getElementById("eval-btn");
  btn.disabled = true;
  btn.textContent = "Running…";
  document.getElementById("eval-result").classList.add("hidden");

  try {
    const response = await fetch("/api/evaluate", { method: "POST" });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Evaluation failed.");
    }

    renderEvaluation(data);
  } catch (err) {
    document.getElementById("eval-result").innerHTML =
      `<p style="color:var(--danger)">${err.message}</p>`;
    document.getElementById("eval-result").classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Run Evaluation";
  }
}

function renderEvaluation(d) {
  const fmt = (v) => (v * 100).toFixed(1) + "%";

  const metrics = [
    { label: "Accuracy", value: fmt(d.accuracy) },
    { label: "F1 Macro", value: fmt(d.f1_macro) },
    { label: "F1 Weighted", value: fmt(d.f1_weighted) },
    { label: "Precision Macro", value: fmt(d.precision_macro) },
    { label: "Recall Macro", value: fmt(d.recall_macro) },
    { label: "Test Samples", value: d.total_samples },
  ];

  const grid = metrics
    .map(
      (m) =>
        `<div class="metric-card">
          <div class="metric-value">${m.value}</div>
          <div class="metric-label">${m.label}</div>
        </div>`
    )
    .join("");

  const categories = ["feature", "bugfix", "documentation", "refactor", "test", "chore"];
  const rows = categories
    .filter((c) => d.per_class && d.per_class[c])
    .map((c) => {
      const pc = d.per_class[c];
      return `<tr>
        <td>${c}</td>
        <td>${(pc.precision * 100).toFixed(1)}%</td>
        <td>${(pc.recall * 100).toFixed(1)}%</td>
        <td>${(pc.f1 * 100).toFixed(1)}%</td>
        <td>${pc.support}</td>
      </tr>`;
    })
    .join("");

  document.getElementById("eval-result").innerHTML = `
    <h3 style="margin-bottom:16px;font-size:0.95rem;">Overall Metrics</h3>
    <div class="metrics-grid">${grid}</div>
    <h3 style="margin-bottom:8px;font-size:0.95rem;">Per-Class Metrics</h3>
    <table class="per-class-table">
      <thead>
        <tr>
          <th>Category</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
    <p class="hint" style="margin-top:12px;">Full results saved to data/evaluation_results.json</p>
  `;
  document.getElementById("eval-result").classList.remove("hidden");
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showSection(id) {
  document.getElementById(id).classList.remove("hidden");
}
function hideSection(id) {
  document.getElementById(id).classList.add("hidden");
}
function showError(msg) {
  const box = document.getElementById("error-box");
  box.textContent = "Error: " + msg;
  box.classList.remove("hidden");
}
function hideError() {
  document.getElementById("error-box").classList.add("hidden");
}
