"use strict";

const FIELDS = ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
  "MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
  "TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling",
  "PaymentMethod","MonthlyCharges","TotalCharges"];
const NUMERIC = new Set(["tenure","MonthlyCharges","TotalCharges"]);
const ADDON_FIELDS = ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
  "StreamingTV","StreamingMovies"];

const MODEL_LABELS = {
  logistic_regression: "Logistic Regression",
  random_forest: "Random Forest",
  hist_gradient_boosting: "Gradient Boosted Trees",
};

/**
 * Build the /predict request body from raw form values. Unchanged from the
 * previous version -- this redesign is presentation-only, and the payload
 * this produces is what's under test in test_contract.py and test_api.py
 * on the backend.
 */
function buildPayload(values){
  const body = {};
  for (const f of FIELDS){
    const raw = values[f];
    body[f] = NUMERIC.has(f) ? (raw === "" || raw == null ? null : Number(raw)) : raw;
  }
  if (body.InternetService === "No"){
    for (const f of ADDON_FIELDS) body[f] = "No internet service";
  }
  if (body.PhoneService === "No"){
    body.MultipleLines = "No phone service";
  }
  return body;
}

/**
 * Locks the six internet add-ons and Multiple lines to match what
 * buildPayload will actually send, and disables them while their parent
 * service is off, so the visible form can never show a value different
 * from what gets submitted.
 *
 * `fields` is a plain map of id -> anything with .value/.disabled -- a real
 * <select> and a plain test double both satisfy that, which is what makes
 * this testable with ordinary objects instead of a real DOM.
 * `remembered` is mutated in place: a field's value is captured the moment
 * it's disabled, and restored the moment it's re-enabled, so switching
 * internet service off and back on doesn't silently discard what the user
 * had picked.
 */
function syncDependentFields(internetService, phoneService, fields, remembered){
  lockField(fields.MultipleLines, phoneService === "No", "No phone service", "MultipleLines", remembered);
  for (const id of ADDON_FIELDS){
    lockField(fields[id], internetService === "No", "No", id, remembered);
  }
}

function lockField(el, shouldLock, lockedValue, key, remembered){
  if (!el) return;
  if (shouldLock){
    if (!el.disabled) remembered[key] = el.value;
    el.value = lockedValue;
    el.disabled = true;
  } else if (el.disabled){
    el.disabled = false;
    if (remembered[key] != null){
      el.value = remembered[key];
      delete remembered[key];
    }
  }
}

/** Same-origin by default; file:// (only relevant while editing this file) falls back to local dev. */
function apiBaseUrl(protocol, origin){
  return protocol && protocol.startsWith("http") ? origin : "http://localhost:8000";
}

function escapeHtml(s){
  return String(s).replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
}

function humanModelName(modelVersion){
  const family = String(modelVersion || "").split("@")[0];
  if (!family) return "Unknown model";
  if (MODEL_LABELS[family]) return MODEL_LABELS[family];
  return family.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

function riskLabel(band){
  if (!band) return "Unknown risk";
  return band.charAt(0).toUpperCase() + band.slice(1) + " risk";
}

function interpretationSentence(data){
  return data.will_churn
    ? "High churn risk. The predicted churn probability is above the model's decision threshold."
    : "Lower churn risk. The predicted churn probability is below the model's decision threshold.";
}

function formatPercent(fraction, digits){
  return (fraction * 100).toFixed(digits == null ? 1 : digits) + "%";
}

/** "20.7 pts above threshold" / "10.3 pts below threshold" -- from the real probability and threshold, nothing hardcoded. */
function thresholdDistance(probability, threshold){
  const diffPts = (probability - threshold) * 100;
  const magnitude = Math.abs(diffPts).toFixed(1);
  return `${magnitude} pts ${diffPts >= 0 ? "above" : "below"} threshold`;
}

function interpretError(status){
  if (status == null){
    return "Unable to connect to the prediction service. Please make sure the API is running and try again.";
  }
  if (status === 503){
    return "The prediction service is temporarily unavailable. Please try again in a moment.";
  }
  if (status === 422){
    return "Some of the entered customer details couldn't be validated. Please check the form and try again.";
  }
  return "Something went wrong while scoring this customer. Please try again.";
}

// --------------------------------------------------------------------- //
// Gauge geometry -- a semicircular arc, 0% at the left point (180°) to
// 100% at the right point (0°), sweeping clockwise over the top (90°).
// Plain trigonometry, not a charting library: this is a single arc, and a
// dependency buys nothing a few lines of geometry doesn't already give us.
// Angles are in the ordinary math convention (0°=right, 90°=up, 180°=left)
// so the halfway point of a p=50 arc is straight up, which is the easiest
// property to sanity-check by eye and the one the tests below rely on.
// --------------------------------------------------------------------- //

function percentToAngle(percent){
  return 180 - (clampPercent(percent) / 100) * 180;
}

function clampPercent(p){
  return Math.max(0, Math.min(100, p));
}

/** A point on the gauge's circle at the given math-convention angle (degrees). */
function pointOnArc(cx, cy, r, angleDeg){
  const rad = (angleDeg * Math.PI) / 180;
  // SVG's y-axis increases downward, so the y term is subtracted rather
  // than added -- otherwise the arc would open upward instead of downward.
  return {
    x: round2(cx + r * Math.cos(rad)),
    y: round2(cy - r * Math.sin(rad)),
  };
}

function round2(n){
  return Math.round(n * 100) / 100;
}

/** The SVG path `d` string for the arc from 0% up to `percent`. */
function describeArc(cx, cy, r, percent){
  const start = pointOnArc(cx, cy, r, 180);
  const end = pointOnArc(cx, cy, r, percentToAngle(percent));
  const sweptDegrees = 180 - percentToAngle(percent);
  const largeArcFlag = sweptDegrees > 180 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`;
}

// --------------------------------------------------------------------- //
// Tooltip placement -- pure geometry so the "never clips at the viewport
// edge" requirement is something a test can actually check with numbers,
// not something that has to be eyeballed in a browser.
// --------------------------------------------------------------------- //

function clamp(min, value, max){
  return Math.max(min, Math.min(max, value));
}

/** Horizontal position (px, from the viewport's left edge) for a tooltip centred under its trigger. */
function tooltipLeft(triggerCenterX, tooltipWidth, viewportWidth, margin){
  const desired = triggerCenterX - tooltipWidth / 2;
  const max = viewportWidth - tooltipWidth - margin;
  // A tooltip wider than the viewport (a tiny phone, a very long string)
  // has no valid clamp range -- centre it rather than let max<min invert
  // the clamp and push it off-screen the other way.
  if (max < margin) return round2((viewportWidth - tooltipWidth) / 2);
  return round2(clamp(margin, desired, max));
}

/**
 * "above" the trigger unless there's no room, in which case "below" it
 * instead. If genuinely neither direction has room (a very short viewport),
 * whichever side has more room wins, rather than defaulting arbitrarily.
 */
function tooltipPlacement(triggerTop, triggerBottom, tooltipHeight, viewportHeight, margin){
  const roomAbove = triggerTop - margin;
  const roomBelow = viewportHeight - triggerBottom - margin;
  if (roomAbove >= tooltipHeight) return "above";
  if (roomBelow >= tooltipHeight) return "below";
  return roomAbove >= roomBelow ? "above" : "below";
}

/**
 * Where the tooltip's pointer triangle should sit (px from the tooltip's
 * own left edge), so it keeps pointing at the real trigger icon even when
 * the tooltip box itself has been shifted sideways to avoid a viewport
 * edge. Clamped inside the tooltip's own width so the arrow never renders
 * outside the bubble it's attached to.
 */
function arrowLeft(triggerCenterX, tooltipLeftPx, tooltipWidth, margin){
  const raw = triggerCenterX - tooltipLeftPx;
  return round2(clamp(margin, raw, tooltipWidth - margin));
}

/** The result panel's content for a successful prediction. Every value comes from `data`. */
function renderResult(data){
  const pct = formatPercent(data.churn_probability);
  const thresholdPct = formatPercent(data.threshold, 0);
  const probabilityPercent = clampPercent(data.churn_probability * 100);
  const thresholdPercent = clampPercent(data.threshold * 100);
  const band = String(data.risk_band || "unknown");
  const modelName = escapeHtml(humanModelName(data.model_version));
  const predictionText = data.will_churn ? "Likely to churn" : "Not likely to churn";

  // Gauge geometry: a 200x120 viewBox, arc centred at (100,110) radius 88.
  const cx = 100, cy = 110, r = 88;
  const arcPath = describeArc(cx, cy, r, probabilityPercent);
  const trackPath = describeArc(cx, cy, r, 100);
  const tick = pointOnArc(cx, cy, r, percentToAngle(thresholdPercent));
  const tickInner = pointOnArc(cx, cy, r - 14, percentToAngle(thresholdPercent));

  return `<div class="result risk-${escapeHtml(band)}">
    <p class="result-kicker">Risk assessment</p>
    <div class="gauge" role="img" aria-label="Churn probability ${pct} of 100%, decision threshold ${thresholdPct}">
      <svg viewBox="0 0 200 120" class="gauge-svg">
        <path d="${trackPath}" class="gauge-track" fill="none" />
        <path d="${arcPath}" class="gauge-fill" fill="none" />
        <line x1="${tickInner.x}" y1="${tickInner.y}" x2="${tick.x}" y2="${tick.y}" class="gauge-threshold" />
      </svg>
      <div class="gauge-readout">
        <span class="risk-tag">${escapeHtml(riskLabel(band))}</span>
        <span class="prob-number">${pct}</span>
        <span class="prob-label">Probability of churn</span>
        <span class="prob-delta">${thresholdDistance(data.churn_probability, data.threshold)}</span>
      </div>
      <div class="gauge-scale"><span>0%</span><span>100%</span></div>
    </div>
    <p class="interpretation">${interpretationSentence(data)}</p>
    <dl class="facts">
      <div><dt>Prediction</dt><dd>${escapeHtml(predictionText)}</dd></div>
      <div><dt>Decision threshold</dt><dd>${thresholdPct}</dd></div>
      <div><dt>Model</dt><dd>${modelName}</dd></div>
      <div><dt>Risk band</dt><dd>${escapeHtml(riskLabel(band).replace(" risk",""))}</dd></div>
    </dl>
  </div>`;
}

function renderError(message){
  return `<div class="result-error"><p>${escapeHtml(message)}</p></div>`;
}

// ---- DOM wiring. Everything above this line is pure and unit-tested in ----
// ---- app.test.js; everything below reads and writes the actual page.  ----
if (typeof document !== "undefined"){

  const base = apiBaseUrl(window.location.protocol, window.location.origin);

  function collectFormValues(){
    const values = {};
    for (const f of FIELDS) values[f] = document.getElementById(f).value;
    return values;
  }

  // -- status dot, hero model name, and the technical strip: all three come --
  // -- from this one /health call, nothing hardcoded and nothing guessed.  --
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");
  const heroModel = document.getElementById("hero-model");
  const stripModel = document.getElementById("strip-model");
  const stripThreshold = document.getElementById("strip-threshold");

  async function refreshApiStatus(){
    try {
      const res = await fetch(base + "/health");
      const body = await res.json().catch(() => ({}));
      const ok = res.ok && body.status === "ok";

      statusDot.className = "dot" + (ok ? " dot-ok" : " dot-bad");
      statusText.textContent = ok ? "Prediction service — Ready" : "Prediction service — Unavailable";

      if (ok && body.model_family){
        const name = humanModelName(body.model_family);
        heroModel.textContent = name;
        stripModel.textContent = name;
      }
      if (ok && typeof body.threshold === "number"){
        stripThreshold.textContent = formatPercent(body.threshold, 0);
      }
    } catch {
      statusDot.className = "dot dot-bad";
      statusText.textContent = "Prediction service — Unavailable";
    }
  }
  refreshApiStatus();

  // -- shared tooltip: one element, repositioned and repopulated on demand --
  // -- rather than one tooltip per field, so the edge-avoidance logic only --
  // -- has to be right once.                                              --
  const tooltip = document.getElementById("info-tooltip");
  let activeInfoButton = null;

  const GAP = 8; // px between the icon and the tooltip bubble

  function showTooltip(button){
    tooltip.textContent = button.dataset.tip;
    tooltip.hidden = false;
    const trigger = button.getBoundingClientRect();
    const tip = tooltip.getBoundingClientRect();
    const triggerCenterX = trigger.left + trigger.width / 2;

    const left = tooltipLeft(triggerCenterX, tip.width, window.innerWidth, GAP);
    const placement = tooltipPlacement(trigger.top, trigger.bottom, tip.height, window.innerHeight, GAP);

    // .tooltip is `position:fixed`, so its coordinates are already relative
    // to the viewport -- getBoundingClientRect() is too, which is exactly
    // what fixed positioning needs. Adding window.scrollY here (an earlier
    // version of this function did) double-counts the scroll offset: the
    // tooltip would drift further from its icon the further down the page
    // you'd scrolled, which is exactly the "tooltip is too far away" bug
    // this fixes. No scrollY term belongs in either line below.
    tooltip.style.left = left + "px";
    tooltip.style.top = placement === "above"
      ? (trigger.top - tip.height - GAP) + "px"
      : (trigger.bottom + GAP) + "px";

    tooltip.style.setProperty("--arrow-left", arrowLeft(triggerCenterX, left, tip.width, 10) + "px");
    tooltip.classList.toggle("tip-above", placement === "above");
    button.setAttribute("aria-expanded", "true");
    activeInfoButton = button;
  }

  function hideTooltip(){
    tooltip.hidden = true;
    if (activeInfoButton) activeInfoButton.setAttribute("aria-expanded", "false");
    activeInfoButton = null;
  }

  document.querySelectorAll(".info-btn").forEach((btn) => {
    btn.addEventListener("mouseenter", () => showTooltip(btn));
    btn.addEventListener("mouseleave", () => { if (document.activeElement !== btn) hideTooltip(); });
    btn.addEventListener("focus", () => showTooltip(btn));
    btn.addEventListener("blur", () => hideTooltip());
    btn.addEventListener("click", (event) => {
      event.preventDefault(); // it's type="button" already; this just stops any bubbled form quirks
      if (activeInfoButton === btn && !tooltip.hidden) hideTooltip();
      else showTooltip(btn);
    });
  });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") hideTooltip();
  });
  document.addEventListener("click", (event) => {
    if (activeInfoButton && !event.target.closest(".info-btn") && !event.target.closest("#info-tooltip")){
      hideTooltip();
    }
  });
  // Scrolling moves the icon in the document, but the tooltip is
  // `position:fixed` and was placed from a one-time snapshot of the icon's
  // screen position -- so as soon as the page moves, that snapshot is
  // stale. mouseleave doesn't catch this: the pointer itself hasn't moved,
  // only the page under it, so no mouse event fires at all. Listening for
  // `scroll` is the only reliable signal here, and `capture: true` catches
  // it even if it originates from some scrollable element other than the
  // window (scroll events don't bubble, unlike most DOM events). Covers
  // wheel and touchpad scrolling identically -- both just produce a native
  // `scroll` event, there's nothing input-device-specific to branch on.
  window.addEventListener("scroll", () => {
    if (activeInfoButton) hideTooltip();
  }, { passive: true, capture: true });
  window.addEventListener("resize", () => { if (activeInfoButton) showTooltip(activeInfoButton); });

  // -- internet/phone service dependency locking: keeps the visible form --
  // -- state honest about what buildPayload will actually send.         --
  const dependentFields = Object.fromEntries(
    [...ADDON_FIELDS, "MultipleLines"].map((id) => [id, document.getElementById(id)])
  );
  const rememberedValues = {};
  const internetSelect = document.getElementById("InternetService");
  const phoneSelect = document.getElementById("PhoneService");

  function applyDependencies(){
    syncDependentFields(internetSelect.value, phoneSelect.value, dependentFields, rememberedValues);
  }
  internetSelect.addEventListener("change", applyDependencies);
  phoneSelect.addEventListener("change", applyDependencies);
  applyDependencies(); // correct from the first paint, not just after the first change

  // -- the form itself --
  const form = document.getElementById("form");
  const btn = document.getElementById("go");
  const btnLabel = document.getElementById("go-label");
  const result = document.getElementById("result");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    btn.disabled = true;
    btnLabel.textContent = "Predicting…";

    const payload = buildPayload(collectFormValues());

    try {
      const res = await fetch(base + "/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json().catch(() => null);

      result.classList.add("result-updating");
      result.innerHTML = (res.ok && data) ? renderResult(data) : renderError(interpretError(res.status));
      requestAnimationFrame(() => result.classList.remove("result-updating"));
    } catch {
      result.innerHTML = renderError(interpretError(null));
    } finally {
      btn.disabled = false;
      btnLabel.textContent = "Predict Churn";
      result.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });
}

// Exposed for app.test.js. Guarded so this never runs in a browser, where
// `module` doesn't exist.
if (typeof module !== "undefined" && module.exports){
  module.exports = { buildPayload, apiBaseUrl, escapeHtml, humanModelName, riskLabel,
    interpretationSentence, formatPercent, thresholdDistance, interpretError, renderResult, renderError,
    percentToAngle, pointOnArc, describeArc, tooltipLeft, tooltipPlacement, arrowLeft,
    syncDependentFields, ADDON_FIELDS, FIELDS };
}
