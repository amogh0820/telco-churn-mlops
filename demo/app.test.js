// Tests for demo/app.js -- the parts of the frontend that don't need a
// browser. Run with `node demo/app.test.js` (see `make test-frontend`).
//
// Every fixture is copied verbatim from the real contract: the field list
// matches src/api/schemas.py's CustomerFeatures, the response shape matches
// PredictionResponse exactly as main.py returns it, and the /health fixture
// matches HealthResponse. If any of those drift, these fixtures -- not just
// app.js -- need updating too.
"use strict";

const {
  buildPayload, apiBaseUrl, humanModelName, riskLabel, interpretationSentence,
  formatPercent, thresholdDistance, interpretError, renderResult, renderError,
  percentToAngle, pointOnArc, describeArc, tooltipLeft, tooltipPlacement, arrowLeft,
  syncDependentFields, ADDON_FIELDS, FIELDS,
} = require("./app.js");

let failures = 0;
function check(name, actual, expected){
  const a = JSON.stringify(actual), e = JSON.stringify(expected);
  if (a === e) { console.log("  PASS", name); }
  else { console.log("  FAIL", name, "\n    got:     ", a, "\n    expected:", e); failures++; }
}
function checkTrue(name, condition){
  if (condition) { console.log("  PASS", name); }
  else { console.log("  FAIL", name, "(expected a truthy result)"); failures++; }
}

// ===================== payload building (unchanged logic) ==================

const SCHEMA_FIELDS = ["gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
  "MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
  "TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling",
  "PaymentMethod","MonthlyCharges","TotalCharges"];
check("FIELDS matches the real 19-field schema exactly",
  [...FIELDS].sort(), [...SCHEMA_FIELDS].sort());

const DEFAULTS = {
  gender:"Female", SeniorCitizen:"No", Partner:"Yes", Dependents:"No", tenure:"2",
  PhoneService:"Yes", MultipleLines:"No", InternetService:"Fiber optic",
  OnlineSecurity:"No", OnlineBackup:"No", DeviceProtection:"No", TechSupport:"No",
  StreamingTV:"Yes", StreamingMovies:"Yes", Contract:"Month-to-month",
  PaperlessBilling:"Yes", PaymentMethod:"Electronic check",
  MonthlyCharges:"95.5", TotalCharges:"190"
};
const EXAMPLE_CUSTOMER = { // verbatim from EXAMPLE_CUSTOMER in src/api/schemas.py
  gender:"Female", SeniorCitizen:"No", Partner:"Yes", Dependents:"No", tenure:2,
  PhoneService:"Yes", MultipleLines:"No", InternetService:"Fiber optic",
  OnlineSecurity:"No", OnlineBackup:"No", DeviceProtection:"No", TechSupport:"No",
  StreamingTV:"Yes", StreamingMovies:"Yes", Contract:"Month-to-month",
  PaperlessBilling:"Yes", PaymentMethod:"Electronic check",
  MonthlyCharges:95.5, TotalCharges:190.0
};
check("default form values build the exact /docs example payload -- the visual redesign changed nothing here",
  buildPayload(DEFAULTS), EXAMPLE_CUSTOMER);

function addonFieldsAllNo(payload){
  return ["OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    .every(f => payload[f] === "No internet service");
}
check("InternetService=No still forces every add-on to 'No internet service'",
  addonFieldsAllNo(buildPayload({...DEFAULTS, InternetService:"No", OnlineSecurity:"Yes"})), true);
check("PhoneService=No still forces MultipleLines to 'No phone service'",
  buildPayload({...DEFAULTS, PhoneService:"No", MultipleLines:"Yes"}).MultipleLines, "No phone service");

const zeroTenure = buildPayload({...DEFAULTS, tenure:"0", TotalCharges:""});
check("blank TotalCharges still becomes null, not NaN or an empty string", zeroTenure.TotalCharges, null);
check("tenure still coerces to a number, not a string", zeroTenure.tenure, 0);

check("http(s) origin is still the address used, with no visible field to misconfigure",
  apiBaseUrl("https:", "https://telco-churn-api.onrender.com"), "https://telco-churn-api.onrender.com");
check("a file:// origin still falls back to the local dev server",
  apiBaseUrl("file:", "null"), "http://localhost:8000");

// ===================== human-readable labels ==================

check("random_forest reads as Random Forest", humanModelName("random_forest@2026-09-20T14:26:42+00:00"), "Random Forest");
check("logistic_regression reads as Logistic Regression", humanModelName("logistic_regression@x"), "Logistic Regression");
check("hist_gradient_boosting reads as Gradient Boosted Trees", humanModelName("hist_gradient_boosting@x"), "Gradient Boosted Trees");
check("an unrecognised family still gets a readable label", humanModelName("some_new_model@x"), "Some New Model");
check("humanModelName also accepts a bare family (used for the /health-driven hero text)",
  humanModelName("random_forest"), "Random Forest");
check("a missing model doesn't crash", humanModelName(undefined), "Unknown model");

check("risk band reads as a capitalised label", riskLabel("high"), "High risk");
check("medium band reads correctly", riskLabel("medium"), "Medium risk");
check("low band reads correctly", riskLabel("low"), "Low risk");

check("above-threshold reads as the exact requested sentence",
  interpretationSentence({will_churn:true}),
  "High churn risk. The predicted churn probability is above the model's decision threshold.");
check("below-threshold reads as the exact requested sentence",
  interpretationSentence({will_churn:false}),
  "Lower churn risk. The predicted churn probability is below the model's decision threshold.");

check("formats a probability to one decimal by default", formatPercent(0.893103), "89.3%");
check("formats a threshold to zero decimals when asked", formatPercent(0.58, 0), "58%");

// The two worked examples given for this feature, exactly as specified.
check("78.7% vs a 58% threshold reads as the exact requested wording",
  thresholdDistance(0.787, 0.58), "20.7 pts above threshold");
check("47.7% vs a 58% threshold reads as the exact requested wording",
  thresholdDistance(0.477, 0.58), "10.3 pts below threshold");
check("exactly at the threshold reads as 'above' (0.0 pts) -- consistent with the backend's own >= for will_churn",
  thresholdDistance(0.58, 0.58), "0.0 pts above threshold");
check("floating-point subtraction noise doesn't leak into the displayed number",
  thresholdDistance(0.893103, 0.58), "31.3 pts above threshold");

check("no response reads as the exact requested message", interpretError(null),
  "Unable to connect to the prediction service. Please make sure the API is running and try again.");
check("a 503 reads as temporarily unavailable, not a status code", interpretError(503),
  "The prediction service is temporarily unavailable. Please try again in a moment.");
check("a 422 reads as a form problem, not a Pydantic error dump", interpretError(422),
  "Some of the entered customer details couldn't be validated. Please check the form and try again.");

// ===================== gauge geometry: exact coordinates, not eyeballing ==================
// cx=100, cy=110, r=88 is the geometry renderResult actually uses.

check("0% sits at the arc's left point (same height as centre)", pointOnArc(100,110,88,180), {x:12, y:110});
check("50% sits straight above the centre (the easiest point to sanity-check by eye)",
  pointOnArc(100,110,88,90), {x:100, y:22});
check("100% sits at the arc's right point (same height as centre)", pointOnArc(100,110,88,0), {x:188, y:110});
check("percentToAngle: 0% is 180°, 50% is 90°, 100% is 0°",
  [percentToAngle(0), percentToAngle(50), percentToAngle(100)], [180, 90, 0]);
check("percentToAngle clamps a value below 0", percentToAngle(-30), 180);
check("percentToAngle clamps a value above 100", percentToAngle(140), 0);

checkTrue("a 0% arc starts and ends at the same point (no visible fill)",
  describeArc(100,110,88,0) === "M 12 110 A 88 88 0 0 1 12 110");
checkTrue("a 100% arc spans the full semicircle (left point to right point)",
  describeArc(100,110,88,100).includes("12 110") && describeArc(100,110,88,100).includes("188 110"));

// ===================== tooltip placement: real viewport-width numbers ==================
// 375px is an iPhone SE's CSS width -- the narrowest common real device, and
// exactly the case "don't clip at the edge" has to hold up against.

check("a trigger near the right edge of a 375px phone clamps left, not off-screen",
  tooltipLeft(360, 220, 375, 8), 147);
check("a trigger near the left edge clamps to the margin, not negative",
  tooltipLeft(15, 220, 375, 8), 8);
check("a comfortably centred trigger isn't clamped at all",
  tooltipLeft(187, 220, 375, 8), 77);
checkTrue("even a tooltip wider than the viewport itself produces a finite, sane number",
  Number.isFinite(tooltipLeft(50, 500, 375, 8)));

check("with room on both sides, the tooltip now prefers ABOVE the trigger",
  tooltipPlacement(300, 340, 60, 900, 8), "above");
check("no room above (trigger near the very top) falls back to below",
  tooltipPlacement(10, 40, 60, 900, 8), "below");
check("no room in EITHER direction picks whichever side has more room (here: 12px above vs 42px below)",
  tooltipPlacement(20, 850, 60, 900, 8), "below");
check("...and the reverse case genuinely picks the other side when IT has more room (842px above vs 12px below)",
  tooltipPlacement(850, 880, 60, 900, 8), "above");

check("the arrow points at a trigger centred inside an unclamped tooltip",
  arrowLeft(160, 50, 220, 10), 110);
check("the arrow stays inside the tooltip's own bounds even when the trigger is far outside them",
  arrowLeft(5, 50, 220, 10), 10);
check("the arrow doesn't overshoot the tooltip's right edge either",
  arrowLeft(500, 50, 220, 10), 210);

// ===================== the tooltip position bug this turn actually fixes ==================
// The bug: .tooltip is `position:fixed` (viewport-relative by definition),
// but an earlier version of showTooltip() added window.scrollY to `top` --
// which only belongs there for `position:absolute`. The fix is simply not
// having that term. There's no scrollY parameter anywhere in the placement
// functions any more, which is the property worth locking in: if a future
// edit re-adds a scroll offset to a fixed-position calculation, it would
// have to add a new parameter to do it, not just tweak a call site.
checkTrue("tooltipPlacement's signature takes no scroll-offset parameter",
  tooltipPlacement.length === 5); // triggerTop, triggerBottom, tooltipHeight, viewportHeight, margin

// ===================== internet/phone service dependency locking ==================

function freshFields(overrides){
  const base = {};
  for (const id of [...ADDON_FIELDS, "MultipleLines"]) base[id] = { value: "No", disabled: false };
  return Object.assign(base, overrides);
}

const offFields = freshFields({
  OnlineSecurity: { value: "Yes", disabled: false },
  StreamingTV: { value: "Yes", disabled: false },
  StreamingMovies: { value: "Yes", disabled: false },
});
const remembered1 = {};
syncDependentFields("No", "Yes", offFields, remembered1);
for (const id of ADDON_FIELDS){
  check(`InternetService=No locks ${id} to "No"`, offFields[id].value, "No");
  check(`InternetService=No disables ${id}`, offFields[id].disabled, true);
}
check("MultipleLines is untouched by InternetService (only PhoneService governs it)",
  offFields.MultipleLines.disabled, false);
check("the pre-existing Yes on OnlineSecurity was captured before being overwritten",
  remembered1.OnlineSecurity, "Yes");

syncDependentFields("Fiber optic", "Yes", offFields, remembered1);
check("switching internet back on re-enables OnlineSecurity", offFields.OnlineSecurity.disabled, false);
check("...and restores its real previous value, rather than leaving it stuck on the locked 'No'",
  offFields.OnlineSecurity.value, "Yes");
check("a field that was already 'No' before locking is still 'No' after restore (no false change)",
  offFields.OnlineBackup.value, "No");
check("the remembered map is cleared for a field once it's been restored (no stale memory)",
  Object.prototype.hasOwnProperty.call(remembered1, "OnlineSecurity"), false);

const phoneFields = freshFields({ MultipleLines: { value: "Yes", disabled: false } });
const remembered2 = {};
syncDependentFields("Fiber optic", "No", phoneFields, remembered2);
check("PhoneService=No locks MultipleLines to the REAL backend value 'No phone service', not a lossy 'No'",
  phoneFields.MultipleLines.value, "No phone service");
check("PhoneService=No disables MultipleLines", phoneFields.MultipleLines.disabled, true);
syncDependentFields("Fiber optic", "Yes", phoneFields, remembered2);
check("switching phone service back on restores what the user had actually chosen",
  phoneFields.MultipleLines.value, "Yes");
check("...and re-enables it", phoneFields.MultipleLines.disabled, false);

const bothOffFields = freshFields();
const remembered3 = {};
syncDependentFields("No", "No", bothOffFields, remembered3);
checkTrue("both parents off at once locks every dependent field",
  ADDON_FIELDS.every((id) => bothOffFields[id].disabled) && bothOffFields.MultipleLines.disabled);

check("syncDependentFields silently no-ops on a field that doesn't exist in the map (defensive)",
  (() => { syncDependentFields("No", "No", {}, {}); return "no throw"; })(), "no throw");

// ===================== the rendered result, against the REAL PredictionResponse shape ==================

const highRisk = {
  churn_probability: 0.893103, will_churn: true, threshold: 0.58,
  risk_band: "high", model_version: "random_forest@2026-09-20T14:26:42.065565+00:00"
};
const highHtml = renderResult(highRisk);
checkTrue("shows the real probability, not a hardcoded example", highHtml.includes("89.3%"));
checkTrue("shows the real threshold from the backend", highHtml.includes("58%"));
checkTrue("names the real model, human-readable", highHtml.includes("Random Forest"));
checkTrue("carries the risk-high styling hook", highHtml.includes("risk-high"));
checkTrue("says 'Likely to churn'", highHtml.includes("Likely to churn"));
checkTrue("includes the exact requested interpretation sentence",
  highHtml.includes("High churn risk. The predicted churn probability is above the model's decision threshold."));
checkTrue("includes a Risk band fact matching the actual risk_band, not just the headline tag",
  highHtml.includes("<dt>Risk band</dt><dd>High</dd>"));
checkTrue("includes the threshold-distance caption, computed from the real probability and threshold",
  highHtml.includes("31.3 pts above threshold"));
checkTrue("the gauge's fill arc is present as real SVG path data", /<path d="M [\d.]+ [\d.]+ A/.test(highHtml));
checkTrue("the gauge's threshold tick is positioned from the real threshold value (58%), not fixed",
  highHtml.includes('class="gauge-threshold"'));

const lowRisk = {
  churn_probability: 0.04, will_churn: false, threshold: 0.58,
  risk_band: "low", model_version: "random_forest@x"
};
const lowHtml = renderResult(lowRisk);
checkTrue("low-risk result says 'Not likely to churn'", lowHtml.includes("Not likely to churn"));
checkTrue("low-risk result carries the risk-low styling hook", lowHtml.includes("risk-low"));
checkTrue("low-risk result's threshold-distance caption correctly reads 'below', not 'above'",
  lowHtml.includes("54.0 pts below threshold"));

checkTrue("a rendered result never shows the word GET, POST, or JSON", !/\bGET\b|\bPOST\b|\bJSON\b/i.test(highHtml));
checkTrue("a rendered result never shows a raw snake_case model family", !highHtml.includes("random_forest"));

const weird = { ...highRisk, model_version: '<script>x</script>@y' };
checkTrue("an unusual model_version is escaped, never injected as raw HTML",
  !renderResult(weird).includes("<script>"));

const errHtml = renderError(interpretError(null));
checkTrue("a rendered error shows the human sentence", errHtml.includes("Unable to connect to the prediction service"));
checkTrue("a rendered error never contains a curly brace (no leaked JSON)",
  !errHtml.includes("{") && !errHtml.includes("}"));

// ===================== regression: scrolling must hide an open tooltip ==================
// Everything above tests pure functions. This is different on purpose: the
// bug report was that scrolling the page with the cursor stationary over an
// ⓘ icon left the tooltip visibly detached from it, because mouseenter/
// mouseleave only fire on real pointer movement -- the page moving under a
// still cursor fires neither. That's a wiring bug, not a math bug, so the
// only real test for it is one that actually exercises app.js's DOM-wiring
// branch (normally skipped in Node, since `document` doesn't exist there)
// against a minimal fake DOM, and checks that a real `scroll` event reaches
// the tooltip's hidden state -- not just that hideTooltip() works in
// isolation, which would pass even if nothing ever called it.

function makeFakeElement(overrides){
  const listeners = {};
  return Object.assign({
    dataset: {},
    style: { setProperty(){} },
    classList: { toggle(){} },
    textContent: "",
    hidden: true,
    disabled: false,
    value: "",
    addEventListener(type, handler){ (listeners[type] ||= []).push(handler); },
    dispatch(type, event){ (listeners[type] || []).forEach((h) => h(event || {})); },
    getBoundingClientRect(){ return { left: 100, top: 100, bottom: 120, width: 16, height: 16 }; },
    setAttribute(){}, getAttribute(){ return null; },
    closest(){ return null; },
  }, overrides);
}

function buildFakeDom(){
  const otherIds = ["status-dot", "status-text", "hero-model", "strip-model", "strip-threshold",
    "info-tooltip", "form", "go", "go-label", "result"];
  const elements = {};
  for (const id of [...SCHEMA_FIELDS, ...otherIds]) elements[id] = makeFakeElement({ id });

  const infoBtn = makeFakeElement({ dataset: { tip: "Example tooltip text." } });

  const documentListeners = {};
  const fakeDocument = {
    getElementById: (id) => elements[id] || makeFakeElement({ id }),
    querySelectorAll: (sel) => (sel === ".info-btn" ? [infoBtn] : []),
    addEventListener(type, handler){ (documentListeners[type] ||= []).push(handler); },
    dispatch(type, event){ (documentListeners[type] || []).forEach((h) => h(event || {})); },
    activeElement: null,
  };

  const windowListeners = {};
  const fakeWindow = {
    location: { protocol: "http:", origin: "http://localhost:8000" },
    innerWidth: 1280, innerHeight: 800,
    addEventListener(type, handler){ (windowListeners[type] ||= []).push(handler); },
    dispatch(type, event){ (windowListeners[type] || []).forEach((h) => h(event || {})); },
  };

  return { infoBtn, fakeDocument, fakeWindow };
}

(function scrollHidesTooltipRegressionTest(){
  const { infoBtn, fakeDocument, fakeWindow } = buildFakeDom();

  global.document = fakeDocument;
  global.window = fakeWindow;
  global.fetch = () => Promise.reject(new Error("no network in tests"));

  const modulePath = require.resolve("./app.js");
  delete require.cache[modulePath];
  require("./app.js"); // re-executes the file, running the DOM-wiring branch against the fakes above

  const tooltip = fakeDocument.getElementById("info-tooltip");

  infoBtn.dispatch("mouseenter");
  checkTrue("hovering an info icon shows the tooltip", tooltip.hidden === false);

  fakeWindow.dispatch("scroll");
  checkTrue("scrolling the page immediately hides an open tooltip, with no cursor movement at all",
    tooltip.hidden === true);

  let threw = false;
  try { fakeWindow.dispatch("scroll"); } catch { threw = true; }
  checkTrue("scrolling again with no tooltip open is a safe no-op", !threw);

  // The other dismissal paths this fix must not have broken.
  infoBtn.dispatch("mouseenter");
  infoBtn.dispatch("mouseleave");
  checkTrue("mouseleave still hides the tooltip (a separate, still-working path)", tooltip.hidden === true);

  infoBtn.dispatch("focus");
  checkTrue("keyboard focus still shows the tooltip", tooltip.hidden === false);
  infoBtn.dispatch("blur");
  checkTrue("blur still hides the tooltip", tooltip.hidden === true);

  infoBtn.dispatch("click", { preventDefault(){} });
  checkTrue("a click/tap still toggles the tooltip open (the touch-device path)", tooltip.hidden === false);
  fakeDocument.dispatch("keydown", { key: "Escape" });
  checkTrue("Escape still closes the tooltip", tooltip.hidden === true);

  delete global.document;
  delete global.window;
  delete global.fetch;
  delete require.cache[modulePath];
  require("./app.js"); // restore app.js to its normal no-DOM export state for anything running after
})();

console.log(failures === 0 ? "\nALL CHECKS PASS" : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
