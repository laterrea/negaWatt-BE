/* ==========================================================================
   render.js — binds the data exported by the notebooks (window.NW_DATA) to the
   hand-written HTML cards, and draws the simple Plotly charts.

   Card markup expected on a page:
     <article class="hyp-card" id="car-occupancy" data-hyp="transport.car-occupancy">
       <div class="hyp-card__head"> ... title (hand written) ... </div>
       <div data-role="values"></div>     <- value strip injected here
       <div class="hyp-chart" data-role="chart"></div>  (optional)
       <div class="hyp-card__body"> ...discussion (hand written)... </div>
       <div class="hyp-card__foot">
         <span class="ref" data-field="ref"></span>
         <a data-role="deeplink">Calculation details</a>
       </div>
     </article>

   A page only needs to <script> the relevant data/<sector>.js file(s) before
   this file. Numbers, units, badges and charts then come entirely from the
   notebooks; all prose stays in the HTML.
   ========================================================================== */
(function () {
  "use strict";

  var DATA = window.NW_DATA || {};
  var TEAL = "#1a9c98";
  var TEAL_LIGHT = "#9fd3cf";
  var GREY = "#c4cecd";

  /* ---------- number formatting -------------------------------------- */
  function fmt(v) {
    if (v === null || v === undefined || v === "") return "—";
    if (typeof v === "string") return v;
    var a = Math.abs(v);
    var opts;
    if (a >= 1000)      opts = { maximumFractionDigits: 0 };
    else if (a >= 100)  opts = { maximumFractionDigits: 1 };
    else if (a >= 1)    opts = { maximumFractionDigits: 2 };
    else                opts = { maximumFractionDigits: 3 };
    return v.toLocaleString("en-US", opts);
  }

  function lookup(key) {
    if (!key) return null;
    var parts = key.split(".");
    var sector = DATA[parts[0]];
    if (!sector || !sector.hypotheses) return null;
    return sector.hypotheses[parts.slice(1).join(".")] || null;
  }

  /* ---------- value strip + badge ------------------------------------ */
  function buildValues(h) {
    var refDisp    = h.displayRef    !== undefined ? h.displayRef    : fmt(h.refValue);
    var targetDisp = h.displayTarget !== undefined ? h.displayTarget : fmt(h.targetValue);
    var unit = h.unit ? '<div class="unit">' + h.unit + "</div>" : "";

    // Change badge: prefer explicit label/direction, else derive from values.
    var dir = h.direction;
    var label = h.changeLabel;
    if (label === undefined) {
      if (typeof h.pctChange === "number") {
        var sign = h.pctChange > 0 ? "+" : (h.pctChange < 0 ? "\u2212" : "");
        label = sign + Math.abs(Math.round(h.pctChange)) + "%";
      } else { label = ""; }
    }
    if (dir === undefined) {
      var diff = (typeof h.pctChange === "number")
        ? h.pctChange
        : (Number(h.targetValue) - Number(h.refValue));
      dir = diff < -0.0001 ? "down" : (diff > 0.0001 ? "up" : "flat");
    }
    var arrow = dir === "down" ? "\u2193" : (dir === "up" ? "\u2191" : "\u2192");
    var badge = label
      ? '<span class="badge ' + dir + '">' + arrow + " " + label + "</span>"
      : "";

    return '' +
      '<div class="val-box">' +
        '<div class="yr">' + (h.refYear || "Now") + "</div>" +
        '<div class="num">' + refDisp + "</div>" + unit +
      "</div>" +
      '<div class="val-arrow">\u2192</div>' +
      '<div class="val-box">' +
        '<div class="yr">' + (h.targetYear || "Target") + "</div>" +
        '<div class="num">' + targetDisp + "</div>" + unit +
      "</div>" +
      badge;
  }

  /* ---------- mini per-card chart ------------------------------------ */
  function drawMini(el, h) {
    if (!window.Plotly) return;
    var refV = Number(h.refValue), tgtV = Number(h.targetValue);
    if (!isFinite(refV) || !isFinite(tgtV)) { el.style.display = "none"; return; }
    var data = [{
      type: "bar",
      x: [String(h.refYear || "Now"), String(h.targetYear || "Target")],
      y: [refV, tgtV],
      marker: { color: [GREY, TEAL] },
      text: [fmt(refV), fmt(tgtV)],
      textposition: "outside",
      hovertemplate: "%{x}: %{y}<extra></extra>",
      cliponaxis: false
    }];
    var layout = {
      margin: { l: 8, r: 8, t: 14, b: 22 },
      height: el.clientHeight || 150,
      xaxis: { fixedrange: true, tickfont: { size: 11, color: "#768584" } },
      yaxis: { visible: false, fixedrange: true, range: [0, Math.max(refV, tgtV) * 1.22] },
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      showlegend: false, bargap: 0.45
    };
    Plotly.newPlot(el, data, layout, { displayModeBar: false, responsive: true, staticPlot: false });
  }

  /* ---------- bind one card ------------------------------------------ */
  function bindCard(card) {
    var h = lookup(card.getAttribute("data-hyp"));
    var valuesEl = card.querySelector('[data-role="values"]');
    if (!h) {
      card.classList.add("data-missing");
      if (valuesEl) {
        valuesEl.innerHTML = '<div class="note small">Data pending &mdash; run the notebook export cell.</div>';
      }
      return;
    }
    if (valuesEl) { valuesEl.className = "hyp-values"; valuesEl.innerHTML = buildValues(h); }

    card.querySelectorAll('[data-field]').forEach(function (node) {
      var f = node.getAttribute("data-field");
      if (h[f] !== undefined && node.textContent.trim() === "") node.textContent = h[f];
    });

    var chartEl = card.querySelector('[data-role="chart"]');
    if (chartEl) drawMini(chartEl, h);

    var link = card.querySelector('[data-role="deeplink"]');
    if (link && h.notebook) link.setAttribute("href", h.notebook);
  }

  /* ---------- larger sector figures ---------------------------------- */
  function drawFigure(el) {
    if (!window.Plotly) return;
    var key = el.getAttribute("data-plot");           // e.g. "transport.modalShare"
    var parts = key.split(".");
    var sector = DATA[parts[0]];
    var spec = sector && sector.plots ? sector.plots[parts.slice(1).join(".")] : null;
    if (!spec) { el.innerHTML = '<div class="note small">Chart data pending.</div>'; return; }

    var traces, layout;
    var palette = [TEAL, "#137c79", "#7bc4bf", "#c0612a", "#e0a93b", "#5a8f8c", "#b9d8d4", "#9a6b3f", "#3f6b69"];

    if (spec.type === "groupedBar" || spec.type === "stackedBar") {
      traces = (spec.series || []).map(function (s, i) {
        return {
          type: "bar", name: s.name, x: spec.x, y: s.y,
          marker: { color: palette[i % palette.length] }
        };
      });
      layout = {
        barmode: spec.type === "stackedBar" ? "stack" : "group",
        margin: { l: 56, r: 16, t: 10, b: 40 },
        xaxis: { fixedrange: true, tickfont: { size: 12 } },
        yaxis: { fixedrange: true, title: spec.yTitle || "", gridcolor: "#eef2f1" },
        legend: { orientation: "h", y: -0.18, font: { size: 11 } },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)"
      };
    } else { // simple line
      traces = (spec.series || []).map(function (s, i) {
        return { type: "scatter", mode: "lines+markers", name: s.name, x: spec.x, y: s.y,
                 line: { color: palette[i % palette.length], width: 3 } };
      });
      layout = {
        margin: { l: 56, r: 16, t: 10, b: 40 },
        xaxis: { fixedrange: true }, yaxis: { fixedrange: true, title: spec.yTitle || "", gridcolor: "#eef2f1" },
        legend: { orientation: "h", y: -0.18, font: { size: 11 } },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)"
      };
    }
    Plotly.newPlot(el, traces, layout, { displayModeBar: false, responsive: true });
  }

  /* ---------- auto-rendered list (for stub/replica pages) ----------- */
  // <div data-hyp-list="buildings"></div> builds a compact card for every
  // hypothesis exported by that sector, with no hand-written discussion.
  function buildList(host) {
    var sectorKey = host.getAttribute("data-hyp-list");
    var sector = DATA[sectorKey];
    if (!sector || !sector.hypotheses) {
      host.innerHTML = '<div class="stub-banner">Data pending &mdash; run the ' +
        sectorKey + " notebook's <em>Website export</em> cell to populate this section.</div>";
      return;
    }
    var grid = document.createElement("div");
    grid.className = "hyp-grid";
    Object.keys(sector.hypotheses).forEach(function (id) {
      var h = sector.hypotheses[id];
      var card = document.createElement("article");
      card.className = "hyp-card";
      card.id = id;
      card.innerHTML =
        '<div class="hyp-card__head">' +
          '<div class="hyp-card__icon"><svg><use href="#i-leaf"></use></svg></div>' +
          '<div class="hyp-card__title"><h3>' + (h.name || id) + "</h3>" +
            '<div class="hyp-card__cat">' + (h.category || "") + "</div></div>" +
        "</div>" +
        '<div class="hyp-values">' + buildValues(h) + "</div>" +
        '<div class="hyp-chart"></div>' +
        (h.reference || h.notebook ?
          '<div class="hyp-card__foot"><span class="ref">' + (h.reference || "") + "</span>" +
          (h.notebook ? '<a href="' + h.notebook + '">Calculation details &rarr;</a>' : "") +
          "</div>" : "");
      grid.appendChild(card);
      var chartHost = card.querySelector(".hyp-chart");
      if (chartHost) drawMini(chartHost, h);
    });
    host.innerHTML = "";
    host.appendChild(grid);
  }

  /* ---------- go --------------------------------------------------- */
  function init() {
    document.querySelectorAll(".hyp-card[data-hyp]").forEach(bindCard);
    document.querySelectorAll("[data-hyp-list]").forEach(buildList);
    document.querySelectorAll("[data-plot]").forEach(drawFigure);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }
})();
