/* ==========================================================================
   negaWatt Belgium — workshop charts (window.NW_SPARK)
   --------------------------------------------------------------------------
   Two purpose-built inline-SVG charts, deliberately *not* Plotly: participants
   open the play page on a phone over workshop wifi and plotly.min.js is 4.4 MB.
   Inline SVG also prints crisply on the paper fact cards.

     NW_SPARK.history(el, opts)  the observed curve whose 2050 endpoint is the
                                 participant's slider handle
     NW_SPARK.dots(el, opts)     the reveal: one dot per group on a value axis

   Both redraw on resize and are safe to call repeatedly on the same element.
   ========================================================================== */
(function () {
  "use strict";

  var NS = "http://www.w3.org/2000/svg";
  var TEAL = "#1a9c98";
  var TEAL_DARK = "#137c79";
  var AMBER = "#c0612a";
  var LINE = "#d9e0de";
  var MUTED = "#768584";
  var INK = "#20302f";

  function el(name, attrs, text) {
    var node = document.createElementNS(NS, name);
    for (var k in attrs) {
      if (attrs[k] !== null && attrs[k] !== undefined) {
        node.setAttribute(k, String(attrs[k]));
      }
    }
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  /* Follow the page language, which i18n.js stamps on <html lang>. Chart labels
     sitting next to French text must not read "2.00" where the prose says "2,00". */
  var LOCALES = { fr: "fr-BE", nl: "nl-BE", en: "en-GB" };

  function locale() {
    var lang = (document.documentElement.getAttribute("lang") || "en").slice(0, 2);
    return LOCALES[lang] || "en-GB";
  }

  function fmt(v, decimals) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    var d = decimals === undefined ? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2)
                                   : decimals;
    return v.toLocaleString(locale(), { minimumFractionDigits: d, maximumFractionDigits: d });
  }

  /* Pick a y-domain that shows the curve *and* the participant's endpoint
     without letting either squash the other flat. */
  function domain(values) {
    var vals = values.filter(function (v) { return v !== null && isFinite(v); });
    if (!vals.length) return [0, 1];
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    if (hi === lo) { lo -= Math.abs(lo) * 0.1 || 1; hi += Math.abs(hi) * 0.1 || 1; }
    var pad = (hi - lo) * 0.12;
    lo -= pad; hi += pad;
    if (lo > 0 && lo < (hi - lo) * 0.45) lo = 0;   // prefer a zero baseline when close
    return [lo, hi];
  }

  /* Redraw `fn` whenever the element's width changes. */
  function responsive(node, fn) {
    fn();
    if (node.__nwSparkObserver) node.__nwSparkObserver.disconnect();
    if (typeof ResizeObserver === "function") {
      var last = node.clientWidth;
      var obs = new ResizeObserver(function () {
        if (Math.abs(node.clientWidth - last) > 2) { last = node.clientWidth; fn(); }
      });
      obs.observe(node);
      node.__nwSparkObserver = obs;
    } else {
      window.addEventListener("resize", fn);
    }
  }

  /* ---------------------------------------------------------------- history */
  /* opts: {x, y, unit, refYear, refValue, targetYear, value, targetValue,
            showTarget, decimals, srLabel} */
  function history(node, opts) {
    responsive(node, function () { drawHistory(node, opts); });
  }

  function drawHistory(node, o) {
    var W = Math.max(240, node.clientWidth || 320);
    var H = o.height || 190;
    var mL = 44, mR = 16, mT = 14, mB = 26;
    var iw = W - mL - mR, ih = H - mT - mB;

    var xs = (o.x || []).slice();
    var ys = (o.y || []).slice();
    if (!xs.length) { xs = [o.refYear]; ys = [o.refValue]; }

    var x0 = Math.min(xs[0], o.refYear);
    var x1 = o.targetYear;
    var shown = ys.slice();
    if (o.value !== null && o.value !== undefined) shown.push(o.value);
    if (o.showTarget && o.targetValue !== null && o.targetValue !== undefined) {
      shown.push(o.targetValue);
    }
    // With no observed series there is nothing to scale to, and a lone dot in an
    // empty box looks broken. Fall back to the span of possible answers, so the
    // 2019 anchor sits in a meaningful place and the projection has room.
    var dom = (o.domainMin !== undefined && o.domainMax !== undefined && xs.length <= 1)
      ? [o.domainMin, o.domainMax]
      : domain(shown);

    function px(year) { return mL + (year - x0) / (x1 - x0) * iw; }
    function py(v) { return mT + ih - (v - dom[0]) / (dom[1] - dom[0]) * ih; }

    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H, width: "100%", height: H,
      role: "img", "aria-label": o.srLabel || "Historical trend"
    });

    /* plot frame: a baseline and a light top gridline, nothing more */
    [dom[0], (dom[0] + dom[1]) / 2, dom[1]].forEach(function (v, i) {
      svg.appendChild(el("line", {
        x1: mL, x2: W - mR, y1: py(v), y2: py(v),
        stroke: LINE, "stroke-width": i === 0 ? 1.2 : 1,
        "stroke-dasharray": i === 0 ? null : "3 4"
      }));
    });
    svg.appendChild(el("text", { x: mL - 7, y: py(dom[1]) + 4, "text-anchor": "end",
                                "font-size": 10, fill: MUTED }, fmt(dom[1], o.decimals)));
    svg.appendChild(el("text", { x: mL - 7, y: py(dom[0]) + 4, "text-anchor": "end",
                                "font-size": 10, fill: MUTED }, fmt(dom[0], o.decimals)));

    /* observed series, broken wherever the data has a hole */
    var runs = [], run = [];
    for (var i = 0; i < xs.length; i++) {
      if (ys[i] === null || ys[i] === undefined || !isFinite(ys[i])) {
        if (run.length) { runs.push(run); run = []; }
      } else {
        run.push([px(xs[i]), py(ys[i])]);
      }
    }
    if (run.length) runs.push(run);
    runs.forEach(function (r) {
      if (r.length === 1) {
        svg.appendChild(el("circle", { cx: r[0][0], cy: r[0][1], r: 3, fill: TEAL }));
      } else {
        svg.appendChild(el("polyline", {
          points: r.map(function (p) { return p.join(","); }).join(" "),
          fill: "none", stroke: TEAL, "stroke-width": 2.4,
          "stroke-linejoin": "round", "stroke-linecap": "round"
        }));
      }
    });

    /* the reference-year marker: a tick, never a default value for the slider */
    if (isFinite(o.refValue)) {
      svg.appendChild(el("circle", { cx: px(o.refYear), cy: py(o.refValue), r: 3.5,
                                     fill: "#fff", stroke: TEAL_DARK, "stroke-width": 2 }));
      // Without a curve the single point needs naming, or the chart reads as empty.
      if (xs.length <= 1) {
        svg.appendChild(el("text", {
          x: px(o.refYear) + 9, y: py(o.refValue) + 4, "text-anchor": "start",
          "font-size": 11, "font-weight": 700, fill: TEAL_DARK
        }, String(o.refYear) + " · " + fmt(o.refValue, o.decimals)));
      }
    }

    var lastRun = runs.length ? runs[runs.length - 1] : null;
    var anchor = lastRun ? lastRun[lastRun.length - 1] : [px(o.refYear), py(o.refValue)];

    /* the participant's projection */
    if (o.value !== null && o.value !== undefined && isFinite(o.value)) {
      svg.appendChild(el("line", {
        x1: anchor[0], y1: anchor[1], x2: px(x1), y2: py(o.value),
        stroke: AMBER, "stroke-width": 2.4, "stroke-dasharray": "6 4",
        "stroke-linecap": "round"
      }));
      svg.appendChild(el("circle", { cx: px(x1), cy: py(o.value), r: 5.5, fill: AMBER }));
      var lab = el("text", {
        x: px(x1), y: py(o.value) - 11, "text-anchor": "end",
        "font-size": 12, "font-weight": 700, fill: AMBER
      }, fmt(o.value, o.decimals));
      svg.appendChild(lab);
    }

    /* the negaWatt value, only once revealed */
    if (o.showTarget && isFinite(o.targetValue)) {
      svg.appendChild(el("line", {
        x1: anchor[0], y1: anchor[1], x2: px(x1), y2: py(o.targetValue),
        stroke: TEAL_DARK, "stroke-width": 2.4, "stroke-dasharray": "2 3"
      }));
      svg.appendChild(el("circle", { cx: px(x1), cy: py(o.targetValue), r: 5.5,
                                     fill: TEAL_DARK }));
      svg.appendChild(el("text", {
        x: px(x1), y: py(o.targetValue) + 18, "text-anchor": "end",
        "font-size": 12, "font-weight": 700, fill: TEAL_DARK
      }, "nW " + fmt(o.targetValue, o.decimals)));
    }

    /* x labels: first observed, the reference year, and the horizon */
    var marks = [xs[0], o.refYear, x1];
    marks.filter(function (v, i, a) { return a.indexOf(v) === i; }).forEach(function (yr) {
      svg.appendChild(el("text", {
        x: Math.min(Math.max(px(yr), mL + 8), W - mR - 8),
        y: H - 8, "text-anchor": yr === x1 ? "end" : (yr === xs[0] ? "start" : "middle"),
        "font-size": 10, fill: MUTED
      }, String(yr)));
    });

    node.innerHTML = "";
    node.appendChild(svg);
  }

  /* ------------------------------------------------------------------- dots */
  /* opts: {min, max, values:[{label, value, confidence}], nw, mean,
            unit, decimals, showNw, srLabel} */
  function dots(node, opts) {
    responsive(node, function () { drawDots(node, opts); });
  }

  function drawDots(node, o) {
    var W = Math.max(260, node.clientWidth || 320);
    var H = o.height || 182;
    // the top margin has to clear the negaWatt label, which is drawn above the
    // marker: at mT = 18 it was clipped by the top edge of the SVG
    var mL = 18, mR = 18, mT = 30, mB = 34;
    var iw = W - mL - mR;
    var axisY = H - mB;

    var vals = (o.values || []).filter(function (d) {
      return d && d.value !== null && d.value !== undefined && isFinite(d.value);
    });

    function px(v) {
      var t = (v - o.min) / (o.max - o.min);
      return mL + Math.min(Math.max(t, 0), 1) * iw;
    }

    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H, width: "100%", height: H,
      role: "img", "aria-label": o.srLabel || "Group answers"
    });

    svg.appendChild(el("line", { x1: mL, x2: W - mR, y1: axisY, y2: axisY,
                                 stroke: LINE, "stroke-width": 1.5 }));
    [o.min, o.max].forEach(function (v, i) {
      svg.appendChild(el("line", { x1: px(v), x2: px(v), y1: axisY - 4, y2: axisY + 4,
                                   stroke: LINE, "stroke-width": 1.5 }));
      svg.appendChild(el("text", { x: px(v), y: axisY + 18,
                                   "text-anchor": i ? "end" : "start",
                                   "font-size": 10, fill: MUTED }, fmt(v, o.decimals)));
    });

    /* stack dots that would overlap, so a cluster reads as a cluster */
    var R = 8, occupied = [];
    vals.slice().sort(function (a, b) { return a.value - b.value; }).forEach(function (d) {
      var cx = px(d.value), row = 0;
      while (occupied.some(function (p) {
        return p.row === row && Math.abs(p.cx - cx) < R * 2.1;
      })) { row++; }
      occupied.push({ cx: cx, row: row });
      var cy = axisY - R - 2 - row * (R * 2 + 3);
      var conf = d.confidence || 2;
      var g = el("g", { class: "ws-dot", "data-row": row });
      g.appendChild(el("circle", {
        cx: cx, cy: cy, r: 4 + conf * 1.6, fill: TEAL, "fill-opacity": 0.85,
        stroke: "#fff", "stroke-width": 1.5
      }));
      if (d.label) {
        g.appendChild(el("title", {}, d.label + ": " + fmt(d.value, o.decimals)));
      }
      svg.appendChild(g);
    });

    if (o.mean !== null && o.mean !== undefined && isFinite(o.mean)) {
      svg.appendChild(el("line", {
        x1: px(o.mean), x2: px(o.mean), y1: mT - 6, y2: axisY,
        stroke: TEAL_DARK, "stroke-width": 1.5, "stroke-dasharray": "4 4"
      }));
    }

    if (o.showNw && o.nw !== null && o.nw !== undefined && isFinite(o.nw)) {
      var g2 = el("g", { class: "ws-nw-marker" });
      g2.appendChild(el("line", { x1: px(o.nw), x2: px(o.nw), y1: mT - 10, y2: axisY + 6,
                                  stroke: AMBER, "stroke-width": 3 }));
      g2.appendChild(el("text", {
        x: px(o.nw), y: mT - 16,
        "text-anchor": px(o.nw) > W * 0.75 ? "end" : (px(o.nw) < W * 0.25 ? "start" : "middle"),
        "font-size": 12, "font-weight": 800, fill: AMBER
      }, "négaWatt " + fmt(o.nw, o.decimals)));
      svg.appendChild(g2);
    }

    node.innerHTML = "";
    node.appendChild(svg);
  }

  window.NW_SPARK = { history: history, dots: dots, fmt: fmt, domain: domain };
})();
