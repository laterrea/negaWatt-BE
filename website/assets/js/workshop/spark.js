/* ==========================================================================
   negaWatt Belgium — workshop charts (window.NW_SPARK)
   --------------------------------------------------------------------------
   Two purpose-built inline-SVG charts, deliberately *not* Plotly: participants
   open the play page on a phone over workshop wifi and plotly.min.js is 4.4 MB.
   Inline SVG also prints crisply on the paper fact cards.

     NW_SPARK.history(el, opts)    the observed curve whose 2050 endpoint is the
                                   participant's slider handle
     NW_SPARK.dots(el, opts)       the reveal: one dot per group on a value axis
     NW_SPARK.factChart(el, chart) the optional plot inside an information card

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
    var last = -1;
    function draw() { last = node.clientWidth; fn(); }
    draw();
    /* A card is often built detached and attached a moment later, so the first
       measurement can be zero and the chart would keep a viewBox narrower than
       the box it is stretched into — text twice its intended size. Measure again
       once the browser has laid the card out. */
    if (typeof requestAnimationFrame === "function") {
      requestAnimationFrame(function () {
        if (Math.abs(node.clientWidth - last) > 2) draw();
      });
    }
    if (node.__nwSparkObserver) node.__nwSparkObserver.disconnect();
    if (typeof ResizeObserver === "function") {
      var obs = new ResizeObserver(function () {
        if (Math.abs(node.clientWidth - last) > 2) draw();
      });
      obs.observe(node);
      node.__nwSparkObserver = obs;
    } else {
      window.addEventListener("resize", draw);
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

  /* --------------------------------------------------------------- mini plot */
  /* The small plot a *fact* can carry, declared as a `chart:` block in the
     content YAML. Two shapes cover what an information card needs:

       bars   one value per label — Belgium against its neighbours
       line   a series over time — how something moved

     Inline SVG like everything else here: the play page opens on a phone over
     workshop wifi, and the very same markup prints on the paper cards.

     opts: {kind, x, y, labels, unit, decimals, highlight, height, srLabel} */
  function mini(node, opts) {
    responsive(node, function () {
      (opts.kind === "line" ? drawMiniLine : drawMiniBars)(node, opts);
    });
  }

  function isHighlighted(highlight, i) {
    if (highlight === null || highlight === undefined) return false;
    return Array.isArray(highlight) ? highlight.indexOf(i) !== -1 : highlight === i;
  }

  function drawMiniBars(node, o) {
    var W = Math.max(210, node.clientWidth || 280);
    var ys = (o.y || []).map(Number);
    var labels = o.labels || [];
    var rowH = 17, gap = 7;
    var H = ys.length * (rowH + gap) + 4;
    var labW = Math.min(Math.max(W * 0.28, 46), 104);
    var iw = Math.max(24, W - labW - 46);          // 46px keeps the value legible
    var hi = Math.max.apply(null, ys.concat([0]));
    var lo = Math.min.apply(null, ys.concat([0]));
    var span = (hi - lo) || 1;
    var zero = labW + (-lo / span) * iw;

    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H, width: "100%", height: H,
      role: "img", "aria-label": o.srLabel || "comparison"
    });

    ys.forEach(function (v, i) {
      var y = i * (rowH + gap) + 2;
      var w = Math.abs(v) / span * iw;
      var x = v < 0 ? zero - w : zero;
      var on = isHighlighted(o.highlight, i);
      svg.appendChild(el("rect", {
        x: x, y: y, width: Math.max(w, 1.5), height: rowH, rx: 2,
        fill: on ? AMBER : TEAL, "fill-opacity": on ? 0.95 : 0.5
      }));
      svg.appendChild(el("text", {
        x: labW - 6, y: y + rowH - 4, "text-anchor": "end", "font-size": 11,
        "font-weight": on ? 700 : 500, fill: on ? INK : MUTED
      }, labels[i] === undefined ? "" : String(labels[i])));
      svg.appendChild(el("text", {
        x: x + w + 5, y: y + rowH - 4, "text-anchor": "start", "font-size": 11,
        "font-weight": on ? 800 : 600, fill: on ? AMBER : INK
      }, fmt(v, o.decimals)));
    });

    node.innerHTML = "";
    node.appendChild(svg);
  }

  function drawMiniLine(node, o) {
    var W = Math.max(210, node.clientWidth || 280);
    var H = o.height || 116;
    var mL = 34, mR = 30, mT = 12, mB = 17;
    var iw = W - mL - mR, ih = H - mT - mB;
    var xs = (o.x || []).map(Number);
    var ys = (o.y || []).map(Number);
    var dom = domain(ys);
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs);

    function px(v) { return mL + (x1 === x0 ? iw / 2 : (v - x0) / (x1 - x0) * iw); }
    function py(v) { return mT + ih - (v - dom[0]) / (dom[1] - dom[0]) * ih; }

    var svg = el("svg", {
      viewBox: "0 0 " + W + " " + H, width: "100%", height: H,
      role: "img", "aria-label": o.srLabel || "series"
    });

    [dom[0], dom[1]].forEach(function (v, i) {
      svg.appendChild(el("line", {
        x1: mL, x2: W - mR, y1: py(v), y2: py(v), stroke: LINE,
        "stroke-width": i === 0 ? 1.2 : 1, "stroke-dasharray": i === 0 ? null : "3 4"
      }));
      svg.appendChild(el("text", { x: mL - 6, y: py(v) + 4, "text-anchor": "end",
                                   "font-size": 10, fill: MUTED }, fmt(v, o.decimals)));
    });

    var points = [];
    for (var i = 0; i < xs.length; i++) {
      if (ys[i] === null || ys[i] === undefined || !isFinite(ys[i])) continue;
      points.push([px(xs[i]), py(ys[i])]);
    }
    if (points.length > 1) {
      svg.appendChild(el("polyline", {
        points: points.map(function (p) { return p.join(","); }).join(" "),
        fill: "none", stroke: TEAL, "stroke-width": 2.2,
        "stroke-linejoin": "round", "stroke-linecap": "round"
      }));
    }
    points.forEach(function (p, i) {
      var on = isHighlighted(o.highlight, i) || i === points.length - 1;
      if (!on && points.length > 2) return;
      svg.appendChild(el("circle", { cx: p[0], cy: p[1], r: on ? 4 : 3,
                                     fill: on ? AMBER : TEAL }));
    });
    if (points.length) {
      var last = points[points.length - 1];
      svg.appendChild(el("text", {
        x: Math.min(last[0] + 6, W - 2), y: last[1] + 4, "text-anchor": "end",
        "font-size": 11, "font-weight": 700, fill: AMBER
      }, fmt(ys[ys.length - 1], o.decimals)));
    }

    [x0, x1].forEach(function (yr, i) {
      if (i && yr === x0) return;
      svg.appendChild(el("text", {
        x: px(yr), y: H - 4, "text-anchor": i ? "end" : "start",
        "font-size": 10, fill: MUTED
      }, String(yr)));
    });

    node.innerHTML = "";
    node.appendChild(svg);
  }

  /* Build the <figure> for a fact's `chart:` block and append it to `parent`.
     Called by the play screen, the printable cards and the reveal, so one
     `chart:` in the YAML shows up in all three without touching any of them. */
  function factChart(parent, chart, opts) {
    if (!parent || !chart || !chart.y || !chart.y.length) return null;
    opts = opts || {};
    var T = window.NW_I18N;
    var pick = T ? T.pick : function (v) {
      return typeof v === "string" ? v : (v && (v.en || v.fr)) || "";
    };

    var fig = document.createElement("figure");
    fig.className = "ws-fact__chart" + (opts.className ? " " + opts.className : "");
    var plot = document.createElement("div");
    fig.appendChild(plot);

    var caption = document.createElement("figcaption");
    var unit = chart.unit ? (T ? T.unit(chart.unit) : chart.unit) : "";
    var head = [unit, pick(chart.caption)].filter(Boolean).join(" · ");
    if (head) caption.appendChild(document.createTextNode(head));
    // A chart carries its own source when it plots something other than the
    // sentence above it; otherwise the fact's source already covers it.
    if (chart.source) {
      var lead = document.createElement("span");
      lead.textContent = (head ? " — " : "") +
                         (T ? T.t("common.source") : "Source") + ": ";
      caption.appendChild(lead);
      if (chart.url) {
        var a = document.createElement("a");
        a.href = chart.url;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.textContent = chart.source;
        caption.appendChild(a);
      } else {
        caption.appendChild(document.createTextNode(chart.source));
      }
    }
    if (caption.childNodes.length) fig.appendChild(caption);

    parent.appendChild(fig);
    mini(plot, {
      kind: chart.kind, x: chart.x, y: chart.y,
      labels: (chart.labels || []).map(pick),
      unit: chart.unit, decimals: chart.decimals, highlight: chart.highlight,
      height: opts.height || chart.height,
      srLabel: pick(chart.caption) || unit
    });
    return fig;
  }

  window.NW_SPARK = { history: history, dots: dots, mini: mini,
                      factChart: factChart, fmt: fmt, domain: domain };
})();
