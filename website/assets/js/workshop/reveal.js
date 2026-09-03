/* ==========================================================================
   negaWatt Belgium — workshop reveal screen (reveal.html)
   --------------------------------------------------------------------------
   The projector. Group answers arrive live (a 3-second poll), then the
   facilitator drops in the negaWatt value with its written justification, one
   lever at a time.

     reveal.html?topic=inland-mobility

   Which answers are shown is a question of *when*, not of which session: the
   date filter selects the groups that started inside its window. It opens on
   today — one workshop — and widening the start date summarises every sitting
   ever run on the topic.
   ========================================================================== */
(function () {
  "use strict";

  var T = window.NW_I18N;
  var API = window.NW_API;
  var POLL_MS = 3000;

  var state = {
    topic: null, sector: null, order: [], index: 0,
    from: null, to: null,       // local 'YYYY-MM-DD', or null for an open end
    revealed: {},               // leverId -> true once the nW value is shown
    results: null,
    labels: {},                 // groupId -> the name to draw
    summary: false,
    timer: null
  };

  function $(id) { return document.getElementById(id); }
  function param(name) {
    var m = new RegExp("[?&]" + name + "=([^&]*)").exec(window.location.search);
    return m ? decodeURIComponent(m[1]) : null;
  }
  function levers() {
    var d = window.NW_LEVERS && window.NW_LEVERS[state.sector];
    return (d && d.levers) || {};
  }
  function topicContent() {
    var c = window.NW_WS_CONTENT;
    return (c && c.topics && c.topics[state.topic]) || null;
  }
  function leverContent(id) {
    var t = topicContent();
    return (t && t.levers && t.levers[id]) || {};
  }
  function current() { return state.order[state.index]; }

  function applyStaticText() {
    document.querySelectorAll("[data-ui]").forEach(function (node) {
      node.textContent = T.t(node.getAttribute("data-ui"));
    });
    $("from").setAttribute("aria-label", T.t("reveal.filter.from"));
    $("to").setAttribute("aria-label", T.t("reveal.filter.to"));
  }

  function buildLangSwitch() {
    var box = $("lang-switch");
    box.innerHTML = "";
    T.available().forEach(function (code) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = code;
      b.setAttribute("aria-pressed", code === T.lang() ? "true" : "false");
      b.addEventListener("click", function () {
        T.setLang(code); buildLangSwitch(); applyStaticText(); showWindow(); render();
      });
      box.appendChild(b);
    });
  }

  /* ------------------------------------------------------------------ dates */
  function pad(n) { return (n < 10 ? "0" : "") + n; }

  function localToday() {
    var d = new Date();
    return d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate());
  }

  /* The filter is a pair of local dates; the API window is UTC. Going through a
     real Date is what makes an evening workshop count as that evening rather
     than as the next UTC day. */
  function toUtc(date, endOfDay) {
    if (!date) return null;
    var p = date.split("-");
    var d = new Date(+p[0], +p[1] - 1, +p[2],
                     endOfDay ? 23 : 0, endOfDay ? 59 : 0, endOfDay ? 59 : 0);
    return d.toISOString().slice(0, 19).replace("T", " ");
  }

  function pretty(date) {
    var p = date.split("-");
    return new Date(+p[0], +p[1] - 1, +p[2])
      .toLocaleDateString(document.documentElement.lang || undefined,
                          { day: "numeric", month: "short", year: "numeric" });
  }

  function showWindow() {
    var badge = $("window-badge");
    if (!state.from && !state.to) badge.textContent = T.t("reveal.filter.allLabel");
    else if (state.from === state.to) badge.textContent = pretty(state.from);
    else {
      badge.textContent = (state.from ? pretty(state.from) : "…") + " – " +
                          (state.to ? pretty(state.to) : "…");
    }
    $("from").value = state.from || "";
    $("to").value = state.to || "";
  }

  function setWindow(from, to) {
    state.from = from || null;
    state.to = to || null;
    state.results = null;
    showWindow();
    render();
    startPolling();
  }

  /* ------------------------------------------------------------------ stats */

  /* Two tables that both called themselves "Table 3", or a window spanning
     several sittings, would draw two dots with one label. Number the duplicates
     rather than hiding one of them. */
  function relabel() {
    var groups = (state.results && state.results.groups) || [];
    var count = {};
    groups.forEach(function (g) {
      var name = g.name || ("#" + g.id);
      count[name] = (count[name] || 0) + 1;
    });
    var nth = {};
    var labels = {};
    groups.forEach(function (g) {
      var name = g.name || ("#" + g.id);
      nth[name] = (nth[name] || 0) + 1;
      labels[g.id] = count[name] > 1 ? name + " ·" + nth[name] : name;
    });
    state.labels = labels;
  }

  function answersFor(leverId) {
    if (!state.results) return [];
    return (state.results.answers || [])
      .filter(function (a) { return a.lever_id === leverId; })
      .map(function (a) {
        return { label: state.labels[a.group_id] || ("#" + a.group_id), value: a.value,
                 confidence: a.confidence, condition: a.condition };
      });
  }

  function mean(values) {
    if (!values.length) return null;
    return values.reduce(function (s, v) { return s + v; }, 0) / values.length;
  }

  /* ------------------------------------------------------------------ render */
  function render() {
    if (state.summary) { renderSummary(); return; }

    var id = current();
    if (!id || !levers()[id]) return;

    $("screen-lever").classList.remove("ws-hidden");
    $("screen-summary").classList.add("ws-hidden");
    $("actions").classList.remove("ws-hidden");

    var lever = levers()[id];
    var content = leverContent(id);
    var rows = answersFor(id);
    var values = rows.map(function (r) { return r.value; });
    var avg = mean(values);
    var shown = !!state.revealed[id];

    var t = topicContent();
    $("topic-title").textContent = t ? T.pick(t.title) : "";
    $("question").textContent = T.pick(content.question) || lever.name;
    $("subtitle").textContent = T.pick(content.subtitle);

    window.NW_SPARK.dots($("dots"), {
      min: lever.slider.min, max: lever.slider.max,
      values: rows, mean: avg, nw: lever.targetValue, showNw: shown,
      decimals: lever.decimals, srLabel: T.pick(content.question)
    });

    var legend = $("dots-legend");
    legend.innerHTML = "";
    function chip(cls, text) {
      var s = document.createElement("span");
      s.className = cls;
      s.innerHTML = "<i></i>" + text;
      legend.appendChild(s);
    }
    chip("obs", rows.length === 1 ? T.t("reveal.groupsOne")
                                  : T.t("reveal.groups", { n: rows.length }));
    if (avg !== null) chip("nw", T.t("reveal.mean") + " " + T.num(avg, lever.decimals));
    if (shown) chip("you", T.t("reveal.nwValue"));

    var stats = $("stats");
    stats.innerHTML = "";
    function stat(value, label) {
      var d = document.createElement("div");
      d.innerHTML = "<b>" + value + "</b>" + label;
      stats.appendChild(d);
    }
    if (!rows.length) {
      stats.innerHTML = '<div class="muted">' + T.t("reveal.waiting") + "</div>";
    } else {
      stat(String(rows.length), T.t("reveal.groupsLabel"));
      stat(T.num(avg, lever.decimals) + " " + T.unit(lever.unit), T.t("reveal.mean"));
      stat(T.num(Math.min.apply(null, values), lever.decimals) + " – " +
           T.num(Math.max.apply(null, values), lever.decimals), "min – max");
      if (shown) {
        stat(T.num(lever.targetValue, lever.decimals) + " " + T.unit(lever.unit),
             T.t("reveal.nwValue"));
      }
    }

    $("nw-panel").classList.toggle("ws-hidden", !shown);
    if (!shown) {
      // clear rather than merely hide, so a stale justification can never be
      // read off the previous lever
      $("justification").textContent = "";
      $("debate").textContent = "";
      $("verdict").textContent = "";
      $("reveal-facts").innerHTML = "";
    }
    if (shown) {
      T.rich($("justification"), T.pick(content.justification));
      if (content.debate) {
        $("debate-box").classList.remove("ws-hidden");
        T.rich($("debate"), T.pick(content.debate));
      } else {
        $("debate-box").classList.add("ws-hidden");
      }
      // the facts held back from the participants: what the scenario itself assumes
      var extra = $("reveal-facts");
      extra.innerHTML = "";
      (content.facts || []).filter(function (f) { return f.reveal; }).forEach(function (fact) {
        var card = document.createElement("article");
        card.className = "ws-fact";
        card.dataset.kind = fact.kind || "structure";
        var kind = document.createElement("span");
        kind.className = "ws-fact__kind";
        kind.textContent = T.t("play.facts.kind." + (fact.kind || "structure"));
        card.appendChild(kind);
        var text = document.createElement("p");
        text.className = "ws-fact__text";
        T.rich(text, T.pick(fact.text));
        card.appendChild(text);
        if (fact.source) {
          var src = document.createElement("p");
          src.className = "ws-fact__source";
          src.textContent = T.t("common.source") + ": " + fact.source;
          card.appendChild(src);
        }
        extra.appendChild(card);
      });

      var verdict = "";
      if (avg !== null) {
        var gap = avg - lever.targetValue;
        var bolder = lever.better === "up" ? gap > 0 : gap < 0;
        var span = lever.slider.max - lever.slider.min;
        verdict = Math.abs(gap) < span * 0.02 ? T.t("reveal.aligned")
                : (bolder ? T.t("reveal.bolder") : T.t("reveal.shyer"));
      }
      $("verdict").textContent = verdict;
    }

    var quotes = rows.filter(function (r) { return r.condition; });
    $("conditions-box").classList.toggle("ws-hidden", !(shown && quotes.length));
    var box = $("conditions");
    box.innerHTML = "";
    quotes.forEach(function (r) {
      var q = document.createElement("blockquote");
      q.className = "ws-quote";
      // no-break spaces inside the guillemets, French typography
      q.textContent = "« " + r.condition + " »";
      var cite = document.createElement("cite");
      cite.textContent = r.label;
      q.appendChild(cite);
      box.appendChild(q);
    });

    buildProgress();
    $("progress-label").textContent = T.t("play.progress", {
      n: state.index + 1, total: state.order.length
    });
    $("btn-reveal").textContent = shown ? T.t("reveal.next") : T.t("reveal.show");
    $("btn-prev").disabled = state.index === 0;
  }

  function buildProgress() {
    var box = $("progress-dots");
    box.innerHTML = "";
    state.order.forEach(function (id, i) {
      var dot = document.createElement("button");
      dot.type = "button";
      dot.className = "ws-progress__dot";
      dot.dataset.state = i === state.index ? "current"
                        : (state.revealed[id] ? "answered" : "todo");
      dot.setAttribute("aria-label", String(i + 1));
      dot.addEventListener("click", function () { go(i); });
      box.appendChild(dot);
    });
  }

  function renderSummary() {
    $("screen-lever").classList.add("ws-hidden");
    $("screen-summary").classList.remove("ws-hidden");

    var table = $("summary-table");
    table.innerHTML = "";
    var head = table.insertRow();
    ["", T.t("reveal.mean"), T.t("reveal.nwValue"), ""].forEach(function (label) {
      var th = document.createElement("th");
      th.textContent = label;
      head.appendChild(th);
    });

    var perGroup = {};
    state.order.forEach(function (id) {
      var lever = levers()[id];
      var rows = answersFor(id);
      var avg = mean(rows.map(function (r) { return r.value; }));
      var span = lever.slider.max - lever.slider.min;

      var tr = table.insertRow();
      tr.insertCell().textContent = T.pick(leverContent(id).short) ||
                                    T.pick(leverContent(id).question) || lever.name;
      var c1 = tr.insertCell(); c1.className = "num";
      c1.textContent = avg === null ? "—" : T.num(avg, lever.decimals);
      var c2 = tr.insertCell(); c2.className = "num";
      c2.textContent = T.num(lever.targetValue, lever.decimals);
      var c3 = tr.insertCell();
      if (avg !== null) {
        var gap = avg - lever.targetValue;
        var bolder = lever.better === "up" ? gap > 0 : gap < 0;
        // no arrow: "lower value" and "more ambitious" point opposite ways on
        // half these levers, so an arrow would only mislead
        c3.textContent = Math.abs(gap) < span * 0.02 ? T.t("reveal.alignedShort")
                       : (bolder ? T.t("reveal.bolderShort") : T.t("reveal.shyerShort"));
      }

      rows.forEach(function (r) {
        perGroup[r.label] = perGroup[r.label] || { distance: 0, ambition: 0, n: 0 };
        var entry = perGroup[r.label];
        entry.distance += Math.abs(r.value - lever.targetValue) / span;
        var gap = (r.value - lever.targetValue) / span;
        entry.ambition += lever.better === "up" ? gap : -gap;
        entry.n += 1;
      });
    });

    var awards = $("awards");
    awards.innerHTML = "";
    var names = Object.keys(perGroup);
    if (names.length) {
      var closest = names.slice().sort(function (a, b) {
        return perGroup[a].distance / perGroup[a].n - perGroup[b].distance / perGroup[b].n;
      })[0];
      var boldest = names.slice().sort(function (a, b) {
        return perGroup[b].ambition / perGroup[b].n - perGroup[a].ambition / perGroup[a].n;
      })[0];
      var d1 = document.createElement("div");
      d1.innerHTML = "<b>" + closest + "</b>" + T.t("reveal.closest");
      var d2 = document.createElement("div");
      d2.innerHTML = "<b>" + boldest + "</b>" + T.t("reveal.boldest");
      awards.appendChild(d1);
      awards.appendChild(d2);
    }
    $("btn-reveal").textContent = T.t("reveal.summary");
  }

  /* ------------------------------------------------------------------ moves */
  function go(index) {
    state.summary = false;
    state.index = Math.max(0, Math.min(index, state.order.length - 1));
    render();
  }

  function revealOrAdvance() {
    if (state.summary) return;
    var id = current();
    if (!state.revealed[id]) {
      state.revealed[id] = true;
      render();
      return;
    }
    if (state.index === state.order.length - 1) {
      state.summary = true;
      renderSummary();
    } else {
      go(state.index + 1);
    }
  }

  /* ------------------------------------------------------------------- poll */
  function poll() {
    return API.getResults({
      topic: state.topic,
      from: toUtc(state.from, false),
      to: toUtc(state.to, true)
    }).then(function (doc) {
      state.results = doc;
      relabel();
      var n = (doc.groups || []).length;
      $("live-badge").textContent = n === 1 ? T.t("reveal.groupsOne")
                                            : T.t("reveal.groups", { n: n });
      if (!state.summary) render(); else renderSummary();
    }).catch(function () {
      $("live-badge").textContent = T.t("common.error");
    });
  }

  function startPolling() {
    if (state.timer) window.clearInterval(state.timer);
    state.timer = window.setInterval(poll, POLL_MS);
    poll();
  }

  /* ------------------------------------------------------------------- init */
  function fail(message) {
    $("screen-lever").classList.add("ws-hidden");
    $("screen-summary").classList.add("ws-hidden");
    $("actions").classList.add("ws-hidden");
    $("screen-error").classList.remove("ws-hidden");
    $("error-text").textContent = message;
  }

  function init() {
    if (!window.NW_WS_CONTENT) {
      document.body.innerHTML = '<main class="ws-main"><div class="ws-note ws-note--warn">' +
        "workshop_content.js is missing — run scripts/build_workshop_content.py</div></main>";
      return;
    }
    buildLangSwitch();
    applyStaticText();

    state.topic = param("topic") || Object.keys(window.NW_WS_CONTENT.topics)[0];
    var topic = topicContent();
    if (!topic) return fail("Unknown workshop topic: " + state.topic);
    state.sector = topic.sector;
    state.order = (topic.order || []).filter(function (id) { return !!levers()[id]; });
    if (!state.order.length) return fail("No levers to reveal for topic " + state.topic);

    $("btn-prev").addEventListener("click", function () { go(state.index - 1); });
    $("btn-next").addEventListener("click", function () {
      if (state.index === state.order.length - 1) { state.summary = true; renderSummary(); }
      else go(state.index + 1);
    });
    $("btn-reveal").addEventListener("click", revealOrAdvance);
    $("preset-today").addEventListener("click", function () {
      setWindow(localToday(), localToday());
    });
    $("preset-all").addEventListener("click", function () { setWindow(null, null); });
    $("from").addEventListener("change", function () { setWindow(this.value, state.to); });
    $("to").addEventListener("change", function () { setWindow(state.from, this.value); });
    document.addEventListener("keydown", function (e) {
      if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); revealOrAdvance(); }
      if (e.key === "ArrowLeft") { e.preventDefault(); go(state.index - 1); }
    });

    // Today, unless the URL asks otherwise: one workshop, which is the normal case.
    setWindow(param("from") || localToday(), param("to") || localToday());
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }
})();
