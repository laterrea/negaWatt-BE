/* ==========================================================================
   negaWatt Belgium — workshop reveal screen (reveal.html)
   --------------------------------------------------------------------------
   The projector. Group answers arrive live (a 3-second poll), then the
   facilitator drops in the negaWatt value with its written justification, one
   lever at a time.

   Two scopes:
     ?code=ABCD&admin=TOKEN     one session
     ?topic=inland-mobility&key=KEY   every open session on the topic, which is
                                      what makes a fully remote workshop work
   ========================================================================== */
(function () {
  "use strict";

  var T = window.NW_I18N;
  var API = window.NW_API;
  var POLL_MS = 3000;

  var state = {
    scope: null, code: null, adminToken: null, adminKey: null,
    topic: null, sector: null, order: [], index: 0,
    revealed: {},          // leverId -> true once the nW value is shown
    results: null,
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
        T.setLang(code); buildLangSwitch(); applyStaticText(); render();
      });
      box.appendChild(b);
    });
  }

  /* ------------------------------------------------------------------ stats */
  function answersFor(leverId) {
    if (!state.results) return [];
    var names = {};
    (state.results.groups || []).forEach(function (g) {
      names[g.id] = g.name + (state.scope === "topic" && g.session ? " · " + g.session : "");
    });
    return (state.results.answers || [])
      .filter(function (a) { return a.lever_id === leverId; })
      .map(function (a) {
        return { label: names[a.group_id] || ("#" + a.group_id), value: a.value,
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

    // The topic only becomes known once a poll succeeds. If it has not — no
    // facilitator key, a wrong code, the network down — there is nothing to draw
    // and the projector must show the waiting screen, not crash on it.
    var id = current();
    if (!id || !levers()[id]) {
      $("screen-lever").classList.add("ws-hidden");
      $("screen-summary").classList.add("ws-hidden");
      $("actions").classList.add("ws-hidden");
      $("screen-setup").classList.remove("ws-hidden");
      return;
    }

    $("screen-lever").classList.remove("ws-hidden");
    $("screen-summary").classList.add("ws-hidden");
    $("screen-setup").classList.add("ws-hidden");
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
      q.textContent = "\u00ab\u00a0" + r.condition + "\u00a0\u00bb";
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
    $("screen-setup").classList.add("ws-hidden");

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
    pushStep();
    render();
  }

  function pushStep() {
    // keep any second screen on the same lever; harmless if we are not admin
    if (state.scope !== "session" || !state.adminToken) return;
    API.setReveal(state.code, state.adminToken, state.index).catch(function () {});
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
    var opts = state.scope === "topic"
      ? { topic: state.topic, adminKey: state.adminKey }
      : { code: state.code, adminToken: state.adminToken };
    return API.getResults(opts).then(function (doc) {
      state.results = doc;
      if (!state.topic) state.topic = doc.topic;
      var badge = $("live-badge");
      badge.textContent = (doc.groups || []).length === 1
        ? T.t("reveal.groupsOne") : T.t("reveal.groups", { n: (doc.groups || []).length });
      if (state.scope === "topic") {
        badge.textContent += " · " + T.t("reveal.sessions", { n: (doc.sessions || []).length });
      }
      if (!state.summary) render();
    }).catch(function (err) {
      $("live-badge").textContent = T.t("common.error");
      if (err && err.status === 403) showSetup(T.t("reveal.needKey"));
    });
  }

  function startPolling() {
    if (state.timer) window.clearInterval(state.timer);
    state.timer = window.setInterval(poll, POLL_MS);
    poll();
  }

  /* ------------------------------------------------------------------ setup */
  function showSetup(message) {
    if (state.timer) window.clearInterval(state.timer);
    $("screen-lever").classList.add("ws-hidden");
    $("screen-summary").classList.add("ws-hidden");
    $("screen-setup").classList.remove("ws-hidden");
    $("actions").classList.add("ws-hidden");
    if (message) {
      $("setup-error").textContent = message;
      $("setup-error").classList.remove("ws-hidden");
    }
    $("setup-help").textContent = "API: " + API.base();
  }

  function beginFromSetup() {
    var code = $("setup-code").value.trim().toUpperCase();
    var token = $("setup-token").value.trim();
    var key = $("setup-key").value.trim();
    if (key) {
      state.scope = "topic";
      state.adminKey = key;
      state.topic = state.topic || Object.keys(window.NW_WS_CONTENT.topics)[0];
    } else if (code) {
      state.scope = "session";
      state.code = code;
      state.adminToken = token || null;
    } else {
      return;
    }
    prepareOrder();
    startPolling();
  }

  function prepareOrder() {
    var topic = topicContent();
    if (!topic) return;
    state.sector = topic.sector;
    state.order = (topic.order || []).filter(function (id) { return !!levers()[id]; });
  }

  function init() {
    if (!window.NW_WS_CONTENT) {
      document.body.innerHTML = '<main class="ws-main"><div class="ws-note ws-note--warn">' +
        "workshop_content.js is missing — run scripts/build_workshop_content.py</div></main>";
      return;
    }
    buildLangSwitch();
    applyStaticText();

    $("btn-prev").addEventListener("click", function () { go(state.index - 1); });
    $("btn-next").addEventListener("click", function () {
      if (state.index === state.order.length - 1) { state.summary = true; renderSummary(); }
      else go(state.index + 1);
    });
    $("btn-reveal").addEventListener("click", revealOrAdvance);
    $("setup-go").addEventListener("click", beginFromSetup);
    document.addEventListener("keydown", function (e) {
      if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      if (e.key === "ArrowRight" || e.key === " ") { e.preventDefault(); revealOrAdvance(); }
      if (e.key === "ArrowLeft") { e.preventDefault(); go(state.index - 1); }
    });

    state.code = param("code");
    state.adminToken = param("admin");
    state.adminKey = param("key");
    state.topic = param("topic");

    if (state.adminKey) {
      state.scope = "topic";
      state.topic = state.topic || Object.keys(window.NW_WS_CONTENT.topics)[0];
      $("scope-badge").textContent = "topic";
    } else if (state.code) {
      state.scope = "session";
      $("scope-badge").textContent = state.code;
    } else {
      return showSetup(null);
    }

    if (state.topic) prepareOrder();
    startPolling();
    // for the session scope the topic only becomes known from the first poll
    window.setTimeout(function () {
      if (!state.order.length) { prepareOrder(); render(); }
    }, 600);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }
})();
