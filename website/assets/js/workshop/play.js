/* ==========================================================================
   negaWatt Belgium — workshop participant page (play.html)
   --------------------------------------------------------------------------
   One lever per screen. Three rules drive the whole interaction:

     1. the slider handle IS the 2050 endpoint of the observed curve;
     2. it starts UNSET — a default value would anchor the group;
     3. every change autosaves, so the reveal screen is live even before the
        group presses Finish.

   Data comes from three generated globals: NW_LEVERS (numbers, from the
   notebook), NW_HISTORY (observed series, from the notebook) and NW_WS_CONTENT
   (wording, from the topic YAML). Nothing here is hard-coded.
   ========================================================================== */
(function () {
  "use strict";

  var T = window.NW_I18N;
  var API = window.NW_API;

  var state = {
    topic: null,
    sector: null,
    order: [],
    index: 0,
    answers: {},        // leverId -> {value, confidence, condition}
    dirty: {},          // leverId -> debounce timer
    finished: false
  };

  /* ------------------------------------------------------------------ utils */
  function $(id) { return document.getElementById(id); }

  function param(name) {
    var m = new RegExp("[?&]" + name + "=([^&]*)").exec(window.location.search);
    return m ? decodeURIComponent(m[1].replace(/\+/g, " ")) : null;
  }

  function levers() {
    var data = window.NW_LEVERS && window.NW_LEVERS[state.sector];
    return (data && data.levers) || {};
  }

  function historySeries() {
    var data = window.NW_HISTORY && window.NW_HISTORY[state.sector];
    return (data && data.series) || {};
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

  /* Values a tangible sentence may ask for. {value} is the slider itself;
     the two derived ones let a card say "x per year, i.e. y per day" without
     making the author do the arithmetic in the YAML. */
  function tangibleParams(lever, value) {
    return {
      value: T.num(value, lever.decimals),
      valuePerDay: T.num(value / 365, value / 365 >= 10 ? 0 : 1),
      valuePerYear: T.num(value * 365, 0)
    };
  }

  /* ------------------------------------------------------------------ chrome */
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
      b.title = T.t("lang." + code);
      b.setAttribute("aria-pressed", code === T.lang() ? "true" : "false");
      b.addEventListener("click", function () {
        T.setLang(code);
        buildLangSwitch();
        applyStaticText();
        render();
      });
      box.appendChild(b);
    });
  }

  function showStatus() {
    var badge = $("status-badge");
    var s = API.status();
    if (!s.online) {
      badge.className = "ws-badge ws-badge--warn";
      badge.textContent = T.t("play.offline");
      badge.classList.remove("ws-hidden");
    } else if (s.queued) {
      badge.className = "ws-badge ws-badge--quiet";
      badge.textContent = s.queued + " ⇡";
      badge.classList.remove("ws-hidden");
    } else {
      badge.classList.add("ws-hidden");
    }
  }

  /* --------------------------------------------------------------- progress */
  function buildProgress() {
    var box = $("progress-dots");
    box.innerHTML = "";
    state.order.forEach(function (id, i) {
      var dot = document.createElement("button");
      dot.type = "button";
      dot.className = "ws-progress__dot";
      dot.setAttribute("aria-label", String(i + 1));
      dot.dataset.state = i === state.index ? "current"
                        : (state.answers[id] ? "answered" : "todo");
      dot.addEventListener("click", function () { go(i); });
      box.appendChild(dot);
    });
    $("progress-label").textContent = T.t("play.progress", {
      n: state.index + 1, total: state.order.length
    });
  }

  /* ------------------------------------------------------------------ chart */
  function drawChart(lever, content, value) {
    var series = null;
    if (content.history && content.history.x) {
      series = content.history;                       // hand-curated in the YAML
    } else if (lever.history && historySeries()[lever.history]) {
      series = historySeries()[lever.history];
    }

    window.NW_SPARK.history($("chart"), {
      x: series ? series.x : [],
      y: series ? series.y : [],
      refYear: lever.refYear, refValue: lever.refValue,
      targetYear: lever.targetYear,
      value: value,
      decimals: lever.decimals,
      domainMin: lever.slider.min, domainMax: lever.slider.max,
      srLabel: T.pick(content.question)
    });

    var legend = $("chart-legend");
    legend.innerHTML = "";
    function add(cls, text) {
      var s = document.createElement("span");
      s.className = cls;
      s.innerHTML = '<i></i>' + text;
      legend.appendChild(s);
    }
    if (series) add("obs", (series.x[0] || "") + "–" + (series.x[series.x.length - 1] || ""));
    else add("obs", String(lever.refYear));
    if (value !== null && value !== undefined) add("you", T.t("play.yourAnswer"));

    var note = "";
    if (series) {
      note = T.t("common.source") + ": " + (T.pick(series.label) || "") +
             (series.source ? " — " + series.source : "");
    } else if (content.historyNote) {
      note = T.pick(content.historyNote);
    }
    $("chart-note").textContent = note;
  }

  /* --------------------------------------------------------------- leverage */
  function drawLeverage(lever, value) {
    var box = $("leverage");
    if (!lever.impact || lever.impact.kind === "negligible") {
      box.classList.add("ws-hidden");
      return;
    }
    box.classList.remove("ws-hidden");

    if (value === null || value === undefined) {
      box.classList.add("is-hidden");
      $("leverage-value").textContent = T.t("play.leverage.hidden");
      $("leverage-delta").textContent = "";
      $("leverage-delta").removeAttribute("data-dir");
      return;
    }
    box.classList.remove("is-hidden");

    // What THIS answer contributes, against simply keeping today's level of the
    // same indicator. Never the sector total (which reads as the combined effect
    // of all eight levers) and never a comparison with negaWatt (which a group
    // could solve by sliding until the difference vanished).
    var delta = window.NW_IMPACT.contribution(lever.impact, value, lever.refValue);
    var main = $("leverage-value");
    var sub = $("leverage-delta");

    if (delta === null) {
      main.textContent = "";
      sub.textContent = "";
      sub.removeAttribute("data-dir");
      return;
    }

    if (Math.abs(delta) < 0.05) {
      main.textContent = T.t("play.leverage.neutral");
      main.dataset.dir = "flat";
    } else {
      main.textContent = T.t(delta < 0 ? "play.leverage.saves" : "play.leverage.costs",
                             { delta: T.num(Math.abs(delta), 1) });
      main.dataset.dir = delta < 0 ? "down" : "up";
    }
    // Levers already expressed as "% of 2019" have a reference of 100 % of 2019,
    // and naming it makes the sentence circular.
    sub.textContent = lever.unit === "% of 2019"
      ? T.t("play.leverage.versusPlain", { year: lever.refYear })
      : T.t("play.leverage.versus", {
          year: lever.refYear,
          ref: T.num(lever.refValue, lever.decimals) + " " + T.unit(lever.unit)
        });
    sub.dataset.dir = "flat";
  }

  /* ------------------------------------------------------------------ facts */
  function drawFacts(content) {
    var box = $("facts");
    box.innerHTML = "";
    // facts flagged `reveal: true` describe negaWatt's own choice and belong to
    // the reveal screen, never to the page where the group is still deciding
    (content.facts || []).filter(function (f) { return !f.reveal; }).forEach(function (fact) {
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
        src.textContent = T.t("common.source") + ": ";
        if (fact.url) {
          var a = document.createElement("a");
          a.href = fact.url;
          a.target = "_blank";
          a.rel = "noopener noreferrer";
          a.textContent = fact.source;
          src.appendChild(a);
        } else {
          src.appendChild(document.createTextNode(fact.source));
        }
        card.appendChild(src);
      }
      box.appendChild(card);
    });
  }

  /* ------------------------------------------------------------- confidence */
  var CONFIDENCE = [
    { value: 1, key: "play.confidence.hunch" },
    { value: 2, key: "play.confidence.fairly" },
    { value: 3, key: "play.confidence.confident" }
  ];

  function drawConfidence(id) {
    var box = $("confidence");
    box.innerHTML = "";
    var answer = state.answers[id] || {};
    CONFIDENCE.forEach(function (opt) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = T.t(opt.key);
      b.setAttribute("aria-pressed", answer.confidence === opt.value ? "true" : "false");
      b.addEventListener("click", function () {
        update(id, { confidence: answer.confidence === opt.value ? null : opt.value });
      });
      box.appendChild(b);
    });
  }

  /* ------------------------------------------------------------------ render */
  function render() {
    if (state.finished) { renderDone(); return; }
    $("screen-play").classList.remove("ws-hidden");
    $("screen-done").classList.add("ws-hidden");

    var id = current();
    var lever = levers()[id];
    var content = leverContent(id);
    var answer = state.answers[id] || {};
    var value = answer.value === undefined ? null : answer.value;

    var t = topicContent();
    $("topic-title").textContent = t ? T.pick(t.title) : "";
    $("question").textContent = T.pick(content.question) || lever.name;
    $("subtitle").textContent = T.pick(content.subtitle);

    var slider = $("slider");
    slider.min = lever.slider.min;
    slider.max = lever.slider.max;
    slider.step = lever.slider.step;
    slider.value = value === null
      ? (lever.slider.min + lever.slider.max) / 2      // parked, but visibly unset
      : value;
    slider.setAttribute("aria-valuetext",
      value === null ? T.t("play.unanswered")
                     : T.num(value, lever.decimals) + " " + T.unit(lever.unit));
    $("slider-wrap").classList.toggle("is-answered", value !== null);
    // one decimal at most: "1,0 … 2,6" reads better than "1 … 3", which would
    // round the end of the range away
    var scaleDp = Math.min(lever.decimals, 1);
    $("scale-min").textContent = T.num(lever.slider.min, scaleDp);
    $("scale-max").textContent = T.num(lever.slider.max, scaleDp) + " " + T.unit(lever.unit);

    var readout = $("readout");
    readout.classList.toggle("is-empty", value === null);
    $("readout-num").textContent = value === null
      ? T.t("play.unanswered") : T.num(value, lever.decimals);
    $("readout-unit").textContent = T.unit(lever.unit);
    $("readout-tangible").textContent = value === null ? ""
      : T.interpolate(T.pick(content.tangible), tangibleParams(lever, value));

    drawChart(lever, content, value);
    drawLeverage(lever, value);
    drawFacts(content);
    drawConfidence(id);
    $("condition").value = answer.condition || "";
    $("condition").placeholder = T.t("play.condition.placeholder");

    renderIdentity();
    buildProgress();
    $("btn-back").disabled = state.index === 0;
    $("btn-next").textContent = state.index === state.order.length - 1
      ? T.t("play.finish") : T.t("play.next");
    $("btn-next").className = state.index === state.order.length - 1
      ? "ws-btn ws-btn--amber" : "ws-btn";
    showStatus();
  }

  /* Who this device is answering as. Server-assigned names are fine remotely; in
     a room the group usually wants to be "Table 3", so offer a rename in place. */
  function renderIdentity() {
    var box = $("identity");
    if (!box) return;
    var id = API.identity();
    if (!id || !id.name) { box.classList.add("ws-hidden"); return; }
    box.classList.remove("ws-hidden");
    box.innerHTML = "";
    var label = document.createElement("span");
    label.textContent = id.name;
    box.appendChild(label);
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ws-linkbtn";
    btn.textContent = T.t("group.rename");
    btn.addEventListener("click", function () {
      var next = window.prompt(T.t("group.renamePrompt"), id.name);
      if (!next || next.trim() === "" || next === id.name) return;
      API.renameGroup(next.trim())
        .then(renderIdentity)
        .catch(function () { window.alert(T.t("group.renameFailed")); });
    });
    box.appendChild(btn);
  }

  /* Light-touch update while dragging: only the parts that depend on the value. */
  function renderValueOnly(value) {
    var id = current();
    var lever = levers()[id];
    var content = leverContent(id);
    $("slider-wrap").classList.add("is-answered");
    $("readout").classList.remove("is-empty");
    $("readout-num").textContent = T.num(value, lever.decimals);
    $("readout-tangible").textContent =
      T.interpolate(T.pick(content.tangible), tangibleParams(lever, value));
    drawChart(lever, content, value);
    drawLeverage(lever, value);
  }

  function renderDone() {
    $("screen-play").classList.add("ws-hidden");
    $("screen-done").classList.remove("ws-hidden");
    var table = $("summary-table");
    table.innerHTML = "";
    var head = table.insertRow();
    [T.t("reveal.summary"), T.t("play.yourAnswer")].forEach(function (label) {
      var th = document.createElement("th");
      th.textContent = label;
      head.appendChild(th);
    });
    state.order.forEach(function (id) {
      var lever = levers()[id];
      var answer = state.answers[id] || {};
      var row = table.insertRow();
      row.insertCell().textContent = T.pick(leverContent(id).short) ||
                                     T.pick(leverContent(id).question) || lever.name;
      var cell = row.insertCell();
      cell.className = "num";
      cell.textContent = answer.value === undefined ? "—"
        : T.num(answer.value, lever.decimals) + " " + T.unit(lever.unit);
    });
    $("btn-back").disabled = false;
    $("btn-next").classList.add("ws-hidden");
  }

  /* ----------------------------------------------------------------- updates */
  function update(id, patch) {
    var answer = state.answers[id] || {};
    Object.keys(patch).forEach(function (k) { answer[k] = patch[k]; });
    state.answers[id] = answer;
    if (answer.value === undefined || answer.value === null) return render();
    scheduleSave(id);
    render();
  }

  function scheduleSave(id) {
    if (state.dirty[id]) window.clearTimeout(state.dirty[id]);
    state.dirty[id] = window.setTimeout(function () {
      delete state.dirty[id];
      var answer = state.answers[id];
      if (!answer || answer.value === undefined || answer.value === null) return;
      API.saveAnswer(id, answer).then(showStatus, showStatus);
    }, 450);
  }

  function go(index) {
    if (index < 0 || index >= state.order.length) return;
    state.index = index;
    state.finished = false;
    render();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  /* -------------------------------------------------------------------- init */
  function wire() {
    var slider = $("slider");
    slider.addEventListener("input", function () {
      var value = parseFloat(slider.value);
      var answer = state.answers[current()] || {};
      answer.value = value;
      state.answers[current()] = answer;
      renderValueOnly(value);            // cheap: no full re-render while dragging
      scheduleSave(current());
    });
    slider.addEventListener("change", function () { render(); });

    $("condition").addEventListener("input", function () {
      var answer = state.answers[current()] || {};
      answer.condition = this.value.slice(0, 280);
      state.answers[current()] = answer;
      if (answer.value !== undefined && answer.value !== null) scheduleSave(current());
    });

    $("btn-back").addEventListener("click", function () {
      if (state.finished) { state.finished = false; render(); return; }
      go(state.index - 1);
    });
    $("btn-next").addEventListener("click", function () {
      if (state.index === state.order.length - 1) {
        state.finished = true;
        API.flush();
        renderDone();
        window.scrollTo({ top: 0, behavior: "smooth" });
      } else {
        go(state.index + 1);
      }
    });

    API.onStatus(showStatus);
  }

  function fail(message) {
    document.querySelector(".ws-shell").innerHTML =
      '<main class="ws-main"><div class="ws-note ws-note--warn">' +
      message.replace(/</g, "&lt;") + "</div></main>";
  }

  /* A link of the form play.html?w=<slug> joins the workshop by itself: the
     server hands out a group name and the participant lands on question 1
     without typing anything. Re-opening the same link keeps the same group. */
  function ensureJoined() {
    var slug = param("w");
    if (!slug) return Promise.resolve(null);
    var identity = API.identity();
    if (identity && identity.slug === slug) return Promise.resolve(identity);
    return API.joinAuto({ slug: slug }, T.t("group.autoPrefix"))
      .catch(function (err) {
        // Offline, or the facilitator has not created it yet: fall through and
        // let the page work locally rather than blocking on the network.
        if (err && err.status === 404) throw err;
        return null;
      });
  }

  function init() {
    var content = window.NW_WS_CONTENT;
    if (!content) return fail("workshop_content.js is missing — run scripts/build_workshop_content.py");

    var identity = API.identity();
    state.topic = param("topic") || (identity && identity.topic) || Object.keys(content.topics)[0];
    var topic = content.topics[state.topic];
    if (!topic) return fail("Unknown workshop topic: " + state.topic);
    state.sector = topic.sector;
    state.order = (topic.order || []).filter(function (id) { return !!levers()[id]; });
    if (!state.order.length) return fail("No levers to play for topic " + state.topic);

    // restore anything this device already answered (reload, or offline session)
    var stored = API.localAnswers();
    Object.keys(stored).forEach(function (id) {
      if (state.order.indexOf(id) !== -1) state.answers[id] = stored[id];
    });
    var first = state.order.findIndex(function (id) { return !state.answers[id]; });
    state.index = first === -1 ? 0 : first;

    buildLangSwitch();
    applyStaticText();
    wire();
    render();
    API.flush();
  }

  function boot() {
    ensureJoined()
      .then(function (identity) {
        if (identity && identity.topic) {
          // the link decides the topic; a stale ?topic= must not override it
          var url = new URL(window.location.href);
          if (!url.searchParams.get("topic")) state.topic = identity.topic;
        }
        init();
      })
      .catch(function (err) {
        if (err && err.status === 404) {
          return fail(T.t("join.badLink"));
        }
        init();
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else { boot(); }
})();
