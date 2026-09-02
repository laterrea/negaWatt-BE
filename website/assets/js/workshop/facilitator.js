/* ==========================================================================
   negaWatt Belgium — facilitator console (facilitator.html)
   Create a session, watch the groups arrive and answer, then open the projector.
   The admin token is kept in this browser only; it is what unlocks the results.
   ========================================================================== */
(function () {
  "use strict";

  var T = window.NW_I18N;
  var API = window.NW_API;
  var LS = "nw.ws.facilitator";
  var POLL_MS = 4000;

  var state = { code: null, slug: null, adminToken: null, topic: null, timer: null };

  function $(id) { return document.getElementById(id); }

  function remember() {
    try {
      window.localStorage.setItem(LS, JSON.stringify({
        code: state.code, slug: state.slug,
        adminToken: state.adminToken, topic: state.topic
      }));
    } catch (e) { /* private mode */ }
  }

  /* "Atelier Namur 2026" -> "atelier-namur-2026", which is what goes in the link. */
  function slugify(text) {
    return String(text || "")
      .normalize("NFD").replace(/[\u0300-\u036f]/g, "")   // strip accents
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "")
      .slice(0, 64);
  }

  function recall() {
    try {
      var raw = window.localStorage.getItem(LS);
      return raw ? JSON.parse(raw) : null;
    } catch (e) { return null; }
  }

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
        T.setLang(code); buildLangSwitch(); applyStaticText(); buildTopics(); refreshLinks();
      });
      box.appendChild(b);
    });
  }

  function topics() {
    var c = window.NW_WS_CONTENT;
    return (c && c.topics) || {};
  }

  function buildTopics() {
    var select = $("topic");
    var keep = select.value;
    select.innerHTML = "";
    Object.keys(topics()).forEach(function (id) {
      var option = document.createElement("option");
      option.value = id;
      option.textContent = T.pick(topics()[id].title) || id;
      select.appendChild(option);
    });
    if (keep) select.value = keep;
  }

  function fail(message) {
    $("error").textContent = message;
    $("error").classList.remove("ws-hidden");
  }

  function joinUrl() {
    // The participant link. With a slug nobody has to type anything at all;
    // without one we fall back to prefilling the code.
    var query = state.slug
      ? "?w=" + encodeURIComponent(state.slug)
      : "?code=" + encodeURIComponent(state.code);
    return new URL("play.html" + query + "&lang=" + encodeURIComponent(T.lang()),
                   window.location.href).href;
  }

  function refreshLinks() {
    if (!state.code) return;
    var lang = "&lang=" + encodeURIComponent(T.lang());
    $("link-reveal").href = "reveal.html?code=" + encodeURIComponent(state.code) +
      "&admin=" + encodeURIComponent(state.adminToken) + lang;
    $("link-cards").href = "cards.html?topic=" + encodeURIComponent(state.topic) +
      "&code=" + encodeURIComponent(state.code) + lang;
    $("join-url").textContent = joinUrl();
  }

  function showLive() {
    $("create-panel").classList.add("ws-hidden");
    $("live-panel").classList.remove("ws-hidden");
    $("code").textContent = state.code;
    $("admin-token").textContent = state.adminToken;
    refreshLinks();
    startPolling();
  }

  function renderGroups(session) {
    var table = $("groups-table");
    table.innerHTML = "";
    var groups = session.groups || [];
    $("waiting").classList.toggle("ws-hidden", groups.length > 0);
    if (!groups.length) return;

    var total = (topics()[state.topic] || {}).order || [];
    var head = table.insertRow();
    [T.t("join.groupLabel"), T.t("facil.progress")].forEach(function (label) {
      var th = document.createElement("th");
      th.textContent = label;
      head.appendChild(th);
    });
    groups.forEach(function (group) {
      var row = table.insertRow();
      row.insertCell().textContent = group.name;
      var cell = row.insertCell();
      cell.className = "num";
      cell.textContent = T.t("facil.answers", { done: group.answered, total: total.length });
    });
  }

  function poll() {
    if (!state.code) return;
    API.getSession(state.code).then(renderGroups).catch(function () { /* keep trying */ });
  }

  function startPolling() {
    if (state.timer) window.clearInterval(state.timer);
    state.timer = window.setInterval(poll, POLL_MS);
    poll();
  }

  function create() {
    $("error").classList.add("ws-hidden");
    var topic = $("topic").value;
    var label = $("label").value.trim();
    if (!topic) return;
    var slug = slugify(label);
    $("btn-create").disabled = true;
    API.createSession({ topic: topic, label: label, slug: slug || undefined })
      .then(function (session) {
        state.code = session.code;
        state.slug = session.slug;
        state.adminToken = session.admin_token;
        state.topic = session.topic;
        remember();
        showLive();
      })
      .catch(function (err) {
        $("btn-create").disabled = false;
        if (err && err.status === 409) return fail(T.t("facil.slugTaken"));
        fail(T.t("common.error") + " (" + ((err && err.message) || "network") + ")");
      });
  }

  function init() {
    if (!window.NW_WS_CONTENT) {
      document.querySelector(".ws-main").innerHTML =
        '<div class="ws-note ws-note--warn">workshop_content.js is missing — ' +
        "run scripts/build_workshop_content.py</div>";
      return;
    }
    buildLangSwitch();
    applyStaticText();
    buildTopics();
    $("api-note").textContent = "API: " + API.base();
    $("btn-create").addEventListener("click", create);
    $("label").addEventListener("input", function () {
      var slug = slugify(this.value);
      $("slug-preview").textContent = slug ? "…/workshop/play.html?w=" + slug : "";
    });
    $("btn-copy").addEventListener("click", function () {
      var url = joinUrl();
      var done = function () {
        this.textContent = T.t("facil.copied");
        window.setTimeout(function () {
          $("btn-copy").textContent = T.t("facil.copyLink");
        }, 1500);
      }.bind(this);
      if (navigator.clipboard) {
        navigator.clipboard.writeText(url).then(done, function () { window.prompt("", url); });
      } else {
        window.prompt("", url);
      }
    });

    var stored = recall();
    if (stored && stored.code && stored.adminToken) {
      state.code = stored.code;
      state.slug = stored.slug || null;
      state.adminToken = stored.adminToken;
      state.topic = stored.topic;
      // only resume a session the server still knows about
      API.getSession(state.code).then(function (session) {
        if (session.closed) return;
        state.topic = session.topic;
        state.slug = session.slug || state.slug;
        showLive();
      }).catch(function () { /* stale: fall back to the create form */ });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }
})();
