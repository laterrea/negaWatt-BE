/* ==========================================================================
   negaWatt Belgium — workshop entry page (index.html)
   Join a facilitator's session by code, or create a private solo session.
   ========================================================================== */
(function () {
  "use strict";

  var T = window.NW_I18N;
  var API = window.NW_API;

  function $(id) { return document.getElementById(id); }

  function applyStaticText() {
    document.querySelectorAll("[data-ui]").forEach(function (node) {
      node.textContent = T.t(node.getAttribute("data-ui"));
    });
    $("name").placeholder = T.t("join.groupPlaceholder");
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
        buildTopics();
      });
      box.appendChild(b);
    });
  }

  function topics() {
    var c = window.NW_WS_CONTENT;
    return (c && c.topics) || {};
  }

  function buildTopics() {
    var select = $("solo-topic");
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
    var box = $("error");
    box.textContent = message;
    box.classList.remove("ws-hidden");
  }

  function clearError() { $("error").classList.add("ws-hidden"); }

  function goPlay(topic) {
    window.location.href = "play.html?topic=" + encodeURIComponent(topic) +
                           "&lang=" + encodeURIComponent(T.lang());
  }

  function join() {
    clearError();
    var code = $("code").value.trim().toUpperCase();
    var name = $("name").value.trim();
    if (!code) return fail(T.t("join.badCode"));
    if (!name) return fail(T.t("join.needName"));

    $("btn-join").disabled = true;
    API.getSession(code)
      .then(function (session) {
        if (session.closed) throw new Error("closed");
        return API.joinGroup(code, name).then(function () { goPlay(session.topic); });
      })
      .catch(function (err) {
        $("btn-join").disabled = false;
        if (err && err.status === 404) return fail(T.t("join.badCode"));
        if (err && err.message === "closed") return fail(T.t("join.badCode"));
        fail(T.t("common.error") + " (" + ((err && err.message) || "network") + ")");
      });
  }

  function solo() {
    clearError();
    var topic = $("solo-topic").value;
    if (!topic) return;
    $("btn-solo").disabled = true;
    API.createSession({ topic: topic, label: "solo", mode: "solo" })
      .then(function (session) {
        return API.joinGroup(session.code, T.t("join.solo"))
          .then(function () { goPlay(topic); });
      })
      .catch(function (err) {
        $("btn-solo").disabled = false;
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

    var prefill = /[?&]code=([A-Za-z0-9]+)/.exec(window.location.search);
    if (prefill) $("code").value = prefill[1].toUpperCase();

    $("code").addEventListener("input", function () {
      this.value = this.value.toUpperCase().replace(/[^A-Z0-9]/g, "");
      clearError();
    });
    $("name").addEventListener("input", clearError);
    $("name").addEventListener("keydown", function (e) { if (e.key === "Enter") join(); });
    $("btn-join").addEventListener("click", join);
    $("btn-solo").addEventListener("click", solo);

    $("api-note").textContent = "API: " + API.base();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }
})();
