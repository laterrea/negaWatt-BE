/* ==========================================================================
   negaWatt Belgium — workshop entry page (index.html)
   --------------------------------------------------------------------------
   Pick a topic, optionally name the group, start. Nothing else: no code to type,
   no session to create, no solo mode. The group name is a nicety for a room with
   tables — left empty, the server names the group "Groupe 3".
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
    $("name").placeholder = T.t("start.groupPlaceholder");
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
        refreshLinks();
      });
      box.appendChild(b);
    });
  }

  function topics() {
    var c = window.NW_WS_CONTENT;
    return (c && c.topics) || {};
  }

  function topic() { return $("topic").value || Object.keys(topics())[0]; }

  function buildTopics() {
    var select = $("topic");
    var keep = select.value;
    var ids = Object.keys(topics());
    select.innerHTML = "";
    ids.forEach(function (id) {
      var option = document.createElement("option");
      option.value = id;
      option.textContent = T.pick(topics()[id].title) || id;
      select.appendChild(option);
    });
    if (keep) select.value = keep;
    $("topic-field").classList.toggle("ws-hidden", ids.length < 2);
  }

  /* The two facilitator destinations follow the topic and the language. */
  function refreshLinks() {
    var query = "?topic=" + encodeURIComponent(topic()) +
                "&lang=" + encodeURIComponent(T.lang());
    $("link-reveal").href = "reveal.html" + query;
    $("link-cards").href = "cards.html" + query;
  }

  function fail(message) {
    var box = $("error");
    box.textContent = message;
    box.classList.remove("ws-hidden");
  }

  function clearError() { $("error").classList.add("ws-hidden"); }

  function begin() {
    clearError();
    var chosen = topic();
    if (!chosen) return;
    $("btn-start").disabled = true;
    API.start(chosen, $("name").value.trim(), T.t("group.autoPrefix"))
      .then(function () {
        window.location.href = "play.html?topic=" + encodeURIComponent(chosen) +
                               "&lang=" + encodeURIComponent(T.lang());
      })
      .catch(function (err) {
        $("btn-start").disabled = false;
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

    var wanted = /[?&]topic=([A-Za-z0-9_-]+)/.exec(window.location.search);
    if (wanted && topics()[wanted[1]]) $("topic").value = wanted[1];
    refreshLinks();

    $("topic").addEventListener("change", refreshLinks);
    $("name").addEventListener("input", clearError);
    $("name").addEventListener("keydown", function (e) { if (e.key === "Enter") begin(); });
    $("btn-start").addEventListener("click", begin);

    $("api-note").textContent = "API: " + API.base();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }
})();
