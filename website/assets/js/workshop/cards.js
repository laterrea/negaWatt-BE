/* ==========================================================================
   negaWatt Belgium — printable discussion cards (cards.html)
   Same questions and the same four facts as the screen, laid out A5 so a group
   can argue over paper. Deliberately does NOT show the negaWatt value.
   ========================================================================== */
(function () {
  "use strict";

  var T = window.NW_I18N;

  function param(name) {
    var m = new RegExp("[?&]" + name + "=([^&]*)").exec(window.location.search);
    return m ? decodeURIComponent(m[1]) : null;
  }

  function applyStaticText() {
    document.querySelectorAll("[data-ui]").forEach(function (node) {
      node.textContent = T.t(node.getAttribute("data-ui"));
    });
  }

  function buildLangSwitch(rebuild) {
    var box = document.getElementById("lang-switch");
    box.innerHTML = "";
    T.available().forEach(function (code) {
      var b = document.createElement("button");
      b.type = "button";
      b.textContent = code;
      b.setAttribute("aria-pressed", code === T.lang() ? "true" : "false");
      b.addEventListener("click", function () {
        T.setLang(code);
        buildLangSwitch(rebuild);
        applyStaticText();
        rebuild();
      });
      box.appendChild(b);
    });
  }

  function build() {
    var content = window.NW_WS_CONTENT;
    var topicId = param("topic") || Object.keys(content.topics)[0];
    var topic = content.topics[topicId];
    var levers = (window.NW_LEVERS[topic.sector] || {}).levers || {};
    var code = param("code");
    var box = document.getElementById("cards");
    box.innerHTML = "";

    (topic.order || []).forEach(function (id, index) {
      var lever = levers[id];
      var text = topic.levers[id];
      if (!lever || !text) return;

      var card = document.createElement("article");
      card.className = "ws-card";

      var head = document.createElement("div");
      head.className = "ws-card__head";
      var title = document.createElement("div");
      var q = document.createElement("p");
      q.className = "ws-card__q";
      q.textContent = T.pick(text.question);
      title.appendChild(q);
      if (text.subtitle) {
        var sub = document.createElement("p");
        sub.className = "ws-card__sub";
        sub.textContent = T.pick(text.subtitle);
        title.appendChild(sub);
      }
      var n = document.createElement("span");
      n.className = "ws-card__n";
      n.textContent = (index + 1) + "/" + topic.order.length;
      head.appendChild(title);
      head.appendChild(n);
      card.appendChild(head);

      var anchor = document.createElement("div");
      anchor.className = "ws-card__anchor";
      var a1 = document.createElement("div");
      a1.innerHTML = "<b>" + T.num(lever.refValue, lever.decimals) + " " +
                     T.unit(lever.unit) + "</b>" + T.t("play.refValue", { year: lever.refYear });
      anchor.appendChild(a1);
      var a2 = document.createElement("div");
      a2.className = "ws-card__scale";
      a2.innerHTML = "<b>" + T.num(lever.slider.min, Math.min(lever.decimals, 1)) + " – " +
                     T.num(lever.slider.max, Math.min(lever.decimals, 1)) + "</b>" +
                     T.t("play.yourAnswer");
      anchor.appendChild(a2);
      card.appendChild(anchor);

      var facts = document.createElement("div");
      facts.className = "ws-card__facts";
      // reveal-only facts state negaWatt's answer: they must not be printed
      (text.facts || []).filter(function (f) { return !f.reveal; }).forEach(function (fact) {
        var p = document.createElement("p");
        p.className = "ws-card__fact";
        var kind = document.createElement("b");
        kind.textContent = T.t("play.facts.kind." + (fact.kind || "structure"));
        p.appendChild(kind);
        var span = document.createElement("span");
        T.rich(span, T.pick(fact.text));
        p.appendChild(span);
        if (fact.source) {
          var cite = document.createElement("cite");
          cite.textContent = T.t("cards.source") + ": " + fact.source;
          p.appendChild(cite);
        }
        facts.appendChild(p);
      });
      card.appendChild(facts);

      var foot = document.createElement("div");
      foot.className = "ws-card__foot";
      var left = document.createElement("span");
      left.textContent = T.pick(topic.title);
      foot.appendChild(left);
      var right = document.createElement("span");
      if (code) {
        right.innerHTML = T.t("cards.scan") + ' <span class="ws-card__code">' +
                          code.replace(/[^A-Z0-9]/gi, "") + "</span>";
      } else {
        right.textContent = T.pick(text.short) || "";
      }
      foot.appendChild(right);
      card.appendChild(foot);

      box.appendChild(card);
    });
  }

  function init() {
    if (!window.NW_WS_CONTENT) {
      document.body.innerHTML = '<div class="ws-note ws-note--warn">' +
        "workshop_content.js is missing — run scripts/build_workshop_content.py</div>";
      return;
    }
    buildLangSwitch(build);
    applyStaticText();
    build();
    document.getElementById("btn-print").addEventListener("click", function () {
      window.print();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else { init(); }
})();
