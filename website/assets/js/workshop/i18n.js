/* ==========================================================================
   negaWatt Belgium — workshop translations (window.NW_I18N)
   --------------------------------------------------------------------------
   Trilingual FR / NL / EN. Strings come from window.NW_WS_CONTENT.ui, which is
   generated from website/workshop/content/ui.yaml — nothing is hard-coded here.

   Language resolution order: ?lang= → previous choice → browser → default.
   ========================================================================== */
(function () {
  "use strict";

  var STORE_KEY = "nw.ws.lang";
  var state = { lang: null, content: null };

  function content() {
    if (!state.content) state.content = window.NW_WS_CONTENT || null;
    return state.content;
  }

  function available() {
    var c = content();
    return (c && c.languages) || ["fr", "nl", "en"];
  }

  function fromQuery() {
    var m = /[?&]lang=([a-z]{2})\b/i.exec(window.location.search);
    return m ? m[1].toLowerCase() : null;
  }

  function fromStore() {
    try { return window.localStorage.getItem(STORE_KEY); } catch (e) { return null; }
  }

  function fromBrowser() {
    var langs = navigator.languages || [navigator.language || ""];
    for (var i = 0; i < langs.length; i++) {
      var code = String(langs[i]).slice(0, 2).toLowerCase();
      if (available().indexOf(code) !== -1) return code;
    }
    return null;
  }

  function lang() {
    if (state.lang) return state.lang;
    var c = content();
    var candidates = [fromQuery(), fromStore(), fromBrowser(),
                      (c && c.defaultLanguage) || "fr"];
    for (var i = 0; i < candidates.length; i++) {
      if (candidates[i] && available().indexOf(candidates[i]) !== -1) {
        state.lang = candidates[i];
        break;
      }
    }
    if (!state.lang) state.lang = available()[0];
    document.documentElement.setAttribute("lang", state.lang);
    return state.lang;
  }

  function setLang(code) {
    if (available().indexOf(code) === -1) return lang();
    state.lang = code;
    try { window.localStorage.setItem(STORE_KEY, code); } catch (e) { /* private mode */ }
    document.documentElement.setAttribute("lang", code);
    return code;
  }

  /* Pick the current language out of a {fr, nl, en} block, falling back through
     the other languages rather than showing nothing. */
  function pick(node) {
    if (node === null || node === undefined) return "";
    if (typeof node === "string") return node;
    var order = [lang()].concat(available());
    for (var i = 0; i < order.length; i++) {
      if (node[order[i]]) return node[order[i]];
    }
    return "";
  }

  function interpolate(text, params) {
    if (!params) return text;
    return text.replace(/\{([A-Za-z][A-Za-z0-9_]*)\}/g, function (whole, key) {
      return Object.prototype.hasOwnProperty.call(params, key) ? String(params[key]) : whole;
    });
  }

  /* Interface string by key. Returns the key itself if missing, which makes a
     gap obvious on screen instead of rendering an empty element. */
  function t(key, params) {
    var c = content();
    var node = c && c.ui && c.ui[key];
    if (!node) return key;
    return interpolate(pick(node), params);
  }

  /* Format a number the way the current language writes it. */
  function num(value, decimals) {
    if (value === null || value === undefined || !isFinite(value)) return "—";
    var d = decimals;
    if (d === undefined) {
      d = Math.abs(value) >= 1000 ? 0 : Math.abs(value) >= 10 ? 1 : 2;
    }
    var locale = { fr: "fr-BE", nl: "nl-BE", en: "en-GB" }[lang()] || "en-GB";
    return value.toLocaleString(locale, {
      minimumFractionDigits: d, maximumFractionDigits: d
    });
  }

  function signed(value, decimals) {
    if (value === null || value === undefined || !isFinite(value)) return "—";
    return (value > 0 ? "+" : value < 0 ? "−" : "") + num(Math.abs(value), decimals);
  }

  /* Translate a unit the notebook exports in English, e.g. "persons/car". */
  function unit(raw) {
    if (!raw) return "";
    var key = "unit." + raw;
    var translated = t(key);
    return translated === key ? raw : translated;
  }

  /* Render *emphasis* into a node without ever touching innerHTML: the content
     comes from a YAML file, but building DOM nodes keeps that guarantee local. */
  function rich(node, text) {
    node.textContent = "";
    String(text === null || text === undefined ? "" : text)
      .split(/\*([^*]+)\*/)
      .forEach(function (part, i) {
        if (!part) return;
        if (i % 2 === 1) {
          var em = document.createElement("em");
          em.textContent = part;
          node.appendChild(em);
        } else {
          node.appendChild(document.createTextNode(part));
        }
      });
    return node;
  }

  window.NW_I18N = {
    lang: lang, setLang: setLang, available: available,
    t: t, pick: pick, num: num, signed: signed, interpolate: interpolate, rich: rich, unit: unit
  };
})();
