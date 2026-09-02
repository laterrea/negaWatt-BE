/* ==========================================================================
   negaWatt Belgium — workshop value collection (window.NW_API)
   --------------------------------------------------------------------------
   Thin wrapper over the JSON endpoints in website/api/*.php. The same contract
   is implemented by scripts/dev_api.py, so the UI can be developed offline and
   a workshop can be run from a laptop with no internet.

   Offline behaviour is a feature, not an afterthought: workshop wifi fails. Every
   answer is written to localStorage first and queued for the server; the queue is
   flushed whenever a later call succeeds or the browser comes back online. The
   participant never loses work and never sees a spinner.
   ========================================================================== */
(function () {
  "use strict";

  var LS_BASE = "nw.ws.apiBase";
  var LS_QUEUE = "nw.ws.queue";
  var LS_ANSWERS = "nw.ws.answers";
  var LS_IDENTITY = "nw.ws.identity";

  var listeners = [];
  var online = true;

  /* ------------------------------------------------------------- local store */
  function read(key, fallback) {
    try {
      var raw = window.localStorage.getItem(key);
      return raw ? JSON.parse(raw) : fallback;
    } catch (e) { return fallback; }
  }

  function write(key, value) {
    try { window.localStorage.setItem(key, JSON.stringify(value)); } catch (e) { /* full or private */ }
  }

  /* --------------------------------------------------------------- api base */
  function base() {
    var m = /[?&]api=([^&]+)/.exec(window.location.search);
    if (m) {
      var explicit = decodeURIComponent(m[1]).replace(/\/$/, "");
      write(LS_BASE, explicit);
      return explicit;
    }
    var stored = read(LS_BASE, null);
    if (stored) return stored;
    // Default: ../api relative to website/workshop/*.html
    return new URL("../api", window.location.href).href.replace(/\/$/, "");
  }

  function setBase(url) { write(LS_BASE, String(url).replace(/\/$/, "")); }

  /* ------------------------------------------------------------------ status */
  function onStatus(fn) { listeners.push(fn); fn(status()); }

  function status() {
    return { online: online, queued: read(LS_QUEUE, []).length };
  }

  function announce() {
    var s = status();
    listeners.forEach(function (fn) { try { fn(s); } catch (e) { /* ignore */ } });
  }

  function setOnline(value) {
    if (online !== value) { online = value; announce(); }
  }

  /* -------------------------------------------------------------- transport */
  function request(path, options) {
    options = options || {};
    var url = base() + path;
    var init = { method: options.method || "GET", headers: {} };
    if (options.body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    return fetch(url, init).then(function (res) {
      return res.text().then(function (text) {
        var data = null;
        try { data = text ? JSON.parse(text) : null; } catch (e) { /* not JSON */ }
        if (!res.ok) {
          var err = new Error((data && data.error) || ("HTTP " + res.status));
          err.status = res.status;
          err.data = data;
          throw err;
        }
        setOnline(true);
        return data;
      });
    }).catch(function (err) {
      // A 4xx is a real answer from a reachable server, not an outage.
      if (!err.status) setOnline(false);
      throw err;
    });
  }

  /* ------------------------------------------------------------- identity */
  function identity() { return read(LS_IDENTITY, null); }
  function forgetIdentity() { write(LS_IDENTITY, null); }

  /* --------------------------------------------------------------- sessions */
  function createSession(payload) {
    return request("/session.php", { method: "POST", body: payload });
  }

  function getSession(code) {
    return request("/session.php?code=" + encodeURIComponent(code));
  }

  function remember(data, slug) {
    var id = {
      code: data.code, slug: slug || data.slug || null,
      groupId: data.group_id, token: data.token,
      name: data.name, topic: data.topic
    };
    // Joining a *different* group or workshop must not inherit the answers of the
    // previous one: they would be restored on screen and, worse, pushed into the
    // new group by the autosave.
    var previous = identity();
    if (previous && (previous.code !== id.code || previous.groupId !== id.groupId)) {
      write(LS_ANSWERS, {});
      write(LS_QUEUE, []);
      announce();
    }
    write(LS_IDENTITY, id);
    return id;
  }

  function joinGroup(code, name) {
    return request("/group.php", { method: "POST", body: { code: code, name: name } })
      .then(function (data) { return remember(data, null); });
  }

  /* Join a workshop straight from its link: the server hands out the next free
     "Groupe 3", so the participant types nothing at all. `prefix` carries the
     language, which only the client knows. */
  function joinAuto(target, prefix) {
    var body = { auto_name: true, name_prefix: prefix };
    if (target.slug) body.slug = target.slug; else body.code = target.code;
    return request("/group.php", { method: "POST", body: body })
      .then(function (data) { return remember(data, target.slug || null); });
  }

  function renameGroup(name) {
    var id = identity();
    if (!id) return Promise.reject(new Error("no identity"));
    var body = { group_id: id.groupId, token: id.token, name: name };
    if (id.slug) body.slug = id.slug; else body.code = id.code;
    return request("/rename.php", { method: "POST", body: body })
      .then(function (data) {
        id.name = data.name;
        write(LS_IDENTITY, id);
        return id;
      });
  }

  function getSessionBySlug(slug) {
    return request("/session.php?slug=" + encodeURIComponent(slug));
  }

  /* ---------------------------------------------------------------- answers */
  function localAnswers() { return read(LS_ANSWERS, {}); }

  function saveAnswer(leverId, answer) {
    var id = identity();
    var all = localAnswers();
    all[leverId] = {
      value: answer.value,
      confidence: answer.confidence || null,
      condition: answer.condition || null,
      at: new Date().toISOString()
    };
    write(LS_ANSWERS, all);

    if (!id) return Promise.resolve({ local: true });

    var body = {
      code: id.code, group_id: id.groupId, token: id.token,
      lever_id: leverId, value: answer.value,
      confidence: answer.confidence || null,
      condition: answer.condition || null
    };
    return request("/answer.php", { method: "POST", body: body })
      .then(function (data) { flush(); return data; })
      .catch(function (err) {
        if (err.status) throw err;             // the server said no: surface it
        enqueue(body);
        return { queued: true };
      });
  }

  function enqueue(body) {
    var queue = read(LS_QUEUE, []);
    // one entry per lever: a later value supersedes an earlier one
    queue = queue.filter(function (item) { return item.lever_id !== body.lever_id; });
    queue.push(body);
    write(LS_QUEUE, queue);
    announce();
  }

  function flush() {
    var queue = read(LS_QUEUE, []);
    if (!queue.length) return Promise.resolve({ flushed: 0 });
    var pending = queue.slice();
    write(LS_QUEUE, []);
    announce();
    var failed = [];
    return pending.reduce(function (chain, body) {
      return chain.then(function () {
        return request("/answer.php", { method: "POST", body: body })
          .catch(function (err) { if (!err.status) failed.push(body); });
      });
    }, Promise.resolve()).then(function () {
      if (failed.length) {
        write(LS_QUEUE, read(LS_QUEUE, []).concat(failed));
        announce();
      }
      return { flushed: pending.length - failed.length };
    });
  }

  /* ---------------------------------------------------------------- results */
  function getResults(opts) {
    var params = [];
    if (opts.code) params.push("code=" + encodeURIComponent(opts.code));
    if (opts.topic) params.push("topic=" + encodeURIComponent(opts.topic));
    if (opts.adminToken) params.push("admin_token=" + encodeURIComponent(opts.adminToken));
    if (opts.adminKey) params.push("admin_key=" + encodeURIComponent(opts.adminKey));
    return request("/results.php?" + params.join("&"));
  }

  function setReveal(code, adminToken, step) {
    return request("/reveal.php", {
      method: "POST",
      body: { code: code, admin_token: adminToken, step: step }
    });
  }

  function selftest() { return request("/selftest.php"); }

  window.addEventListener("online", function () { setOnline(true); flush(); });
  window.addEventListener("offline", function () { setOnline(false); });

  window.NW_API = {
    base: base, setBase: setBase,
    status: status, onStatus: onStatus,
    identity: identity, forgetIdentity: forgetIdentity,
    createSession: createSession, getSession: getSession,
    getSessionBySlug: getSessionBySlug,
    joinGroup: joinGroup, joinAuto: joinAuto, renameGroup: renameGroup,
    saveAnswer: saveAnswer, localAnswers: localAnswers, flush: flush,
    getResults: getResults, setReveal: setReveal, selftest: selftest
  };
})();
