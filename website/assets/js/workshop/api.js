/* ==========================================================================
   negaWatt Belgium — workshop value collection (window.NW_API)
   --------------------------------------------------------------------------
   Thin wrapper over the JSON endpoints in website/api/*.php. The same contract
   is implemented by scripts/dev_api.py, so the UI can be developed offline and
   a workshop can be run from a laptop with no internet.

   There is no session and no code: a group belongs to a topic, and the reveal
   screen selects groups by date.

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

  // A group is a workshop sitting, not an account. Re-opening the page an hour
  // later must return to the same group; re-opening it next week must not, or the
  // new answers would be filed under a group the reveal screen no longer shows.
  var IDENTITY_TTL_MS = 12 * 3600 * 1000;

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

  function fresh(id) {
    if (!id || !id.groupId) return false;
    var at = Date.parse(id.at || "");
    return isFinite(at) && (Date.now() - at) < IDENTITY_TTL_MS;
  }

  function remember(data, topic) {
    var id = {
      groupId: data.group_id, token: data.token,
      name: data.name, topic: data.topic || topic,
      at: new Date().toISOString()
    };
    // Answering as a *different* group must not inherit the answers of the
    // previous one: they would be restored on screen and, worse, pushed into the
    // new group by the autosave.
    var previous = identity();
    if (previous && previous.groupId !== id.groupId) {
      write(LS_ANSWERS, {});
      write(LS_QUEUE, []);
      announce();
    }
    write(LS_IDENTITY, id);
    return id;
  }

  /* ----------------------------------------------------------------- groups */

  /* Start answering. `name` is optional: without one the server names the group
     after its rank in the day, and `prefix` carries the language, which only the
     client knows. */
  function start(topic, name, prefix) {
    var body = { topic: topic, name_prefix: prefix || undefined };
    if (name) body.name = name;
    return request("/group.php", { method: "POST", body: body })
      .then(function (data) {
        var id = remember(data, topic);
        // Anything answered before the group existed (offline first load) belongs
        // to it now.
        if (Object.keys(localAnswers()).length) pushLocal();
        return id;
      });
  }

  /* The identity this device should answer with, creating a group if needed. */
  function ensure(topic, prefix) {
    var id = identity();
    if (id && id.topic === topic && fresh(id)) return Promise.resolve(id);
    return start(topic, null, prefix);
  }

  function renameGroup(name) {
    var id = identity();
    if (!id) return Promise.reject(new Error("no identity"));
    return request("/rename.php", {
      method: "POST",
      body: { group_id: id.groupId, token: id.token, name: name }
    }).then(function (data) {
      id.name = data.name;
      write(LS_IDENTITY, id);
      return id;
    });
  }

  /* ---------------------------------------------------------------- answers */
  function localAnswers() { return read(LS_ANSWERS, {}); }

  function payload(id, leverId, answer) {
    return {
      group_id: id.groupId, token: id.token, lever_id: leverId,
      value: answer.value,
      confidence: answer.confidence || null,
      condition: answer.condition || null
    };
  }

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

    var body = payload(id, leverId, answer);
    return request("/answer.php", { method: "POST", body: body })
      .then(function (data) { flush(); return data; })
      .catch(function (err) {
        if (err.status) throw err;             // the server said no: surface it
        enqueue(body);
        return { queued: true };
      });
  }

  /* Send everything this device holds locally. Used once, right after a group is
     created, so answers given while offline are not stranded. */
  function pushLocal() {
    var id = identity();
    if (!id) return Promise.resolve({ pushed: 0 });
    var all = localAnswers();
    Object.keys(all).forEach(function (leverId) {
      if (all[leverId] && all[leverId].value !== undefined && all[leverId].value !== null) {
        enqueue(payload(id, leverId, all[leverId]));
      }
    });
    return flush();
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

  /* Every group that started on the topic inside the window. `from`/`to` are UTC
     'YYYY-MM-DD HH:MM:SS'; leave either out for an open end. */
  function getResults(opts) {
    var params = ["topic=" + encodeURIComponent(opts.topic)];
    if (opts.from) params.push("from=" + encodeURIComponent(opts.from));
    if (opts.to) params.push("to=" + encodeURIComponent(opts.to));
    return request("/results.php?" + params.join("&"));
  }

  function selftest() { return request("/selftest.php"); }

  window.addEventListener("online", function () { setOnline(true); flush(); });
  window.addEventListener("offline", function () { setOnline(false); });

  window.NW_API = {
    base: base, setBase: setBase,
    status: status, onStatus: onStatus,
    identity: identity, forgetIdentity: forgetIdentity, fresh: fresh,
    start: start, ensure: ensure, renameGroup: renameGroup,
    saveAnswer: saveAnswer, localAnswers: localAnswers,
    pushLocal: pushLocal, flush: flush,
    getResults: getResults, selftest: selftest
  };
})();
