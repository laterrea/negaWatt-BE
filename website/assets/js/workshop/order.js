/* ==========================================================================
   negaWatt Belgium — the order a group plays the questions in (window.NW_ORDER)
   --------------------------------------------------------------------------
   The last questions of a series get fewer, and less careful, answers. So each
   group plays a `random` topic in its own shuffled order (D63), and every
   question keeps the letter of its place in the YAML — A, B, C… — which is what
   the printed card and the reveal show.

   The shuffle is seeded by the group's id. The same group therefore sees the
   same order on every device and after every reload, and the order it was shown
   can be recomputed later from the database alone:

       NW_ORDER.sequence(NW_WS_CONTENT.topics["inland-mobility"], 157)

   A topic with `questionOrder: fixed` is played in the YAML order, as before.
   Questions `linked` in the YAML (`topic.units`) move together, in their order.
   Pure functions, no DOM: play.js decides which seed to use.
   ========================================================================== */
(function () {
  "use strict";

  /* A small seeded PRNG (mulberry32): uniform floats in [0, 1). */
  function mulberry32(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* The question ids of `topic` in the order a group seeded by `seed` plays
     them. Fisher-Yates over the units, then each unit unrolled in place. */
  function sequence(topic, seed) {
    var order = (topic && topic.order) || [];
    if (!topic || topic.questionOrder === "fixed") return order.slice();
    var units = (topic.units && topic.units.length ? topic.units : order.map(function (id) {
      return [id];
    })).map(function (u) { return u.slice(); });
    var rand = mulberry32(seed);
    for (var i = units.length - 1; i > 0; i--) {
      var j = Math.floor(rand() * (i + 1));
      var tmp = units[i]; units[i] = units[j]; units[j] = tmp;
    }
    return units.reduce(function (all, u) { return all.concat(u); }, []);
  }

  /* A seed for a device that has no group yet (offline first load). */
  function randomSeed() {
    return Math.floor(Math.random() * 4294967296);
  }

  window.NW_ORDER = { mulberry32: mulberry32, sequence: sequence, randomSeed: randomSeed };
})();
