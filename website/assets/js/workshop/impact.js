/* ==========================================================================
   negaWatt Belgium — workshop leverage (window.NW_IMPACT)
   --------------------------------------------------------------------------
   Turns a slider value into "what this does to Belgium's 2050 inland-mobility
   energy demand". The coefficients come from the notebook (see the Workshop
   export cell of nW_BE_demand_model_transports.ipynb), so this file contains
   arithmetic only — no assumptions of its own.

   Every lever is calibrated so that f(vTarget) === impact.total, the négaWatt
   value. That puts all eight levers on one comparable scale.

   IMPORTANT: exact for a single lever moved on its own, with the others left at
   their négaWatt values. Cross-lever interaction is not modelled, and the UI
   says so.
   ========================================================================== */
(function () {
  "use strict";

  /* TWh of 2050 inland-mobility demand implied by `value` for this lever. */
  function evaluate(impact, value) {
    if (!impact || value === null || value === undefined || !isFinite(value)) return null;
    var total = impact.total;
    var scaled = impact.scaled || 0;
    var fixed = total - scaled;

    switch (impact.kind) {
      case "proportional":
        if (!impact.vTarget) return null;
        return fixed + scaled * (value / impact.vTarget);
      case "inverse":
        if (!value) return null;                     // an occupancy of zero is meaningless
        return fixed + scaled * (impact.vTarget / value);
      case "linear-shift":
        return total + (value - impact.vTarget) * (impact.slope || 0);
      case "negligible":
        return null;
      default:
        return null;
    }
  }

  /* TWh above (+) or below (-) the négaWatt scenario. */
  function delta(impact, value) {
    var twh = evaluate(impact, value);
    if (twh === null) return null;
    return twh - impact.total;
  }

  /* Same, as a percentage of the négaWatt inland-mobility total. */
  function deltaPct(impact, value) {
    var d = delta(impact, value);
    if (d === null || !impact.total) return null;
    return d / impact.total * 100;
  }

  /* Percentage change against an arbitrary reference — used by the participant
     page, which compares to the observed 2019 demand rather than to negaWatt.
     Comparing to negaWatt while the group is still deciding would let them slide
     until the difference reached zero and read the answer straight off. */
  function changeVs(impact, value, reference) {
    var twh = evaluate(impact, value);
    if (twh === null || !reference) return null;
    return (twh / reference - 1) * 100;
  }

  /* How much of the whole scenario this lever can swing across its slider range
     — used to sort levers by how much they actually matter. */
  function leverageRange(impact, slider) {
    if (!impact || !slider) return null;
    var lo = evaluate(impact, slider.min);
    var hi = evaluate(impact, slider.max);
    if (lo === null || hi === null) return null;
    return Math.abs(hi - lo);
  }

  window.NW_IMPACT = {
    evaluate: evaluate,
    delta: delta,
    deltaPct: deltaPct,
    changeVs: changeVs,
    leverageRange: leverageRange
  };
})();
