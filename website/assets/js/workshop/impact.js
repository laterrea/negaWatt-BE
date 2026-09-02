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

  /* Percentage change against an arbitrary reference. */
  function changeVs(impact, value, reference) {
    var twh = evaluate(impact, value);
    if (twh === null || !reference) return null;
    return (twh / reference - 1) * 100;
  }

  /* What THIS lever contributes: the difference between the answer and simply
     keeping the reference-year level of the same indicator, everything else
     unchanged.
     
     This is the number the participant page shows. Two earlier attempts were
     both misleading. The sector total at the answer (≈22 TWh) reads as the
     combined effect of all eight assumptions. Comparing that total to 2019
     is worse: electrification alone cuts inland mobility by three quarters, so
     every answer showed a large decrease and the sign never moved, however hard
     the group pushed the slider the wrong way.
     
     Exact for a single lever, because each response function is linear or
     hyperbolic in its own value and the others are held fixed. */
  function contribution(impact, value, refValue) {
    var here = evaluate(impact, value);
    var base = evaluate(impact, refValue);
    if (here === null || base === null) return null;
    return here - base;
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
    contribution: contribution,
    leverageRange: leverageRange
  };
})();
