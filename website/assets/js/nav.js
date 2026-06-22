/* ==========================================================================
   nav.js — injects the shared header + footer so markup lives in one place.

   Each page must set two attributes on <body>:
     data-base="..."   relative path to the website/ root ("" for top-level
                       pages, "../" for pages inside sectors/)
     data-page="..."   id of the current page, used to highlight the nav
                       (home | transport | buildings | industry | population | about)
   ========================================================================== */
(function () {
  var base = document.body.getAttribute("data-base") || "";
  var page = document.body.getAttribute("data-page") || "";

  // Primary navigation entries (label, page id, href relative to root).
  var links = [
    { id: "home",       label: "Home",        href: "index.html" },
    { id: "transport",  label: "Transport",   href: "sectors/transport.html" },
    { id: "buildings",  label: "Buildings",   href: "sectors/buildings.html" },
    { id: "industry",   label: "Industry",    href: "sectors/industry.html" },
    { id: "population", label: "Population",   href: "sectors/population.html" },
    { id: "about",      label: "Methodology", href: "about.html" }
  ];

  var navHtml = links.map(function (l) {
    var cls = l.id === page ? ' class="active"' : "";
    return '<a href="' + base + l.href + '"' + cls + ">" + l.label + "</a>";
  }).join("");

  var header =
    '<div class="header-bar">' +
      'The hypotheses behind the first negaWatt Belgium scenario &mdash; ' +
      '<a href="https://www.negawatt.be" target="_blank" rel="noopener">negawatt.be</a>' +
    '</div>' +
    '<div class="header-main"><div class="container">' +
      '<a class="brand" href="' + base + 'index.html">' +
        '<img src="' + base + 'assets/img/nW_BE_logo_rectangle.png" alt="negaWatt Belgium">' +
      '</a>' +
      '<button class="nav-toggle" aria-label="Toggle navigation">&#9776;</button>' +
      '<nav class="nav">' + navHtml + '</nav>' +
    '</div></div>';

  // Inline SVG icon sprite (Feather-style, 24x24). Use with
  //   <svg class="ico"><use href="#i-car"></use></svg>
  var sprite =
    '<svg xmlns="http://www.w3.org/2000/svg" style="display:none" aria-hidden="true">' +
    '<defs>' +
    icon("i-car", '<path d="M5 13l1.5-4.5A2 2 0 0 1 8.4 7h7.2a2 2 0 0 1 1.9 1.5L19 13"/><path d="M3 13h18v4H3z"/><circle cx="7" cy="17.5" r="1.5"/><circle cx="17" cy="17.5" r="1.5"/>') +
    icon("i-bus", '<rect x="4" y="4" width="16" height="13" rx="2"/><path d="M4 11h16"/><circle cx="8" cy="19" r="1.3"/><circle cx="16" cy="19" r="1.3"/>') +
    icon("i-train", '<rect x="6" y="3" width="12" height="13" rx="3"/><path d="M6 10h12"/><path d="M9 20l-2 2M15 20l2 2"/><circle cx="9" cy="13" r="1"/><circle cx="15" cy="13" r="1"/>') +
    icon("i-tram", '<rect x="6" y="4" width="12" height="12" rx="2"/><path d="M12 2v2M6 9h12M8 20l8 0M10 16l-1 4M14 16l1 4"/>') +
    icon("i-bike", '<circle cx="6" cy="17" r="3"/><circle cx="18" cy="17" r="3"/><path d="M6 17l4-7h5l-3 7M10 10l2-3h2"/>') +
    icon("i-walk", '<circle cx="13" cy="4" r="1.6"/><path d="M13 7l-2 4 3 2 1 5M11 11l-3 1-1 4M14 13l3 1"/>') +
    icon("i-moto", '<circle cx="5" cy="17" r="2.5"/><circle cx="19" cy="17" r="2.5"/><path d="M5 17l4-4h5l2 4M9 13l-2-3H5M14 9h4"/>') +
    icon("i-plane", '<path d="M10 3.5a1.5 1.5 0 0 1 3 0V10l8 4.5V17l-8-2.5V20l2.5 1.5V23L11.5 22 8 23v-1.5L10.5 20v-5.5L3 17v-2.5L10 10z"/>') +
    icon("i-truck", '<rect x="2" y="6" width="12" height="9"/><path d="M14 9h4l3 3v3h-7z"/><circle cx="6" cy="17.5" r="1.6"/><circle cx="18" cy="17.5" r="1.6"/>') +
    icon("i-van", '<rect x="2" y="7" width="14" height="8" rx="1"/><path d="M16 9h2.5l2.5 3v3H16z"/><circle cx="7" cy="17" r="1.5"/><circle cx="18" cy="17" r="1.5"/>') +
    icon("i-ship", '<path d="M3 15l1.5 5h15L21 15zM5 15V8h14v7M9 8V5h6v3M12 2v3"/>') +
    icon("i-bolt", '<path d="M13 2L4 14h6l-1 8 9-12h-6z"/>') +
    icon("i-gauge", '<path d="M4 19a8 8 0 1 1 16 0"/><path d="M12 19l4-6"/>') +
    icon("i-users", '<circle cx="9" cy="8" r="3"/><path d="M3 20a6 6 0 0 1 12 0"/><path d="M16 5.5a3 3 0 0 1 0 5M21 20a6 6 0 0 0-5-5.9"/>') +
    icon("i-leaf", '<path d="M5 19C5 9 13 5 20 5c0 9-5 14-12 14a6 6 0 0 1-3-0.8z"/><path d="M5 19c3-4 7-7 11-8"/>') +
    icon("i-home", '<path d="M4 11l8-7 8 7"/><path d="M6 10v9h12v-9"/><path d="M10 19v-5h4v5"/>') +
    icon("i-factory", '<path d="M3 21V10l6 4V10l6 4V7h3v14z"/><path d="M3 21h18"/>') +
    icon("i-fire", '<path d="M12 3c2 3 5 5 5 9a5 5 0 0 1-10 0c0-2 1-3 2-4 0 1 1 2 2 2 0-3-1-5-1-7z"/>') +
    icon("i-snow", '<path d="M12 2v20M4 6l16 12M20 6L4 18M12 6l3 2M12 6l-3 2M12 18l3-2M12 18l-3-2"/>') +
    icon("i-drop", '<path d="M12 3c4 5 6 8 6 11a6 6 0 0 1-12 0c0-3 2-6 6-11z"/>') +
    icon("i-bulb", '<path d="M9 18h6M10 21h4"/><path d="M12 3a6 6 0 0 0-4 10c1 1 1 2 1 3h6c0-1 0-2 1-3a6 6 0 0 0-4-10z"/>') +
    icon("i-fridge", '<rect x="6" y="3" width="12" height="18" rx="2"/><path d="M6 10h12M9 6v2M9 13v3"/>') +
    icon("i-chip", '<rect x="7" y="7" width="10" height="10" rx="1"/><path d="M10 4v3M14 4v3M10 17v3M14 17v3M4 10h3M4 14h3M17 10h3M17 14h3"/>') +
    icon("i-people", '<circle cx="12" cy="7" r="3.5"/><path d="M5 21a7 7 0 0 1 14 0"/>') +
    icon("i-cube", '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"/>') +
    icon("i-recycle", '<path d="M7 7l2-3 3 1M17 8l1 3-3 1M9 17l-3-1 0-3M12 5l3 5-3 1M19 11l-2 5-3-1M5 13l3 4 3-1"/>') +
    '</defs></svg>';

  function icon(id, body) {
    return '<symbol id="' + id + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
           'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">' + body + "</symbol>";
  }

  var spriteWrap = document.createElement("div");
  spriteWrap.innerHTML = sprite;
  document.body.insertBefore(spriteWrap, document.body.firstChild);

  var headerEl = document.createElement("header");
  headerEl.className = "site-header";
  headerEl.innerHTML = header;
  document.body.insertBefore(headerEl, document.body.firstChild);

  // Mobile nav toggle.
  var toggle = headerEl.querySelector(".nav-toggle");
  var nav = headerEl.querySelector(".nav");
  toggle.addEventListener("click", function () { nav.classList.toggle("open"); });

  var footer = document.createElement("footer");
  footer.className = "site-footer";
  footer.innerHTML =
    '<div class="container">' +
      '<div class="foot-brand">' +
        '<img src="' + base + 'assets/img/nW_BE_logo_rectangle.png" alt="negaWatt Belgium"><br>' +
        '<span class="small">A society-wide debate around a sober and fair energy transition.</span>' +
      '</div>' +
      '<div><h4>Sectors</h4><ul>' +
        '<li><a href="' + base + 'sectors/transport.html">Transport</a></li>' +
        '<li><a href="' + base + 'sectors/buildings.html">Buildings</a></li>' +
        '<li><a href="' + base + 'sectors/industry.html">Industry</a></li>' +
        '<li><a href="' + base + 'sectors/population.html">Population &amp; context</a></li>' +
      '</ul></div>' +
      '<div><h4>Go deeper</h4><ul>' +
        '<li><a href="' + base + 'about.html">Methodology</a></li>' +
        '<li><a href="' + base + 'notebooks/nW_BE_demand_model_transports.html">Calculation notebooks</a></li>' +
        '<li><a href="https://www.negawatt.be" target="_blank" rel="noopener">negawatt.be</a></li>' +
      '</ul></div>' +
      '<div class="copyright">' +
        'Hypotheses and figures are generated directly from the negaWatt-BE demand notebooks. ' +
        'Reference year 2019, horizon 2050.' +
      '</div>' +
    '</div>';
  document.body.appendChild(footer);
})();
