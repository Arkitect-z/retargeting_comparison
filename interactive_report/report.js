(() => {
  "use strict";

  const DATA = JSON.parse(document.getElementById("rtcmp-data").textContent);
  const NS = "http://www.w3.org/2000/svg";
  const METHODS = DATA.operating_points;
  const METHOD_META = {
    "sparse-neutral": { label: "Sparse", short: "SPARSE", color: "#f17842", description: "Minimal root, hand, and foot tasks; neutral initialization." },
    dense: { label: "Dense", short: "DENSE", color: "#1f9e93", description: "Controlled Mink baseline with expanded whole-body targets." },
    gmr: { label: "GMR", short: "GMR", color: "#e9b949", description: "Official LAFAN-to-G1 generalized motion retargeting path." },
    omniretarget: { label: "OmniRetarget", short: "OMNI", color: "#8575ef", description: "Official Holosoma LAFAN-to-G1 whole-body path." },
  };
  const METRICS = {
    rf_kpe_all_mean_m: { label: "RF-KPE all", unit: "m", digits: 4 },
    rf_kpe_targeted_mean_m: { label: "Targeted RF-KPE", unit: "m", digits: 4 },
    rf_kpe_untracked_mean_m: { label: "Untracked RF-KPE", unit: "m", digits: 4 },
    root_translation_common_scale_mean_m: { label: "Root error · common scale", unit: "m", digits: 4 },
    root_translation_native_scale_mean_m: { label: "Root error · native scale", unit: "m", digits: 4 },
    root_translation_scale_invariant_mean_m: { label: "Root error · path shape", unit: "m", digits: 4 },
    rf_kpe_all_m: { label: "RF-KPE all", unit: "m", digits: 3 },
    rf_kpe_targeted_m: { label: "Targeted KPE", unit: "m", digits: 3 },
    rf_kpe_untracked_m: { label: "Untracked KPE", unit: "m", digits: 3 },
    ground_penetration_depth_m: { label: "Ground penetration", unit: "m", digits: 3 },
    pose_jump_rms_m: { label: "Pose jump", unit: "m", digits: 3 },
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
  const format = (value, digits = 3) => Number(value).toFixed(digits);
  const pct = (value, digits = 1) => `${(Number(value) * 100).toFixed(digits)}%`;
  const titleCase = (value) => String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

  function svgNode(tag, attrs = {}, text = null) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
    if (text !== null) node.textContent = text;
    return node;
  }

  function htmlNode(tag, className = "", text = null) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== null) node.textContent = text;
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function svgSize(svg, fallbackHeight) {
    const width = Math.max(320, Math.round(svg.getBoundingClientRect().width || 960));
    const height = Math.max(280, Math.round(svg.getBoundingClientRect().height || fallbackHeight));
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    return { width, height };
  }

  function linear(domainMin, domainMax, rangeMin, rangeMax) {
    const span = domainMax - domainMin || 1;
    return (value) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
  }

  function logScale(domainMin, domainMax, rangeMin, rangeMax) {
    const low = Math.log10(domainMin);
    const high = Math.log10(domainMax);
    return (value) => rangeMin + ((Math.log10(value) - low) / (high - low)) * (rangeMax - rangeMin);
  }

  function niceTicks(minimum, maximum, count = 5) {
    const span = maximum - minimum || 1;
    const rough = span / Math.max(1, count - 1);
    const power = 10 ** Math.floor(Math.log10(rough));
    const normalized = rough / power;
    const step = (normalized < 1.5 ? 1 : normalized < 3 ? 2 : normalized < 7 ? 5 : 10) * power;
    const start = Math.floor(minimum / step) * step;
    const ticks = [];
    for (let value = start; value <= maximum + step * 0.5; value += step) {
      if (value >= minimum - step * 0.2) ticks.push(value);
    }
    return ticks;
  }

  function linePath(values, xScale, yScale) {
    return values.map((value, index) => `${index ? "L" : "M"}${xScale(index).toFixed(2)},${yScale(value).toFixed(2)}`).join(" ");
  }

  function showTooltip(tooltip, event, html) {
    tooltip.innerHTML = html;
    const stage = tooltip.parentElement.getBoundingClientRect();
    tooltip.style.left = `${clamp(event.clientX - stage.left, 100, stage.width - 100)}px`;
    tooltip.style.top = `${clamp(event.clientY - stage.top, 75, stage.height - 10)}px`;
    tooltip.classList.add("visible");
  }

  function hideTooltip(tooltip) {
    tooltip.classList.remove("visible");
  }

  function setActiveButton(container, value) {
    $$('button[data-value]', container).forEach((button) => {
      const active = button.dataset.value === value;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
  }

  function methodStyle(method) {
    return `--method-color:${METHOD_META[method].color}`;
  }

  function initTheme() {
    const toggle = $("#theme-toggle");
    let saved = null;
    try { saved = localStorage.getItem("rtcmp-theme"); } catch (_) { /* file privacy mode */ }
    if (saved === "dark" || saved === "light") document.documentElement.dataset.theme = saved;
    toggle.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("rtcmp-theme", next); } catch (_) { /* file privacy mode */ }
      window.dispatchEvent(new Event("resize"));
    });
  }

  function initScrollNarrative() {
    const progress = $("#reading-progress-bar");
    const chapters = $$(".chapter[id]");
    const railLinks = $$(".chapter-rail a");
    const update = () => {
      const scrollable = document.documentElement.scrollHeight - window.innerHeight;
      progress.style.width = `${scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0}%`;
      const target = chapters.reduce((current, chapter) => {
        const distance = Math.abs(chapter.getBoundingClientRect().top - window.innerHeight * 0.33);
        return distance < current.distance ? { id: chapter.id, distance } : current;
      }, { id: "design", distance: Infinity });
      railLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${target.id}`));
    };
    window.addEventListener("scroll", update, { passive: true });
    update();

    if (!("IntersectionObserver" in window)) {
      $$(".reveal").forEach((node) => node.classList.add("in-view"));
      return;
    }
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("in-view");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12 });
    $$(".reveal").forEach((node) => observer.observe(node));
  }

  function initDesign() {
    const checks = Object.values(DATA.validation.checks);
    const stats = [
      [DATA.pilot.num_frames, "frames", "frozen source"],
      [format(DATA.pilot.duration_s, 2), "seconds", "at 30 fps"],
      [DATA.core.length, "operating points", "all complete"],
      [new Set(DATA.interaction.map((row) => row.case)).size, "interaction cases", "Full + No-Hard"],
      [`${checks.filter(Boolean).length}/${checks.length}`, "acceptance checks", "all passed"],
    ];
    const grid = $("#design-stats");
    stats.forEach(([value, label, note]) => {
      const card = htmlNode("article", "stat-card");
      card.append(htmlNode("span", "", label), htmlNode("strong", "", value), htmlNode("small", "", note));
      grid.append(card);
    });
    const ledger = $("#method-ledger");
    METHODS.forEach((method) => {
      const card = htmlNode("article", "method-card");
      card.style.cssText = methodStyle(method);
      const top = htmlNode("div", "method-card-top");
      top.append(htmlNode("strong", "", METHOD_META[method].label), htmlNode("i", "method-swatch"));
      card.append(top, htmlNode("p", "", METHOD_META[method].description));
      ledger.append(card);
    });
  }

  let frontierMetric = "rf_kpe_all_mean_m";
  function renderFrontier() {
    const svg = $("#frontier-chart");
    const tooltip = $("#frontier-tooltip");
    clear(svg);
    const { width, height } = svgSize(svg, 560);
    const margin = { top: 45, right: 55, bottom: 72, left: 76 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;
    const xValues = DATA.core.map((row) => Number(row.end_to_end_rtf_median));
    const yValues = DATA.core.map((row) => Number(row[frontierMetric]));
    const yPad = Math.max((Math.max(...yValues) - Math.min(...yValues)) * 0.32, 0.005);
    const yMin = Math.max(0, Math.min(...yValues) - yPad);
    const yMax = Math.max(...yValues) + yPad;
    const x = logScale(Math.min(...xValues) * 0.7, Math.max(...xValues) * 1.35, margin.left, margin.left + innerWidth);
    const y = linear(yMin, yMax, margin.top + innerHeight, margin.top);
    const xTicks = [0.03, 0.1, 0.3, 1, 3, 10].filter((tick) => tick >= Math.min(...xValues) * .7 && tick <= Math.max(...xValues) * 1.35);
    const yTicks = niceTicks(yMin, yMax, 6);

    const grid = svgNode("g", { class: "axis" });
    xTicks.forEach((tick) => {
      const px = x(tick);
      grid.append(svgNode("line", { class: "grid-line", x1: px, x2: px, y1: margin.top, y2: margin.top + innerHeight }));
      grid.append(svgNode("text", { x: px, y: margin.top + innerHeight + 25, "text-anchor": "middle" }, tick < 1 ? tick.toFixed(2).replace(/0+$/, "") : String(tick)));
    });
    yTicks.forEach((tick) => {
      const py = y(tick);
      grid.append(svgNode("line", { class: "grid-line", x1: margin.left, x2: margin.left + innerWidth, y1: py, y2: py }));
      grid.append(svgNode("text", { x: margin.left - 13, y: py + 4, "text-anchor": "end" }, tick.toFixed(3)));
    });
    grid.append(svgNode("text", { class: "axis-label", x: margin.left + innerWidth / 2, y: height - 15, "text-anchor": "middle" }, "END-TO-END STEADY RTF · LOG SCALE · LOWER IS BETTER"));
    const metric = METRICS[frontierMetric];
    grid.append(svgNode("text", { class: "axis-label", x: 17, y: margin.top + innerHeight / 2, transform: `rotate(-90 17 ${margin.top + innerHeight / 2})`, "text-anchor": "middle" }, `${metric.label.toUpperCase()} (${metric.unit}) · LOWER IS BETTER`));
    svg.append(grid);

    DATA.core.forEach((row, index) => {
      const method = row.label;
      const radius = 7 + Math.sqrt(Number(row.artifact_rate)) * 30;
      const px = x(Number(row.end_to_end_rtf_median));
      const py = y(Number(row[frontierMetric]));
      const group = svgNode("g", { class: "data-point", tabindex: "0", role: "button", "aria-label": `${METHOD_META[method].label}, RTF ${format(row.end_to_end_rtf_median, 3)}, ${metric.label} ${format(row[frontierMetric], 4)} metres` });
      const halo = svgNode("circle", { cx: px, cy: py, r: radius + 6, fill: "none", stroke: METHOD_META[method].color, "stroke-opacity": .2 });
      const circle = svgNode("circle", { cx: px, cy: py, r: radius, fill: METHOD_META[method].color, "fill-opacity": .88, stroke: "#fff", "stroke-width": 1.5 });
      const labelAnchor = px > width * .73 ? "end" : "start";
      const labelX = labelAnchor === "end" ? px - radius - 10 : px + radius + 10;
      const label = svgNode("text", { class: "point-label", x: labelX, y: py + 4, fill: METHOD_META[method].color, "text-anchor": labelAnchor }, METHOD_META[method].short);
      group.append(halo, circle, label);
      const popup = (event) => showTooltip(tooltip, event, `<strong>${METHOD_META[method].label}</strong><div class="tooltip-grid"><span>Steady RTF</span><b>${format(row.end_to_end_rtf_median, 3)}</b><span>${metric.label}</span><b>${format(row[frontierMetric], metric.digits)} ${metric.unit}</b><span>Artifact rate</span><b>${pct(row.artifact_rate)}</b></div>`);
      group.addEventListener("pointerenter", popup);
      group.addEventListener("pointermove", popup);
      group.addEventListener("pointerleave", () => hideTooltip(tooltip));
      group.addEventListener("focus", () => {
        const box = svg.getBoundingClientRect();
        popup({ clientX: box.left + px, clientY: box.top + py });
      });
      group.addEventListener("blur", () => hideTooltip(tooltip));
      svg.append(group);
      if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        group.animate([{ opacity: 0, transform: `translateY(${40 + index * 3}px)` }, { opacity: 1, transform: "translateY(0)" }], { duration: 550, delay: index * 90, easing: "cubic-bezier(.2,.8,.2,1)", fill: "both" });
      }
    });

    const lowest = DATA.core.reduce((best, row) => Number(row[frontierMetric]) < Number(best[frontierMetric]) ? row : best);
    const fastest = DATA.core.reduce((best, row) => Number(row.end_to_end_rtf_median) < Number(best.end_to_end_rtf_median) ? row : best);
    $("#frontier-annotation").innerHTML = `<span><strong>${METHOD_META[lowest.label].label}</strong> has the lowest ${metric.label.toLowerCase()} in this Pilot.</span><span><strong>${METHOD_META[fastest.label].label}</strong> is ${format(Number(lowest.end_to_end_rtf_median) / Number(fastest.end_to_end_rtf_median), 0)}× faster than that point.</span>`;
  }

  function initFrontier() {
    const controls = $("#frontier-metric");
    controls.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button) return;
      frontierMetric = button.dataset.value;
      setActiveButton(controls, frontierMetric);
      renderFrontier();
    });
    const fastest = DATA.core.reduce((best, row) => Number(row.end_to_end_rtf_median) < Number(best.end_to_end_rtf_median) ? row : best);
    const quality = DATA.core.reduce((best, row) => Number(row.rf_kpe_all_mean_m) < Number(best.rf_kpe_all_mean_m) ? row : best);
    const artifacts = DATA.core.reduce((best, row) => Number(row.artifact_rate) < Number(best.artifact_rate) ? row : best);
    const findings = [
      ["FASTEST", `${format(fastest.end_to_end_rtf_median, 3)} RTF`, `${METHOD_META[fastest.label].label} is the fastest measured operating point.`],
      ["LOWEST RF-KPE ALL", `${format(quality.rf_kpe_all_mean_m, 4)} m`, `${METHOD_META[quality.label].label} leads this one-sequence quality lens.`],
      ["LOWEST ARTIFACT RATE", pct(artifacts.artifact_rate), `${METHOD_META[artifacts.label].label} has the fewest flagged frames.`],
    ];
    const grid = $("#frontier-findings");
    findings.forEach(([label, value, note]) => {
      const card = htmlNode("article", "finding reveal");
      card.append(htmlNode("span", "", label), htmlNode("strong", "", value), htmlNode("p", "", note));
      grid.append(card);
    });
    renderFrontier();
  }

  const timelineState = { metric: "rf_kpe_all_m", active: new Set(METHODS), frame: 0, playing: false, animation: null, started: 0, startFrame: 0 };
  function timelineYDomain() {
    const values = [];
    timelineState.active.forEach((method) => values.push(...DATA.frame_series[method][timelineState.metric].map(Number)));
    const finite = values.filter(Number.isFinite);
    return Math.max(Math.max(...finite) * 1.08, .001);
  }

  function renderTimeline() {
    const svg = $("#timeline-chart");
    clear(svg);
    const { width, height } = svgSize(svg, 430);
    const margin = { top: 25, right: 28, bottom: 88, left: 65 };
    const plotBottom = height - margin.bottom;
    const x = linear(0, DATA.pilot.num_frames - 1, margin.left, width - margin.right);
    const yMax = timelineYDomain();
    const y = linear(0, yMax, plotBottom, margin.top);
    const axis = svgNode("g", { class: "axis" });
    niceTicks(0, yMax, 6).forEach((tick) => {
      const py = y(tick);
      axis.append(svgNode("line", { class: "grid-line", x1: margin.left, x2: width - margin.right, y1: py, y2: py }));
      axis.append(svgNode("text", { x: margin.left - 10, y: py + 4, "text-anchor": "end" }, tick.toFixed(tick < .1 ? 3 : 2)));
    });
    [0, 150, 300, 450, 599].forEach((frame) => {
      const px = x(frame);
      axis.append(svgNode("text", { x: px, y: plotBottom + 24, "text-anchor": "middle" }, `${(frame / Number(DATA.pilot.fps)).toFixed(0)}s`));
    });
    axis.append(svgNode("text", { class: "axis-label", x: 18, y: (margin.top + plotBottom) / 2, transform: `rotate(-90 18 ${(margin.top + plotBottom) / 2})`, "text-anchor": "middle" }, `${METRICS[timelineState.metric].label.toUpperCase()} (${METRICS[timelineState.metric].unit})`));
    svg.append(axis);

    const artifactStart = plotBottom + 38;
    METHODS.forEach((method, methodIndex) => {
      if (!timelineState.active.has(method)) return;
      const values = DATA.frame_series[method][timelineState.metric].map(Number);
      svg.append(svgNode("path", { class: "line-path", d: linePath(values, x, y), stroke: METHOD_META[method].color, opacity: .9 }));
      const rowY = artifactStart + methodIndex * 9;
      svg.append(svgNode("text", { x: margin.left - 8, y: rowY + 4, fill: METHOD_META[method].color, "text-anchor": "end", "font-size": 7, "font-weight": 800 }, METHOD_META[method].short.slice(0, 3)));
      DATA.frame_series[method].artifact.forEach((flag, frame) => {
        if (flag) svg.append(svgNode("rect", { x: x(frame), y: rowY, width: Math.max(1.3, (width - margin.left - margin.right) / 600), height: 5, fill: METHOD_META[method].color, opacity: .65 }));
      });
    });
    svg.append(svgNode("text", { class: "axis-label", x: margin.left, y: artifactStart - 8 }, "CAUSE-TRIGGERED FRAME FLAGS BY METHOD"));
    const head = svgNode("line", { id: "timeline-playhead", class: "playhead", x1: x(timelineState.frame), x2: x(timelineState.frame), y1: margin.top, y2: plotBottom + 67 });
    svg.append(head);
    svg.dataset.plotLeft = margin.left;
    svg.dataset.plotRight = width - margin.right;
    svg.dataset.plotTop = margin.top;
    svg.dataset.plotBottom = plotBottom;
    svg.onpointermove = timelinePointerMove;
    svg.onpointerleave = () => hideTooltip($("#timeline-tooltip"));
    updateTimelineReadout(timelineState.frame);
  }

  function timelinePointerMove(event) {
    const svg = $("#timeline-chart");
    const box = svg.getBoundingClientRect();
    const width = Number(svg.viewBox.baseVal.width);
    const px = (event.clientX - box.left) / box.width * width;
    const left = Number(svg.dataset.plotLeft);
    const right = Number(svg.dataset.plotRight);
    const frame = Math.round(clamp((px - left) / (right - left), 0, 1) * (DATA.pilot.num_frames - 1));
    updateTimelineFrame(frame);
    const rows = [...timelineState.active].map((method) => {
      const cause = DATA.frame_series[method].artifact_causes[frame];
      const suffix = cause && cause !== "none" ? ` · ${cause}` : "";
      return `<span>${METHOD_META[method].short}${suffix}</span><b>${format(DATA.frame_series[method][timelineState.metric][frame], METRICS[timelineState.metric].digits)}</b>`;
    }).join("");
    showTooltip($("#timeline-tooltip"), event, `<strong>Frame ${String(frame).padStart(3,"0")} · ${(frame / Number(DATA.pilot.fps)).toFixed(2)} s</strong><div class="tooltip-grid">${rows}</div>`);
  }

  function updateTimelineFrame(frame) {
    timelineState.frame = clamp(Math.round(frame), 0, DATA.pilot.num_frames - 1);
    $("#timeline-scrubber").value = timelineState.frame;
    const svg = $("#timeline-chart");
    const head = $("#timeline-playhead", svg);
    if (head) {
      const x = linear(0, DATA.pilot.num_frames - 1, Number(svg.dataset.plotLeft), Number(svg.dataset.plotRight));
      head.setAttribute("x1", x(timelineState.frame));
      head.setAttribute("x2", x(timelineState.frame));
    }
    updateTimelineReadout(timelineState.frame);
  }

  function updateTimelineReadout(frame) {
    $("#frame-number").textContent = String(frame).padStart(3, "0");
    $("#frame-time").textContent = `${(frame / Number(DATA.pilot.fps)).toFixed(2)} s`;
    const values = $("#frame-values");
    clear(values);
    METHODS.filter((method) => timelineState.active.has(method)).forEach((method) => {
      const card = htmlNode("div", "frame-value");
      card.style.cssText = methodStyle(method);
      const cause = DATA.frame_series[method].artifact_causes[frame];
      const label = cause && cause !== "none" ? `${METHOD_META[method].short} · ${cause}` : METHOD_META[method].short;
      card.append(htmlNode("small", "", label), htmlNode("strong", "", `${format(DATA.frame_series[method][timelineState.metric][frame], METRICS[timelineState.metric].digits)} ${METRICS[timelineState.metric].unit}`));
      values.append(card);
    });
  }

  function timelineAnimation(timestamp) {
    if (!timelineState.playing) return;
    if (!timelineState.started) timelineState.started = timestamp;
    const elapsed = (timestamp - timelineState.started) / 1000;
    const frame = timelineState.startFrame + Math.floor(elapsed * Number(DATA.pilot.fps));
    if (frame >= DATA.pilot.num_frames) {
      timelineState.playing = false;
      timelineState.started = 0;
      $("#timeline-play").classList.remove("playing");
      $("#timeline-play").innerHTML = '<span aria-hidden="true">▶</span> Play';
      updateTimelineFrame(DATA.pilot.num_frames - 1);
      return;
    }
    updateTimelineFrame(frame);
    timelineState.animation = requestAnimationFrame(timelineAnimation);
  }

  function initTimeline() {
    const toggles = $("#timeline-methods");
    METHODS.forEach((method) => {
      const button = htmlNode("button", "method-toggle active", METHOD_META[method].short);
      button.type = "button";
      button.dataset.method = method;
      button.style.cssText = methodStyle(method);
      button.setAttribute("aria-pressed", "true");
      toggles.append(button);
    });
    toggles.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-method]");
      if (!button) return;
      const method = button.dataset.method;
      if (timelineState.active.has(method) && timelineState.active.size === 1) return;
      timelineState.active.has(method) ? timelineState.active.delete(method) : timelineState.active.add(method);
      button.classList.toggle("active", timelineState.active.has(method));
      button.setAttribute("aria-pressed", String(timelineState.active.has(method)));
      renderTimeline();
    });
    $("#timeline-metric").addEventListener("change", (event) => {
      timelineState.metric = event.target.value;
      renderTimeline();
    });
    $("#timeline-scrubber").addEventListener("input", (event) => {
      if (timelineState.playing) $("#timeline-play").click();
      updateTimelineFrame(Number(event.target.value));
    });
    $("#timeline-play").addEventListener("click", (event) => {
      timelineState.playing = !timelineState.playing;
      event.currentTarget.classList.toggle("playing", timelineState.playing);
      event.currentTarget.innerHTML = timelineState.playing ? '<span aria-hidden="true">Ⅱ</span> Pause' : '<span aria-hidden="true">▶</span> Play';
      if (timelineState.playing) {
        timelineState.startFrame = timelineState.frame >= DATA.pilot.num_frames - 1 ? 0 : timelineState.frame;
        timelineState.started = 0;
        timelineState.animation = requestAnimationFrame(timelineAnimation);
      } else if (timelineState.animation) {
        cancelAnimationFrame(timelineState.animation);
      }
    });
    renderTimeline();
  }

  const seedState = { pair: Object.keys(DATA.seed_series)[0], frame: 0, auto: true, last: 0 };
  function renderSeedChart() {
    const series = DATA.seed_series[seedState.pair];
    const svg = $("#seed-chart");
    clear(svg);
    const { width, height } = svgSize(svg, 370);
    const margin = { top: 24, right: 25, bottom: 52, left: 62 };
    const x = linear(0, 599, margin.left, width - margin.right);
    const maxValue = Math.max(...series.robot_rf_point_rms_m.map(Number)) * 1.08;
    const y = linear(0, maxValue, height - margin.bottom, margin.top);
    const axis = svgNode("g", { class: "axis" });
    niceTicks(0, maxValue, 5).forEach((tick) => {
      const py = y(tick);
      axis.append(svgNode("line", { x1: margin.left, x2: width - margin.right, y1: py, y2: py, stroke: "rgba(255,255,255,.14)" }));
      axis.append(svgNode("text", { x: margin.left - 9, y: py + 4, "text-anchor": "end" }, tick.toFixed(3)));
    });
    [0, 150, 300, 450, 599].forEach((frame) => axis.append(svgNode("text", { x: x(frame), y: height - 18, "text-anchor": "middle" }, `${(frame / Number(DATA.pilot.fps)).toFixed(0)}s`)));
    axis.append(svgNode("text", { class: "axis-label", x: 16, y: height / 2, transform: `rotate(-90 16 ${height/2})`, "text-anchor": "middle" }, "ROBOT RF POINT RMS (m)"));
    svg.append(axis);
    svg.append(svgNode("path", { class: "line-path", d: linePath(series.robot_rf_point_rms_m.map(Number), x, y), stroke: "#fff" }));
    svg.append(svgNode("path", { d: `${linePath(series.robot_rf_point_rms_m.map(Number), x, y)} L${x(599)},${y(0)} L${x(0)},${y(0)} Z`, fill: "rgba(255,255,255,.08)" }));
    svg.append(svgNode("line", { id: "seed-playhead", x1: x(seedState.frame), x2: x(seedState.frame), y1: margin.top, y2: height - margin.bottom, stroke: "#ffb38d", "stroke-width": 1.5, "stroke-dasharray": "4 4" }));
    svg.dataset.plotLeft = margin.left;
    svg.dataset.plotRight = width - margin.right;
    updateSeed(seedState.frame);
  }

  function updateSeed(frame) {
    seedState.frame = clamp(Math.round(frame), 0, 599);
    $("#seed-scrubber").value = seedState.frame;
    $("#seed-frame-number").textContent = String(seedState.frame).padStart(3, "0");
    const svg = $("#seed-chart");
    const head = $("#seed-playhead", svg);
    if (head) {
      const px = linear(0, 599, Number(svg.dataset.plotLeft), Number(svg.dataset.plotRight))(seedState.frame);
      head.setAttribute("x1", px); head.setAttribute("x2", px);
    }
    const pairValues = {};
    Object.entries(DATA.seed_series).forEach(([pair, series]) => { pairValues[pair] = Number(series.robot_rf_point_rms_m[seedState.frame]); });
    const scaleFor = (method) => {
      const relevant = Object.entries(pairValues).filter(([pair]) => pair.includes(method)).map(([, value]) => value);
      return 1 + clamp(Math.max(...relevant, 0) * 5, 0, .6);
    };
    $(".seed-neutral").style.transform = `scale(${scaleFor("neutral")})`;
    $(".seed-a").style.transform = `scale(${scaleFor("sparse-a")})`;
    $(".seed-b").style.transform = `scale(${scaleFor("sparse-b")})`;
    const summary = DATA.seed_summary.find((row) => row.pair === seedState.pair);
    $("#seed-summary").innerHTML = `<article><span>MEAN RF DIVERGENCE</span><strong>${format(summary.robot_rf_point_rms_mean_m, 4)} m</strong></article><article><span>MAX RF DIVERGENCE</span><strong>${format(summary.robot_rf_point_rms_max_m, 4)} m</strong></article>`;
  }

  function seedAutoFrame(timestamp) {
    if (!seedState.auto || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    if (!seedState.last) seedState.last = timestamp;
    if (timestamp - seedState.last > 55) {
      updateSeed((seedState.frame + 1) % 600);
      seedState.last = timestamp;
    }
    requestAnimationFrame(seedAutoFrame);
  }

  function initSeed() {
    const controls = $("#seed-pairs");
    Object.keys(DATA.seed_series).forEach((pair, index) => {
      const parts = pair.replaceAll("sparse-", "").split("__");
      const button = htmlNode("button", index === 0 ? "active" : "", `${parts[0]} ↔ ${parts[1]}`);
      button.type = "button";
      button.dataset.value = pair;
      controls.append(button);
    });
    controls.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button) return;
      seedState.pair = button.dataset.value;
      seedState.auto = false;
      setActiveButton(controls, seedState.pair);
      renderSeedChart();
    });
    $("#seed-scrubber").addEventListener("input", (event) => { seedState.auto = false; updateSeed(Number(event.target.value)); });
    $("#seed-orbit").style.cursor = "pointer";
    $("#seed-orbit").addEventListener("click", () => { seedState.auto = !seedState.auto; if (seedState.auto) requestAnimationFrame(seedAutoFrame); });
    renderSeedChart();
    requestAnimationFrame(seedAutoFrame);
  }

  let interactionCase = "box";
  function interactionRows() {
    return {
      full: DATA.interaction.find((row) => row.case === interactionCase && row.variant === "full"),
      noHard: DATA.interaction.find((row) => row.case === interactionCase && row.variant === "no-hard"),
    };
  }

  function renderConstraintGates() {
    const root = $("#constraint-gates");
    clear(root);
    [["Full", true], ["No-Hard", false]].forEach(([label, enabled]) => {
      const card = htmlNode("article", `gate-card ${enabled ? "full" : "no-hard"}`);
      const title = htmlNode("div", "gate-title");
      title.append(htmlNode("strong", "", label), htmlNode("span", "", enabled ? "HARD ON" : "HARD OFF"));
      const list = htmlNode("div", "gate-list");
      [["Non-penetration", enabled], ["Foot sticking", enabled], ["Joint limits", true]].forEach(([name, on]) => list.append(htmlNode("span", `gate ${on ? "on" : ""}`, name)));
      card.append(title, list); root.append(card);
    });
  }

  function renderInteraction() {
    const { full, noHard } = interactionRows();
    const svg = $("#interaction-chart");
    clear(svg);
    const { width, height } = svgSize(svg, 420);
    const margin = { top: 28, right: 55, bottom: 42, left: 175 };
    const x = linear(0, 1, margin.left, width - margin.right);
    const metrics = [
      ["Strict contact ≤2 cm", "strict_contact_2cm_frame_rate"],
      ["Near contact ≤5 cm", "near_contact_5cm_frame_rate"],
      ["Penetration >1.1 mm", "penetration_frame_rate"],
      ["Foot-sticking violation", "foot_sticking_violation_frame_rate"],
    ];
    const rowGap = (height - margin.top - margin.bottom) / metrics.length;
    [0, .25, .5, .75, 1].forEach((tick) => {
      const px = x(tick);
      svg.append(svgNode("line", { class: "grid-line", x1: px, x2: px, y1: margin.top, y2: height - margin.bottom }));
      svg.append(svgNode("text", { x: px, y: height - 14, "text-anchor": "middle", fill: "#687080", "font-size": 10, "font-family": "monospace" }, pct(tick, 0)));
    });
    metrics.forEach(([label, key], index) => {
      const py = margin.top + rowGap * (index + .5);
      const a = x(Number(full[key]));
      const b = x(Number(noHard[key]));
      svg.append(svgNode("text", { x: margin.left - 15, y: py + 4, "text-anchor": "end", fill: "currentColor", "font-size": 11, "font-weight": 700 }, label));
      svg.append(svgNode("line", { x1: Math.min(a,b), x2: Math.max(a,b), y1: py, y2: py, stroke: "rgba(104,112,128,.45)", "stroke-width": 3, "stroke-linecap": "round" }));
      svg.append(svgNode("circle", { cx: a, cy: py, r: 9, fill: "#2449d8", stroke: "#fff", "stroke-width": 2 }));
      svg.append(svgNode("circle", { cx: b, cy: py, r: 9, fill: "#f17842", stroke: "#fff", "stroke-width": 2 }));
      svg.append(svgNode("text", { x: a, y: py - 15, "text-anchor": "middle", fill: "#2449d8", "font-size": 9, "font-family": "monospace", "font-weight": 700 }, pct(full[key], 1)));
      svg.append(svgNode("text", { x: b, y: py + 25, "text-anchor": "middle", fill: "#f17842", "font-size": 9, "font-family": "monospace", "font-weight": 700 }, pct(noHard[key], 1)));
    });
    svg.append(svgNode("circle", { cx: margin.left, cy: 9, r: 5, fill: "#2449d8" }));
    svg.append(svgNode("text", { x: margin.left + 10, y: 13, fill: "#687080", "font-size": 9, "font-weight": 800 }, "FULL"));
    svg.append(svgNode("circle", { cx: margin.left + 68, cy: 9, r: 5, fill: "#f17842" }));
    svg.append(svgNode("text", { x: margin.left + 78, y: 13, fill: "#687080", "font-size": 9, "font-weight": 800 }, "NO-HARD"));

    const readout = $("#interaction-readout");
    clear(readout);
    const cards = [
      ["Primary penetration", pct(full.penetration_frame_rate), pct(noHard.penetration_frame_rate)],
      ["Foot-sticking violation", pct(full.foot_sticking_violation_frame_rate), pct(noHard.foot_sticking_violation_frame_rate)],
      ["Any negative distance", pct(full.penetration_any_frame_rate), pct(noHard.penetration_any_frame_rate)],
      ["End-to-end RTF", format(full.end_to_end_rtf, 2), format(noHard.end_to_end_rtf, 2)],
    ];
    cards.forEach(([label, fullValue, noHardValue]) => {
      const card = htmlNode("article", "interaction-metric");
      card.append(htmlNode("span", "", label));
      const values = htmlNode("div");
      const fullStrong = htmlNode("strong", "", fullValue); fullStrong.style.color = "#2449d8";
      const noHardWrap = htmlNode("small", "", `No-Hard ${noHardValue}`);
      values.append(fullStrong, noHardWrap); card.append(values); readout.append(card);
    });
  }

  function initInteraction() {
    renderConstraintGates();
    const controls = $("#case-switch");
    controls.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button) return;
      interactionCase = button.dataset.value;
      setActiveButton(controls, interactionCase);
      renderInteraction();
    });
    renderInteraction();
  }

  let timingMode = "end_to_end_rtf";
  function renderTiming() {
    const svg = $("#timing-chart");
    const tooltip = $("#timing-tooltip");
    clear(svg);
    const { width, height } = svgSize(svg, 470);
    const margin = { top: 28, right: 45, bottom: 62, left: 145 };
    const allValues = DATA.timing_raw.map((row) => Number(row[timingMode]));
    const min = Math.min(...allValues) * .7;
    const max = Math.max(...allValues) * 1.35;
    const x = logScale(min, max, margin.left, width - margin.right);
    const rowGap = (height - margin.top - margin.bottom) / METHODS.length;
    const ticks = [.02,.03,.05,.1,.2,.5,1,2,5,10,20].filter((tick) => tick >= min && tick <= max);
    ticks.forEach((tick) => {
      const px = x(tick);
      svg.append(svgNode("line", { class: "grid-line", x1: px, x2: px, y1: margin.top, y2: height - margin.bottom }));
      svg.append(svgNode("text", { x: px, y: height - 24, "text-anchor": "middle", fill: "#687080", "font-size": 9, "font-family": "monospace" }, tick < .1 ? tick.toFixed(2) : String(tick)));
    });
    svg.append(svgNode("text", { class: "axis-label", x: margin.left + (width - margin.right - margin.left)/2, y: height - 5, "text-anchor": "middle" }, `${timingMode === "end_to_end_rtf" ? "END-TO-END" : "NATIVE CORE"} RTF · LOG SCALE`));
    METHODS.forEach((method, methodIndex) => {
      const py = margin.top + rowGap * (methodIndex + .5);
      svg.append(svgNode("text", { x: margin.left - 16, y: py + 4, "text-anchor": "end", fill: METHOD_META[method].color, "font-size": 11, "font-weight": 800 }, METHOD_META[method].short));
      svg.append(svgNode("line", { x1: margin.left, x2: width-margin.right, y1: py, y2: py, stroke: "rgba(104,112,128,.18)" }));
      const rows = DATA.timing_raw.filter((row) => row.method === method);
      rows.forEach((row, index) => {
        const px = x(Number(row[timingMode]));
        let node;
        if (row.role === "cold") {
          node = svgNode("circle", { cx: px, cy: py, r: 8, fill: "none", stroke: METHOD_META[method].color, "stroke-width": 2.5 });
        } else if (row.role === "warmup") {
          node = svgNode("path", { d: `M${px-6},${py-6} L${px+6},${py+6} M${px+6},${py-6} L${px-6},${py+6}`, stroke: METHOD_META[method].color, "stroke-width": 2, opacity: .55 });
        } else {
          node = svgNode("circle", { cx: px, cy: py + (index-3)*2.5, r: 5.5, fill: METHOD_META[method].color, stroke: "#fff", "stroke-width": 1 });
        }
        node.classList.add("data-point");
        const popup = (event) => showTooltip(tooltip, event, `<strong>${METHOD_META[method].label} · ${row.role}</strong><div class="tooltip-grid"><span>${timingMode === "end_to_end_rtf" ? "End-to-end" : "Native core"}</span><b>${format(row[timingMode], 4)} RTF</b><span>Wall time</span><b>${format(row.wall_time_s, 3)} s</b></div>`);
        node.addEventListener("pointerenter", popup); node.addEventListener("pointermove", popup); node.addEventListener("pointerleave", () => hideTooltip(tooltip));
        svg.append(node);
      });
    });
  }

  function initTiming() {
    const controls = $("#timing-mode");
    controls.addEventListener("click", (event) => {
      const button = event.target.closest("button[data-value]");
      if (!button) return;
      timingMode = button.dataset.value;
      setActiveButton(controls, timingMode);
      renderTiming();
    });
    renderTiming();
  }

  function initBudget() {
    const projection = DATA.stage2_projection;
    const total = Number(projection.safe_projected_wall_hours_serial);
    const budget = Number(projection.wall_budget_hours);
    $("#budget-hours").textContent = `${format(total, 2)} h`;
    $(".budget-verdict strong").textContent = `+${format(total - budget, 2)} h`;
    $("#storage-value").textContent = `${format(projection.safe_projected_storage_gb, 2)} GB`;
    const track = $("#budget-track");
    const max = Math.max(total * 1.03, budget * 1.12);
    DATA.stage2_projection_rows.forEach((row) => {
      const segment = htmlNode("div", "budget-segment");
      segment.style.cssText = `${methodStyle(row.method)};width:${Number(row.safe_projected_wall_hours) / max * 100}%`;
      segment.title = `${METHOD_META[row.method].label}: ${format(row.safe_projected_wall_hours, 2)} h`;
      track.append(segment);
    });
    const line = htmlNode("div", "budget-limit");
    line.style.setProperty("--limit-left", `${budget / max * 100}%`);
    track.append(line);
    const legend = $("#budget-legend");
    DATA.stage2_projection_rows.forEach((row) => {
      const item = htmlNode("span"); item.style.cssText = methodStyle(row.method);
      item.append(htmlNode("i"), document.createTextNode(`${METHOD_META[row.method].label} · ${format(row.safe_projected_wall_hours, 2)} h`));
      legend.append(item);
    });
    const slider = $("#worker-slider");
    const update = () => {
      const workers = Number(slider.value);
      const hours = total / workers;
      $("#worker-count").textContent = workers;
      $("#worker-plural").textContent = workers === 1 ? "" : "s";
      $("#what-if-hours").textContent = `${format(hours, 2)} h`;
      const status = $("#what-if-status");
      status.textContent = hours <= budget ? "arithmetically within gate" : "over budget";
      status.classList.toggle("within", hours <= budget);
    };
    slider.addEventListener("input", update); update();
  }

  function evidenceRows(rows, labelKey, valueKey, extraKey = null, hrefKey = null) {
    return rows.map((row) => ({
      label: String(row[labelKey]),
      value: String(row[valueKey]),
      extra: extraKey ? String(row[extraKey] ?? "") : "",
      href: hrefKey ? String(row[hrefKey] ?? "") : "",
    }));
  }

  function initEvidence() {
    const models = [
      { label: "SMPL neutral · chumpy-free", value: DATA.body_models.smpl.sha256, extra: `${DATA.body_models.smpl.size_bytes} bytes · finite forward` },
      { label: "SMPL-X neutral · NPZ", value: DATA.body_models.smplx.sha256, extra: `${DATA.body_models.smplx.size_bytes} bytes · finite forward` },
    ];
    const groups = [
      ["Repository revisions", evidenceRows(DATA.repositories, "name", "commit", "status", "url")],
      ["External body models", models],
      ["Source adapter checks", DATA.adapters.map((row) => ({ label: row.adapter, value: `${format(row.root_aligned_mpjpe_m, 8)} m MPJPE`, extra: row.caveat }))],
      ["Embedded data snapshots", DATA.input_hashes.map((row) => ({ label: row.path, value: row.sha256, extra: `${row.size_bytes} bytes` }))],
      ["Stage 1 acceptance", Object.entries(DATA.validation.checks).map(([key, value]) => ({ label: titleCase(key), value: value ? "PASS" : "FAIL", extra: "" }))],
    ];
    const list = $("#evidence-list");
    groups.forEach(([title, rows], index) => {
      const item = htmlNode("article", `evidence-item ${index === 0 ? "open" : ""}`);
      const button = htmlNode("button"); button.type = "button"; button.setAttribute("aria-expanded", String(index === 0));
      button.append(htmlNode("span", "", String(index + 1).padStart(2,"0")), htmlNode("strong", "", title), htmlNode("i", "", "+"));
      const body = htmlNode("div", "evidence-body");
      const inner = htmlNode("div", "evidence-body-inner");
      rows.forEach((row) => {
        const entry = htmlNode("div", "hash-row");
        const label = htmlNode("span", "", row.label);
        const value = htmlNode("div"); value.append(htmlNode("code", "", row.value));
        if (row.extra) value.append(htmlNode("small", "", ` · ${row.extra}`));
        if (row.href) {
          const link = htmlNode("a", "evidence-source-link", "source ↗");
          link.href = row.href; link.target = "_blank"; link.rel = "noreferrer";
          value.append(link);
        }
        entry.append(label, value); inner.append(entry);
      });
      body.append(inner); item.append(button, body); list.append(item);
      button.addEventListener("click", () => {
        item.classList.toggle("open");
        button.setAttribute("aria-expanded", String(item.classList.contains("open")));
      });
    });
  }

  let resizeTimer = null;
  function initResponsiveCharts() {
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        renderFrontier(); renderTimeline(); renderSeedChart(); renderInteraction(); renderTiming();
      }, 150);
    });
  }

  function init() {
    initTheme();
    initScrollNarrative();
    initDesign();
    initFrontier();
    initTimeline();
    initSeed();
    initInteraction();
    initTiming();
    initBudget();
    initEvidence();
    initResponsiveCharts();
    const requestedChapter = new URLSearchParams(window.location.search).get("chapter")
      || window.location.hash.slice(1);
    if (requestedChapter) {
      const target = document.getElementById(requestedChapter);
      if (target) window.setTimeout(() => target.scrollIntoView({ block: "start" }), 50);
    }
  }

  init();
})();
