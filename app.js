'use strict';
(() => {
  const content = window.IMIR_CONTENT;
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => [...document.querySelectorAll(selector)];
  const comparison = $('#comparison');
  const range = $('#comparison-range');
  function updateSplit() {
    comparison.style.setProperty('--split', `${range.value}%`);
    range.setAttribute('aria-valuetext', `${range.value}% input, ${100 - Number(range.value)}% ImIR output`);
  }
  range.addEventListener('input', updateSplit);
  updateSplit();

  // Retain decoded images and share pending requests between preloading and clicks.
  const imageCache = new Map();
  function loadImage(src) {
    if (imageCache.has(src)) return imageCache.get(src);
    const image = new Image();
    image.decoding = 'async';
    image.fetchPriority = 'low';
    const ready = new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error(`Could not load ${src}`));
    }).then(async () => {
      if (typeof image.decode === 'function') await image.decode();
      return image;
    }).catch(error => {
      imageCache.delete(src); // Allow a failed background request to retry on selection.
      throw error;
    });
    imageCache.set(src, ready);
    image.src = src;
    return ready;
  }
  let taskRequest = 0;
  $$('.task-list button').forEach(button => button.addEventListener('click', async () => {
    const request = ++taskRequest;
    const task = button.dataset.task;
    const item = content.tasks[task];
    comparison.setAttribute('aria-busy', 'true');
    try {
      await Promise.all([loadImage(`assets/${task}-input.webp`), loadImage(`assets/${task}-output.webp`)]);
      if (request !== taskRequest) return;
      $('#input-image').src = `assets/${task}-input.webp`;
      $('#output-image').src = `assets/${task}-output.webp`;
      $('#input-image').alt = `Degraded input: ${item.scene} with ${item.degradation}`;
      $('#output-image').alt = `ImIR ${item.label.toLowerCase()} output: ${item.scene}`;
      $$('.task-list button').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
      range.value = 50;
      updateSplit();
      $('#example-status').textContent = `${item.label} example loaded.`;
    } catch (_) {
      if (request === taskRequest) $('#example-status').textContent = 'This example could not load. Please try again.';
    } finally {
      if (request === taskRequest) comparison.setAttribute('aria-busy', 'false');
    }
  }));

  const scales = ['0', '0.25', '0.5', '0.75', '1', '1.5'];
  const scaleFiles = ['0', '025', '05', '075', '1', '15'];
  let controlTask = 'lowlight';
  const descriptions = {
    lowlight: 'Increasing the scale brightens the low-light image through a range of plausible exposures. Scaling beyond the default begins to over-brighten the output.',
    dehazing: 'Increasing the scale progressively removes haze and moves the output closer to the ground truth. Scaling beyond the default does not negatively impact this example.'
  };
  function updateControl() {
    const index = Number($('#scale-range').value);
    const scale = scales[index];
    $('#control-output').src = `assets/control-${controlTask}-${scaleFiles[index]}.webp`;
    $('#control-input').src = `assets/control-${controlTask}-input.webp`;
    $('#control-output').alt = `${controlTask === 'lowlight' ? 'Low-light' : 'Dehazing'} restoration at instruction scale ${scale}`;
    $('#control-input').alt = `Degraded ${controlTask === 'lowlight' ? 'low-light vegetation' : 'hazed car'} scene`;
    $('#scale-value').textContent = `s = ${scale}`;
    $('#control-caption').textContent = `s = ${scale}`;
    $('#scale-range').setAttribute('aria-valuetext', `Instruction scale ${scale}`);
    $('#control-description').textContent = descriptions[controlTask];
  }
  $('#scale-range').addEventListener('input', updateControl);
  $$('[data-control]').forEach(button => button.addEventListener('click', () => {
    controlTask = button.dataset.control;
    $$('[data-control]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    updateControl();
  }));
  // Let the initially visible pair load first, then warm every interactive image.
  // Background failures are retried by loadImage when that example is selected.
  Promise.all([
    loadImage($('#input-image').getAttribute('src')),
    loadImage($('#output-image').getAttribute('src'))
  ]).catch(() => {}).then(() => {
    const sources = Object.keys(content.tasks).flatMap(task => [
      `assets/${task}-input.webp`, `assets/${task}-output.webp`
    ]);
    ['lowlight', 'dehazing'].forEach(task => {
      ['input', ...scaleFiles].forEach(scale => sources.push(`assets/control-${task}-${scale}.webp`));
    });
    sources.forEach(src => { loadImage(src).catch(() => {}); });
  });

  const metricNames = { psnr: 'PSNR ↑', ssim: 'SSIM ↑', lpips: 'LPIPS ↓' };
  function updateTable(metric) {
    const body = $('#results-body');
    body.replaceChildren();
    content.results.forEach(item => {
      const row = document.createElement('tr');
      const heading = document.createElement('th');
      heading.scope = 'row'; heading.textContent = item.task;
      row.append(heading);
      const digits = metric === 'psnr' ? 1 : 2;
      item[metric].forEach((value, index) => {
        const cell = document.createElement('td'); cell.textContent = value.toFixed(digits);
        if (index === 4) cell.className = 'highlight-col';
        row.append(cell);
      });
      const delta = Number((item[metric][4] - item[metric][0]).toFixed(digits));
      const gain = document.createElement('td'); gain.className = 'gain';
      gain.textContent = `${delta > 0 ? '+' : delta < 0 ? '−' : ''}${Math.abs(delta).toFixed(digits)}${metric === 'psnr' ? ' dB' : ''}`;
      row.append(gain); body.append(row);
    });
    $('#comparison-table').setAttribute('aria-label', `Image instruction versus text: ${metricNames[metric]}`);
    $('#metric-status').textContent = `Table now shows ${metricNames[metric]}.`;
  }
  $$('[data-metric]').forEach(button => button.addEventListener('click', () => {
    $$('[data-metric]').forEach(b => b.setAttribute('aria-pressed', String(b === button)));
    updateTable(button.dataset.metric);
  }));

  const resources = $('#resources');
  resources.replaceChildren();
  content.links.forEach(item => {
    const active = typeof item.url === 'string' && /^https:\/\//.test(item.url);
    const link = document.createElement(active ? 'a' : 'span');
    link.className = `resource${active ? '' : ' unavailable'}`;
    link.textContent = item.label;
    if (active) { link.href = item.url; link.target = '_blank'; link.rel = 'noopener noreferrer'; }
    else { const note = document.createElement('small'); note.textContent = 'Coming soon'; link.append(note); }
    resources.append(link);
  });
  if (content.bibtex) {
    $('#bibtex').textContent = content.bibtex;
    $('#citation-state').hidden = true;
    $('#copy-citation').hidden = false;
    $('#copy-citation').addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(content.bibtex);
        $('#copy-status').textContent = 'Citation copied.';
        $('#copy-citation').textContent = 'Copied';
        setTimeout(() => { $('#copy-citation').textContent = 'Copy citation'; }, 2000);
      } catch (_) { $('#copy-status').textContent = 'Copy was unavailable. Select the citation text to copy it manually.'; }
    });
  }
})();
