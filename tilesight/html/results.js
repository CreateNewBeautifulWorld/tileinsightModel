// Shared results-rendering, used by app.html (an ephemeral wizard run) and regr_result.html (a
// persisted regression job's results page) — kept in one file so the two never drift on the
// JSON shape cli/report_json.py's build_report_json() returns.
//
// Each page defines its own `API`, sets `JOB` to whatever id it downloads with, and sets `DL` —
// download-link functions the two pages fill in differently (app.html regenerates a trace/memmap
// file on demand from its ephemeral job cache; regr_result.html links straight to a file the
// regression worker already wrote into that job's own out directory):
//   DL.xlsx(phase, full) -> url | falsy     DL.csv(phase, full) -> url | falsy
//   DL.pdf(phase)        -> url | falsy     DL.memmap(phase, fmt) -> url | falsy   (fmt: xlsx|csv)
// A falsy return just hides that button — a page that didn't generate a PDF, say, need not fake one.
const $=id=>document.getElementById(id);
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const API=location.origin.startsWith("http")?location.origin:"http://127.0.0.1:8000";
let JOB=null, DL={};

async function downloadFile(btn,url,filename){
  // window.open(url) gives no feedback while the server is still generating a large trace file
  // (the tab just sits blank) — fetch it instead so the button itself can show progress.
  const label=btn.textContent;
  btn.disabled=true; btn.textContent="Generating…";
  try{
    const r=await fetch(url);
    if(!r.ok){ const e=await r.json().catch(()=>({error:r.statusText})); throw new Error(e.error||r.statusText); }
    const blob=await r.blob();
    const a=document.createElement("a");
    a.href=URL.createObjectURL(blob); a.download=filename; a.click();
    URL.revokeObjectURL(a.href);
  }catch(e){
    btn.textContent="failed: "+e.message; await new Promise(s=>setTimeout(s,2500));
  }finally{
    btn.disabled=false; btn.textContent=label;
  }
}
function bars(list){ return list.map(x=>`<tr><td class="mono">${x.name}</td>
  <td class="num">${(100*x.share).toFixed(1)}%</td></tr>`).join(""); }

// --- the steady-state timeline (Gantt), same data/colors/grouping as the PDF report's, drawn
// inline instead of only inside a downloaded PDF — see gpuTilingPerfHWModel/genResult/pdfreport.py
// (_Gantt, LANE_RGB, unit_groups) for the source of truth this mirrors.
const LANE_RGB={switch:"#6f8f6a",tc:"#b4552d",cuda:"#6b8f9c",sfu:"#8a6fb0",smem:"#5b9279",
  tmem:"#c2903a",l1:"#2f7d54",l2:"#4f7cac",ddr:"#b3563a",dma:"#9c4a6b",l2port:"#3f8f8f",sram:"#7f9a52",net:"#7a7a7a"};
function unitGroups(lanes){
  // memory-to-compute order (HBM -> L2 -> buffer -> load paths -> L1 -> cores) — a tile is
  // loaded before an MMA consumes it; mirrors gpuTilingPerfHWModel/genResult/pdfreport.py's
  // unit_groups() so the html view, the PDF and the Excel export all read the same way.
  const g={"memory · HBM":[],"memory · L2 ports":[],"on-chip buffer":[],
           "shader slice · load paths":[],"shader slice · L1/scratchpad":[],"shader slice · cores":[]};
  for(const l of lanes){
    if(["tc","cuda","sfu"].includes(l)) g["shader slice · cores"].push(l);
    else if(["smem","tmem","l1"].includes(l)) g["shader slice · L1/scratchpad"].push(l);
    else if(["sram","switch"].includes(l)) g["on-chip buffer"].push(l);
    else if(l.startsWith("path")) g["shader slice · load paths"].push(l);
    else if(l==="l2") g["memory · L2 ports"].push(l);
    else g["memory · HBM"].push(l);
  }
  return g;
}
function ganttSvg(tl,widthPx){
  if(!tl||!tl.steady||!tl.steady.length) return `<div class="hint">no timeline for this kernel</div>`;
  const groups=unitGroups(tl.lanes||[]);
  const rowH=15,left=100,right=widthPx-10,top=6;
  const span=Math.max(1e-12,...tl.steady.map(i=>i.end));
  let y=top,parts=[];
  for(const [gname,lanes] of Object.entries(groups)){
    if(!lanes.length) continue;
    parts.push(`<text x="2" y="${y+10}" font-size="9" fill="currentColor" opacity=".55">${esc(gname.toUpperCase())}</text>`);
    y+=rowH;
    for(const lane of lanes){
      parts.push(`<text x="${left-6}" y="${y+9}" font-size="9" text-anchor="end" fill="currentColor">${esc(lane)}</text>`);
      parts.push(`<rect x="${left}" y="${y}" width="${right-left}" height="10" fill="currentColor" opacity=".08"/>`);
      for(const it of tl.steady){
        const v=it.lanes[lane]; if(!v) continue;
        const x0=left+(right-left)*it.start/span, w=Math.max(0.6,(right-left)*v/span);
        parts.push(`<rect x="${x0.toFixed(1)}" y="${y}" width="${w.toFixed(1)}" height="10" `+
          `fill="${LANE_RGB[lane]||'#9a8f7a'}"><title>${esc(lane)}: ${esc(it.action)} `+
          `${(v*1e9).toFixed(1)} ns (round ${it.round}${it.iteration!==undefined?`, iter ${it.iteration}`:""})</title></rect>`);
      }
      y+=rowH;
    }
  }
  const h=y+16;
  parts.push(`<line x1="${left}" y1="${top}" x2="${left}" y2="${h-14}" stroke="currentColor" opacity=".15"/>`);
  parts.push(`<text x="${left}" y="${h-2}" font-size="9" fill="currentColor" opacity=".6">0</text>`);
  parts.push(`<text x="${right}" y="${h-2}" font-size="9" text-anchor="end" fill="currentColor" opacity=".6">`+
    `${(span*1e9).toFixed(0)} ns / ${(span*tl.clock_hz).toFixed(0)} cycles · ${tl.rounds_shown} round(s) shown</text>`);
  return `<svg width="${widthPx}" height="${h}" viewBox="0 0 ${widthPx} ${h}" style="display:block">${parts.join("")}</svg>`;
}
function updateGantt(){
  const r=PHASE_RESULTS&&PHASE_RESULTS[CUR_PHASE]; if(!r) return;
  const zoom=parseFloat($("ganttZoom").value);
  $("ganttZoomLabel").textContent=zoom.toFixed(1)+"x";
  $("ganttWrap").innerHTML=ganttSvg(r.timeline,Math.round(820*zoom));
}

const MM_KIND={weight:"#4f7cac",kv:"#b3563a",act:"#7f9a52",workspace:"#c2903a"};
function memmapHtml(mm, phase){
  if(!mm) return "";
  if(mm.error) return `<div class="sec"><h2>HBM address map</h2><div class="hint">not available: ${esc(mm.error)}</div></div>`;
  const W=820, span=Math.max(1,mm.span), GB=x=>(x/1e9).toFixed(2);
  const strip=mm.rows.map(r=>`<rect x="${(W*r.offset/span).toFixed(2)}" y="0" width="${Math.max(0.6,W*r.size/span).toFixed(2)}" height="26" fill="${MM_KIND[r.kind]||"#999"}"><title>${esc(r.name)} · ${r.base}–${r.end} · ${(r.size/1e6).toFixed(1)} MB</title></rect>`).join("");
  const ticks=[0,.25,.5,.75,1].map(f=>`<text x="${Math.min(W-2,Math.max(2,f*W))}" y="40" font-size="10" fill="var(--muted)" text-anchor="${f===0?"start":f===1?"end":"middle"}">0x${(mm.base+Math.round(f*span)).toString(16)}</text>`).join("");
  const kinds=Object.keys(mm.per_port_by_kind), P=mm.hbm_ports;
  const portTot=[...Array(P).keys()].map(p=>kinds.reduce((a,k)=>a+mm.per_port_by_kind[k][p],0));
  const pmax=Math.max(1,...portTot), bw=Math.min(60,(W-40)/P), H=120;
  const ports=[...Array(P).keys()].map(p=>{let y=H; return kinds.map(k=>{const h=H*mm.per_port_by_kind[k][p]/pmax; y-=h;
      return `<rect x="${30+p*bw+2}" y="${y.toFixed(1)}" width="${bw-4}" height="${h.toFixed(1)}" fill="${MM_KIND[k]||"#999"}"><title>port${p} · ${k} ${GB(mm.per_port_by_kind[k][p])} GB</title></rect>`;}).join("")
      +`<text x="${30+p*bw+bw/2}" y="${H+12}" font-size="10" fill="var(--muted)" text-anchor="middle">${p}</text>`;}).join("");
  const legend=Object.entries(mm.totals).map(([k,v])=>`<span style="display:inline-flex;align-items:center;gap:5px;margin-right:14px"><i style="width:10px;height:10px;background:${MM_KIND[k]||"#999"};display:inline-block;border-radius:2px"></i>${k} ${GB(v)} GB</span>`).join("");
  const xlsxUrl=DL.memmap&&DL.memmap(phase,"xlsx"), csvUrl=DL.memmap&&DL.memmap(phase,"csv");
  return `<div class="sec"><h2>HBM address map — where every tensor lives</h2>
    <div class="hint">each tensor is one contiguous region, 2 MB-aligned, allocated in execution order from
      <span class="mono">${mm.base_hex}</span> to <span class="mono">${mm.end_hex}</span>
      (${GB(mm.span)} GB${mm.capacity?` of ${GB(mm.capacity)} GB HBM`:""}, ${mm.regions} regions).
      ${esc(mm.hbm_map)}; ${esc(mm.l2_map)}.</div>
    <div style="margin:8px 0">${legend}</div>
    <svg viewBox="0 0 ${W} 44" width="100%" style="max-width:${W}px">${strip}${ticks}</svg>
    <div class="hint" style="margin-top:8px">bytes per HBM port, by tensor kind (the interleave spreads every region over all ports)</div>
    <svg viewBox="0 0 ${W} ${H+16}" width="100%" style="max-width:${W}px">
      <text x="0" y="10" font-size="10" fill="var(--muted)">${GB(pmax)} GB</text>${ports}</svg>
    <div class="row" style="margin:8px 0">
      <input id="mmFilter" placeholder="filter regions (e.g. kv, experts, layer 3)" style="max-width:320px" oninput="memmapTable()">
      ${xlsxUrl?`<button class="ghost" onclick="downloadFile(this,'${xlsxUrl}','memmap.xlsx')">address map Excel (with charts)</button>`:""}
      ${csvUrl?`<button class="ghost" onclick="downloadFile(this,'${csvUrl}','memmap.csv')">CSV</button>`:""}
    </div>
    <div class="scroll" id="mmTable"></div>
  </div>`;
}
function memmapTable(){
  const el=$("mmTable"); if(!el) return;
  const mm=(PHASE_RESULTS[CUR_PHASE]||{}).memmap; if(!mm||!mm.rows) return;
  const q=(($("mmFilter")||{}).value||"").toLowerCase();
  const rows=mm.rows.filter(r=>!q||(r.name+" "+r.kind).toLowerCase().includes(q));
  el.innerHTML=`<table><thead><tr><th>region</th><th>kind</th><th>base</th><th>end</th><th class="num">size</th>
    <th>shape</th><th class="num">HBM imbalance</th></tr></thead><tbody>${rows.slice(0,80).map(r=>
    `<tr><td class="mono">${esc(r.name)}</td><td><i style="width:9px;height:9px;background:${MM_KIND[r.kind]||"#999"};display:inline-block;border-radius:2px"></i> ${r.kind}</td>
    <td class="mono">${r.base}</td><td class="mono">${r.end}</td><td class="num">${(r.size/1e6).toFixed(1)} MB</td>
    <td class="mono">${esc(r.shape)}</td><td class="num">${r.imb.toFixed(3)}</td></tr>`).join("")}</tbody></table>
    <div class="hint">${rows.length>80?`showing 80 of ${rows.length} matching regions — the Excel/CSV has all of them`:`${rows.length} regions`}${mm.truncated?` (+${mm.truncated} not sent to the page)`:""}</div>`;
}
let PHASE_RESULTS=null, CUR_PHASE="decode", LAST_SECS=0;
function showResults(r,secs){
  // mode "workload" (single phase) returns one flat result; mode "workload_both" returns
  // {prefill:{...}, decode:{...}} already — normalize to the latter shape either way.
  const both = ("step_ms" in r) ? {[r.workload?.run?.phase||"result"]: r} : r;
  PHASE_RESULTS=both; LAST_SECS=secs;
  CUR_PHASE=both.decode?"decode":Object.keys(both)[0];
  renderResultsShell();
}
function switchPhase(p){ CUR_PHASE=p; renderResultsShell(); }
function renderResultsShell(){
  const phases=Object.keys(PHASE_RESULTS);
  const tabs=phases.map(p=>`<button class="ghost" `+
    `style="border-color:${p===CUR_PHASE?'var(--accent)':'var(--line)'};font-weight:${p===CUR_PHASE?700:400};`+
    `text-transform:capitalize" onclick="switchPhase('${p}')">${esc(p)}`+
    `${p==="prefill"?" (processing the prompt)":p==="decode"?" (one token-generation step)":""}</button>`).join(" ");
  $("results").innerHTML=(phases.length>1?`<div style="display:flex;gap:8px;margin-bottom:14px">${tabs}</div>`:"")
    + `<div id="phasebody"></div>`;
  renderPhaseBody(PHASE_RESULTS[CUR_PHASE]);
}
function renderPhaseBody(r){
  const m=r.memory, secs=LAST_SECS;
  const xlsxUrl=DL.xlsx&&DL.xlsx(CUR_PHASE,false), xlsxFullUrl=DL.xlsx&&DL.xlsx(CUR_PHASE,true),
        csvUrl=DL.csv&&DL.csv(CUR_PHASE,false), pdfUrl=DL.pdf&&DL.pdf(CUR_PHASE);
  $("phasebody").innerHTML=`
  <div class="kpis">
    <div class="kpi"><b>${r.step_ms.toFixed(2)} ms</b><span>per layer stack</span></div>
    <div class="kpi"><b>${m.total_GB.toFixed(1)} GB</b><span>memory / GPU</span></div>
    <div class="kpi"><b>${r.bounds.by_resource[0]?.name||"—"}</b><span>top bound</span></div>
    <div class="kpi"><b>${secs.toFixed(1)} s</b><span>compute time</span></div></div>
  ${r.sim_window&&r.sim_window.kernels?`<div class="hint" style="margin:-4px 0 12px">cache simulation replayed
    ${(100*r.sim_window.weighted).toFixed(1)}% of all tile accesses (time-weighted);
    ${r.sim_window.full}/${r.sim_window.kernels} kernels in full${r.sim_window.min<0.999?`, least:
    ${(100*r.sim_window.min).toFixed(1)}% of <span class="mono">${esc(r.sim_window.min_op)}</span>
    — widen the window for more`:""}.</div>`:""}
  <div class="row" style="margin:0 0 14px">
    ${xlsxUrl?`<button class="ghost" onclick="downloadFile(this,'${xlsxUrl}','tilesight_cycles.xlsx')">trace Excel (per cycle, grouped by unit)</button>`:""}
    ${xlsxFullUrl&&xlsxFullUrl!==xlsxUrl?`<button class="ghost" onclick="downloadFile(this,'${xlsxFullUrl}','tilesight_cycles.xlsx')">Excel · whole kernel</button>`:""}
    ${csvUrl?`<button class="ghost" onclick="downloadFile(this,'${csvUrl}','tilesight_cycles.csv')">CSV</button>`:""}
    ${pdfUrl?`<button class="ghost" onclick="downloadFile(this,'${pdfUrl}','tilesight_report.pdf')">PDF report</button>`:""}
  </div>
  ${r.by_block?`<div class="sec"><h2>the model's three blocks</h2>
    <div class="kpis">${Object.entries(r.by_block).map(([k,v])=>
      `<div class="kpi"><b>${(100*v).toFixed(0)}%</b><span>${k.replace(/_/g," ")} — of the critical time</span></div>`).join("")}</div>
    ${r.activity_by_block?`<div class="hint">activity (busy, bottleneck or not): ${
      Object.entries(r.activity_by_block).map(([k,v])=>`${k.replace(/_/g," ")} ${(100*v).toFixed(0)}%`).join(" · ")}</div>`:""}</div>`:""}
  ${r.attn_compare?`<div class="sec"><h2>flash attention on / off</h2>
    <div class="scroll"><table><thead><tr><th></th><th class="num">step</th><th class="num">attention</th>
    <th class="num">activations</th><th>top bound</th></tr></thead><tbody>
    ${["flash","naive"].map(k=>`<tr><td>${k}</td><td class="num">${r.attn_compare[k].step_ms.toFixed(2)} ms</td>
      <td class="num">${r.attn_compare[k].attention_ms.toFixed(2)} ms</td>
      <td class="num">${r.attn_compare[k].activations_GB.toFixed(2)} GB</td>
      <td class="mono">${r.attn_compare[k].top_bound}</td></tr>`).join("")}
    </tbody></table></div>
    <div class="hint">flash is ${r.attn_compare.speedup_of_flash.toFixed(2)}x faster here and saves
      ${r.attn_compare.extra_activation_GB_without_flash.toFixed(1)} GB of activations.</div></div>`:""}
  ${memmapHtml(r.memmap, CUR_PHASE)}
  <h2>GPU architecture (as configured)</h2>
  <div class="scroll">${r.arch_svg||""}</div>
  <div class="sec"><h2>where the time goes</h2>
    <div class="scroll"><table><thead><tr><th>by resource</th><th class="num">share</th></tr></thead>
    <tbody>${bars(r.bounds.by_resource)}</tbody></table></div></div>
  <div class="sec"><h2>by resource and tensor</h2>
    <div class="scroll"><table><thead><tr><th>resource:tensor</th><th class="num">share</th></tr></thead>
    <tbody>${bars(r.bounds.by_tensor.slice(0,10))}</tbody></table></div></div>
  ${r.timeline?`<div class="sec"><h2>steady-state timeline${r.trace_kernel?` — ${esc(r.trace_kernel)}`:""}</h2>
    <div class="hint">the heaviest kernel's steady-state rounds, lane by lane (same picture as the PDF report).
      Drag to stretch it out and see one round in detail.</div>
    <div style="display:flex;align-items:center;gap:8px;margin:8px 0">
      <input type="range" id="ganttZoom" min="1" max="8" step="0.5" value="2" style="width:200px" oninput="updateGantt()">
      <span id="ganttZoomLabel" class="hint">2.0x</span>
    </div>
    <div id="ganttWrap" style="overflow-x:auto;border:1px solid var(--line);border-radius:8px;padding:6px 0"></div>
  </div>`:""}
  <div class="sec"><h2>operators</h2><div class="scroll"><table>
    <thead><tr><th>op</th><th class="num">x</th><th>tile</th><th class="num">time</th>
    <th class="num">share</th><th>occupancy</th><th>bound</th></tr></thead><tbody>
    ${r.ops.slice(0,25).map(o=>`<tr><td class="mono">${o.name}</td><td class="num">${o.repeat}</td>
      <td class="mono">${o.tile}</td><td class="num">${o.time_us.toFixed(1)} µs</td>
      <td class="num">${(100*o.share).toFixed(1)}%</td><td class="mono">${o.occupancy}</td>
      <td class="mono">${o.bound}</td></tr>`).join("")}</tbody></table></div></div>
  <details class="sec"><summary>text report</summary><pre class="mono">${r.summary}</pre></details>
  ${RESULTS_FOOTER?RESULTS_FOOTER():""}`;
  if(r.timeline) updateGantt();
  if(r.memmap) memmapTable();
}
// Each page owns its own "what next" row (app.html: change workload/GPU and re-run in place;
// regr_result.html: back to the dashboard) — set before calling showResults().
let RESULTS_FOOTER=null;
