"""Single-file HTML report: summary, mate reactions and a 3D viewer (three.js from jsDelivr)."""
from __future__ import annotations

import html
import json
from pathlib import Path

TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stress Plan Report</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e2dc;--acc:#0f766e;--bad:#b42318;--ok:#15803d}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--fg:#ecece8;--mut:#a3a39c;--card:#1e1e1c;--line:#33332f;--acc:#2dd4bf;--bad:#f97066;--ok:#4ade80}}
:root[data-theme="dark"]{--bg:#141413;--fg:#ecece8;--mut:#a3a39c;--card:#1e1e1c;--line:#33332f;--acc:#2dd4bf;--bad:#f97066;--ok:#4ade80}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1200px;margin:0 auto;padding:20px 16px 60px}h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.mut{color:var(--mut)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-weight:600;color:var(--mut);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.sf{font-weight:700}.lo{color:var(--bad)}.hi{color:var(--ok)}
#viewer{position:relative;height:min(70vh,620px);border-radius:10px;overflow:hidden;border:1px solid var(--line);background:var(--card)}
#ctl{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}select,button,label{font:inherit;color:var(--fg);background:var(--card);border:1px solid var(--line);border-radius:7px;padding:5px 9px}
button.on{border-color:var(--acc);color:var(--acc)}#legend{position:absolute;left:10px;bottom:10px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 9px;font-size:12px}
#bar{width:180px;height:10px;border-radius:3px;background:linear-gradient(90deg,#d7301f,#fc8d59,#fee08b,#91cf60,#1a9850)}
.note{font-size:13px;color:var(--mut);margin-top:8px}
</style></head><body><main>
<h1>Stress plan: __ASM__</h1>
<div class="mut">Load case <b>__LC__</b> · material __MAT__ · load factor __LF__ · __SECS__ s</div>
<h2>Parts</h2><div class="card"><table id="parts"></table>
<div class="note">Safety factor = 1 / failure index, volume-weighted 99th percentile with load-patch singularities excluded. Interlayer strength is the weak direction; the plan keeps principal tension inside the layers. Max flex is the largest deformation relative to the part's own mean position (each part floats on its mates). Values come from a linear tet FEA and a simple anisotropic failure model: use them to compare orientations, not as a certified rating.</div></div>
<h2>3D view</h2>
<div id="ctl"><select id="part"></select><select id="mode"><option value="sf">Colour: safety factor</option><option value="seg">Colour: segment</option></select>
<button id="gly" class="on">Stress lines</button><button id="pl" class="on">Cut planes</button></div>
<div id="viewer"><div id="legend"><div id="bar"></div><div style="display:flex;justify-content:space-between"><span>SF ≤ 1</span><span>≥ 4</span></div>
<div style="margin-top:4px">lines: <span style="color:#e4572e">tension</span> / <span style="color:#3b82f6">compression</span> (σ₁ direction)</div></div></div>
<div class="note">Print frame as written to stress_plan.3mf: +Z is the root build direction, every cut plane builds along its normal.</div>
<h2>Mate reactions</h2><div class="card"><table id="reac"></table>
<div class="note">__INDET__</div></div>
</main>
<script>const DATA=__DATA__;const VIZ=__VIZ__;</script>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const f=(x,n=2)=>Number.isFinite(x)?x.toFixed(n):'–';const cls=s=>s<1.5?'lo':(s>=3?'hi':'');
const pt=document.getElementById('parts');
pt.innerHTML='<tr><th>Part</th><th>SF as modelled</th><th>SF best 3-axis</th><th>SF multiplanar</th><th>Segments</th><th>Support area</th><th>σvm p99</th><th>Max flex</th></tr>'+
DATA.parts.map(p=>`<tr><td>${p.part}</td><td class="sf ${cls(p.sf_as_modelled)}">${f(p.sf_as_modelled)}</td><td class="sf ${cls(p.sf_best_3axis)}">${f(p.sf_best_3axis)}</td><td class="sf ${cls(p.sf_plan)}">${f(p.sf_plan)}</td><td>${p.n_segments}</td><td>${f(100*p.support_area_frac,0)} %</td><td>${f(p.von_mises_p99)} MPa</td><td>${f(p.max_disp_mm,3)} mm</td></tr>`).join('');
document.getElementById('reac').innerHTML='<tr><th>Connection</th><th>From</th><th>On</th><th>|F| N</th><th>|M| N·mm</th></tr>'+
DATA.statics.reactions.slice().sort((a,b)=>b.force_N-a.force_N).map(r=>`<tr><td>${r.connection}</td><td>${r.part_a}</td><td>${r.part_b}</td><td>${f(r.force_N,1)}</td><td>${f(r.moment_Nmm,0)}</td></tr>`).join('');
const el=document.getElementById('viewer');const R=new THREE.WebGLRenderer({antialias:true});R.setPixelRatio(devicePixelRatio);el.appendChild(R.domElement);
const scene=new THREE.Scene();const bgc=()=>{scene.background=new THREE.Color(getComputedStyle(document.documentElement).getPropertyValue('--card').trim()||'#ffffff')};bgc();matchMedia('(prefers-color-scheme: dark)').addEventListener('change',bgc);const cam=new THREE.PerspectiveCamera(40,1,0.1,1e5);cam.up.set(0,0,1);const ctl=new OrbitControls(cam,R.domElement);
scene.add(new THREE.HemisphereLight(0xffffff,0x444444,1.6));const dl=new THREE.DirectionalLight(0xffffff,1.2);dl.position.set(1,-1.5,2);scene.add(dl);
let group=null,gly=true,pls=true;
const ramp=[[215,48,31],[252,141,89],[254,224,139],[145,207,96],[26,152,80]];
function sfColor(s){let t=Math.min(Math.max((s-1)/3,0),1)*(ramp.length-1);const i=Math.min(Math.floor(t),ramp.length-2),u=t-i;return ramp[i].map((c,k)=>(c+(ramp[i+1][k]-c)*u)/255);}
const segC=[[.62,.64,.66],[.18,.55,.85],[.95,.55,.15],[.55,.35,.85],[.2,.75,.5],[.9,.3,.5]];
function build(){if(group){scene.remove(group);group.traverse(o=>{o.geometry&&o.geometry.dispose()})}
 const v=VIZ[document.getElementById('part').value];const mode=document.getElementById('mode').value;group=new THREE.Group();
 const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(v.vertices,3));g.setIndex(v.faces);
 const col=new Float32Array(v.vertices.length);for(let i=0;i<v.sf.length;i++){const c=mode==='sf'?sfColor(v.sf[i]):segC[v.seg[i]%segC.length];col.set(c,3*i)}
 g.setAttribute('color',new THREE.BufferAttribute(col,3));g.computeVertexNormals();
 group.add(new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.7,transparent:gly,opacity:gly?.55:1,side:THREE.DoubleSide,depthWrite:!gly})));
 if(gly){const P=[],C=[];const L=v.glyph_len/2;for(let i=0;i<v.glyph_sign.length;i++){const p=v.glyph_pos.slice(3*i,3*i+3),d=v.glyph_dir.slice(3*i,3*i+3);
  P.push(p[0]-d[0]*L,p[1]-d[1]*L,p[2]-d[2]*L,p[0]+d[0]*L,p[1]+d[1]*L,p[2]+d[2]*L);const c=v.glyph_sign[i]>0?[.89,.34,.18]:[.23,.51,.96];C.push(...c,...c)}
  const lg=new THREE.BufferGeometry();lg.setAttribute('position',new THREE.Float32BufferAttribute(P,3));lg.setAttribute('color',new THREE.Float32BufferAttribute(C,3));
  group.add(new THREE.LineSegments(lg,new THREE.LineBasicMaterial({vertexColors:true})));}
 g.computeBoundingSphere();const rad=g.boundingSphere.radius;
 if(pls)v.planes.forEach((p,i)=>{const m=new THREE.Mesh(new THREE.PlaneGeometry(rad*1.6,rad*1.6),new THREE.MeshBasicMaterial({color:new THREE.Color(...segC[(i+1)%segC.length]),transparent:true,opacity:.28,side:THREE.DoubleSide,depthWrite:false}));
  m.position.set(...p.point);m.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),new THREE.Vector3(...p.normal));group.add(m);
  const a=new THREE.ArrowHelper(new THREE.Vector3(...p.normal),new THREE.Vector3(...p.point),rad*.5,new THREE.Color(...segC[(i+1)%segC.length]));group.add(a)});
 const bed=new THREE.GridHelper(rad*3,12,0x888888,0x888888);bed.rotation.x=Math.PI/2;bed.position.z=Math.min(...v.vertices.filter((_,i)=>i%3==2));bed.material.opacity=.3;bed.material.transparent=true;group.add(bed);
 scene.add(group);cam.position.set(rad*1.8,-rad*2.4,rad*1.6);ctl.target.set(0,0,0);ctl.update();}
const sel=document.getElementById('part');VIZ.forEach((v,i)=>sel.add(new Option(v.name,i)));
sel.onchange=build;document.getElementById('mode').onchange=build;
document.getElementById('gly').onclick=e=>{gly=!gly;e.target.classList.toggle('on',gly);build()};
document.getElementById('pl').onclick=e=>{pls=!pls;e.target.classList.toggle('on',pls);build()};
function size(){const w=el.clientWidth,h=el.clientHeight;R.setSize(w,h);cam.aspect=w/h;cam.updateProjectionMatrix()}
new ResizeObserver(size).observe(el);size();if(VIZ.length)build();
(function loop(){requestAnimationFrame(loop);R.render(scene,cam)})();
</script></body></html>
"""


def write_report(path, result: dict, viz: list[dict]) -> Path:
    mat = result["material"]
    indet = ("The assembly is statically indeterminate (redundant mates): reactions are the "
             "minimum-norm split, not a stiffness-based one." if result["statics"]["indeterminate"]
             else "Statically determinate: reactions follow from equilibrium alone.")
    if result["statics"]["unbalanced"]:
        indet += " Could not balance: " + ", ".join(result["statics"]["unbalanced"]) + "."
    s = (TEMPLATE.replace("__ASM__", html.escape(result["assembly"]))
         .replace("__LC__", html.escape(result["load_case"]))
         .replace("__MAT__", html.escape(f"{mat['name']} (E {mat['E']:.0f} MPa, S_xy {mat['S_xy']:.0f}, "
                                         f"S_z {mat['S_z']:.0f}, τ_z {mat['S_shear_z']:.0f} MPa)"))
         .replace("__LF__", f"{result['load_factor']:g}").replace("__SECS__", str(result["seconds"]))
         .replace("__INDET__", html.escape(indet))
         .replace("__DATA__", json.dumps(result).replace("</", "<\\/"))
         .replace("__VIZ__", json.dumps(viz).replace("</", "<\\/")))
    Path(path).write_text(s)
    return Path(path)
