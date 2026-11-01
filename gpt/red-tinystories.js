// GPT a la vista: TinyStories-8M (GPT-Neo) corriendo en el navegador, sin bibliotecas.
// Tokenizador BPE de bytes (el de GPT-2) y la red completa, con caché de claves y valores.

/* ---------------- tokenizador ---------------- */
const Tokenizador = (() => {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice(); let n = 0;
  for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n); n++; }
  const b2u = new Map(bs.map((b, i) => [b, String.fromCharCode(cs[i])]));
  const u2b = new Map(bs.map((b, i) => [String.fromCharCode(cs[i]), b]));
  const PAT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
  const enc = new TextEncoder(), dec = new TextDecoder('utf-8');
  let vocab, inv, rank; const cache = new Map();
  function cargar(vocabJson, mergesTxt) {
    vocab = vocabJson; inv = []; for (const [t, i] of Object.entries(vocab)) inv[i] = t;
    rank = new Map();
    mergesTxt.split('\n').slice(1).forEach((l, r) => { if (l.trim()) rank.set(l, r); });
  }
  function bpe(w) {
    if (cache.has(w)) return cache.get(w);
    let s = [...w];
    while (s.length > 1) {
      let best = Infinity, bi = -1;
      for (let i = 0; i < s.length - 1; i++) { const r = rank.get(s[i] + ' ' + s[i + 1]); if (r !== undefined && r < best) { best = r; bi = i; } }
      if (bi < 0) break;
      s = [...s.slice(0, bi), s[bi] + s[bi + 1], ...s.slice(bi + 2)];
    }
    const ids = s.map(t => vocab[t]); cache.set(w, ids); return ids;
  }
  function encode(texto) {
    const out = [];
    for (const m of texto.matchAll(PAT)) out.push(...bpe([...enc.encode(m[0])].map(b => b2u.get(b)).join('')));
    return out;
  }
  function decode(ids) { return dec.decode(new Uint8Array(ids.flatMap(i => [...inv[i]].map(ch => u2b.get(ch))))); }
  return { cargar, encode, decode, size: () => inv.length };
})();

/* ---------------- la red ---------------- */
function half(h) {
  const s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}
const Red = { meta: null, P: {}, L: [] };
function armarPesos(meta, buffers) {
  const total = buffers.reduce((a, b) => a + b.byteLength, 0), u8 = new Uint8Array(total);
  let o = 0; for (const b of buffers) { u8.set(new Uint8Array(b), o); o += b.byteLength; }
  const u16 = new Uint16Array(u8.buffer);
  const tabla = new Float32Array(65536); for (let i = 0; i < 65536; i++) tabla[i] = half(i);
  let p = 0;
  for (const { k, shape } of meta.orden) {
    const n = shape.reduce((a, b) => a * b, 1), a = new Float32Array(n);
    for (let i = 0; i < n; i++) a[i] = tabla[u16[p + i]];
    Red.P[k] = a; p += n;
  }
  Red.meta = meta;
  Red.L = [...Array(meta.capas)].map((_, l) => {
    const g = n => Red.P[`h.${l}.${n}`];
    return { ln1w: g('ln_1.weight'), ln1b: g('ln_1.bias'), qw: g('attn.q_proj.weight'), kw: g('attn.k_proj.weight'), vw: g('attn.v_proj.weight'),
      ow: g('attn.out_proj.weight'), ob: g('attn.out_proj.bias'), ln2w: g('ln_2.weight'), ln2b: g('ln_2.bias'),
      fw: g('mlp.c_fc.weight'), fb: g('mlp.c_fc.bias'), pw: g('mlp.c_proj.weight'), pb: g('mlp.c_proj.bias') };
  });
}
function lnorm(x, w, b, eps) {
  const D = x.length; let m = 0; for (let i = 0; i < D; i++) m += x[i]; m /= D;
  let v = 0; for (let i = 0; i < D; i++) v += (x[i] - m) ** 2; v /= D;
  const r = 1 / Math.sqrt(v + eps), y = new Float32Array(D); for (let i = 0; i < D; i++) y[i] = (x[i] - m) * r * w[i] + b[i]; return y;
}
function lin(x, W, b, nout) {
  const nin = x.length, y = new Float32Array(nout);
  for (let o = 0; o < nout; o++) { let s = b ? b[o] : 0; const off = o * nin; for (let i = 0; i < nin; i++) s += W[off + i] * x[i]; y[o] = s; }
  return y;
}
const gelu = x => 0.5 * x * (1 + Math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
function nuevaCache(registrar) { const NL = Red.meta.capas; return { K: [...Array(NL)].map(() => []), V: [...Array(NL)].map(() => []), n: 0, reg: registrar ? { emb: [], capas: [...Array(NL)].map(() => []) } : null }; }
// procesa un token; guarda los puntajes de atención (antes de softmax) de este token en cada capa y cabeza
function paso(id, c, conLogits) {
  const { dim: D, cabezas: NH, capas: NL, eps } = Red.meta, DH = D / NH, t = c.n;
  const te = Red.P['wte.weight'], pe = Red.P['wpe.weight'], x = new Float32Array(D);
  for (let i = 0; i < D; i++) x[i] = te[id * D + i] + pe[t * D + i];
  if (c.reg) c.reg.emb.push(x.slice());
  const punt = [];
  for (let l = 0; l < NL; l++) {
    const B = Red.L[l], h = lnorm(x, B.ln1w, B.ln1b, eps);
    const q = lin(h, B.qw, null, D), k = lin(h, B.kw, null, D), v = lin(h, B.vw, null, D);
    const r = c.reg ? { q, k, v } : null;
    c.K[l].push(k); c.V[l].push(v);
    const y = new Float32Array(D), pl = [];
    for (let hd = 0; hd < NH; hd++) {
      const sc = new Float32Array(t + 1); let mx = -Infinity;
      for (let j = 0; j <= t; j++) { let s = 0; const kj = c.K[l][j]; for (let d = 0; d < DH; d++) s += q[hd * DH + d] * kj[hd * DH + d]; sc[j] = s; if (s > mx) mx = s; }
      pl.push(sc.slice());
      let se = 0; for (let j = 0; j <= t; j++) { sc[j] = Math.exp(sc[j] - mx); se += sc[j]; }
      for (let j = 0; j <= t; j++) { const w = sc[j] / se, vj = c.V[l][j]; for (let d = 0; d < DH; d++) y[hd * DH + d] += w * vj[hd * DH + d]; }
    }
    punt.push(pl);
    const ao = lin(y, B.ow, B.ob, D); for (let i = 0; i < D; i++) x[i] += ao[i];
    if (r) r.medio = x.slice();
    const f = lin(lnorm(x, B.ln2w, B.ln2b, eps), B.fw, B.fb, 4 * D); for (let i = 0; i < f.length; i++) f[i] = gelu(f[i]);
    const mo = lin(f, B.pw, B.pb, D); for (let i = 0; i < D; i++) x[i] += mo[i];
    if (r) { r.y = y; r.f = f; r.sal = x.slice(); c.reg.capas[l].push(r); }
  }
  c.n++;
  let logits = null;
  if (conLogits) {
    const xf = lnorm(x, Red.P['ln_f.weight'], Red.P['ln_f.bias'], eps), V = Red.meta.vocab; logits = new Float32Array(V);
    for (let v = 0; v < V; v++) { let s = 0; const off = v * D; for (let i = 0; i < D; i++) s += te[off + i] * xf[i]; logits[v] = s; }
  }
  return { punt, logits };
}
// corre la frase entera: devuelve los puntajes de todas las posiciones y los logits del último token
function correr(ids) {
  const c = nuevaCache(true), filas = [];
  let last = null;
  ids.forEach((id, i) => { const r = paso(id, c, i === ids.length - 1); filas.push(r.punt); if (r.logits) last = r.logits; });
  return { cache: c, filas, logits: last };
}
function softmax(a) { let mx = -Infinity; for (const v of a) if (v > mx) mx = v; const p = new Float32Array(a.length); let s = 0; for (let i = 0; i < a.length; i++) { p[i] = Math.exp(a[i] - mx); s += p[i]; } for (let i = 0; i < p.length; i++) p[i] /= s; return p; }

async function pedir(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} respondió ${r.status}`);
  return r;
}
// baja un archivo grande avisando cuántos bytes van; si el navegador ya lo tiene guardado, lo usa
async function bajarBinario(url, alBytes) {
  let cache = null;
  try { cache = await caches.open('gpt-a-la-vista-v1'); const hit = await cache.match(url); if (hit) { const b = await hit.arrayBuffer(); alBytes(b.byteLength); return b; } } catch (e) { cache = null; }
  const r = await pedir(url);
  const partes = []; let n = 0;
  if (r.body && r.body.getReader) {
    const lector = r.body.getReader();
    for (;;) { const { done, value } = await lector.read(); if (done) break; partes.push(value); n += value.byteLength; alBytes(value.byteLength); }
  } else { const b = new Uint8Array(await r.arrayBuffer()); partes.push(b); alBytes(b.byteLength); }
  const u8 = new Uint8Array(partes.reduce((a, p) => a + p.byteLength, 0)); let o = 0; for (const p of partes) { u8.set(p, o); o += p.byteLength; }
  if (cache) { try { await cache.put(url, new Response(u8.slice().buffer, { headers: { 'Content-Type': 'application/octet-stream' } })); } catch (e) { /* sin lugar: se baja de nuevo la próxima vez */ } }
  return u8.buffer;
}
async function cargarModelo(base, alProgreso) {
  const meta = await (await pedir(`${base}/meta.json`)).json();
  const [vocab, merges] = await Promise.all([pedir(`${base}/vocab.json`).then(r => r.json()), pedir(`${base}/merges.txt`).then(r => r.text())]);
  Tokenizador.cargar(vocab, merges);
  const total = meta.orden.reduce((a, { shape }) => a + 2 * shape.reduce((x, y) => x * y, 1), 0);
  let bajado = 0;
  const buffers = await Promise.all(meta.archivos.map(f => bajarBinario(`${base}/${f}`, n => { bajado += n; alProgreso && alProgreso(Math.min(1, bajado / total), bajado, total); })));
  armarPesos(meta, buffers);
  return meta;
}
