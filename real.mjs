// The author's personal read-only tools (job log, local repos) are not published.
// REAL=1 has no effect in this repo; use SIM=1 or BIZ=1.
export const realTools = [];
export const realReadOnly = [];
export const entities = [];
export async function callReal(name) { throw new Error(`real tool ${name} not included`); }
