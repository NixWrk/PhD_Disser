"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { createSurface, closestPoint } = require("../tools/surface_pair_geometry.js");
const close = (a,b,tol=1e-10) => {
  assert.equal(a.length,b.length);
  a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<tol, `${a} != ${b}`));
};
const surface = () => createSurface([
  [-4,0,0],[-4,10,0],[-4,0,10], [6,0,0],[6,10,0],[6,0,10]
], [[0,1,2],[3,4,5]]);

test("mirror through a nonzero plane uses the triangle interior",()=>{
  const p=surface().pair([-4,2,3],0,1);
  close(p.selected,[-4,2,3]); close(p.opposite,[6,2,3]);
  assert.equal(p.projectionMm,0);
});
test("opposite-side click reverses the same symmetric pair",()=>{
  const p=surface().pair([6,2,3],0,1);
  close(p.opposite,[-4,2,3]); assert.equal(p.side,1);
});
test("source and reflected positions are both projected to the surface",()=>{
  const p=surface().pair([-4.1,2,3],0,1);
  close(p.selected,[-4,2,3]); close(p.opposite,[6,2,3]);
});
test("asymmetry projects to an edge interior rather than its closest vertex",()=>{
  const s=createSurface([[-4,0,0],[-4,10,0],[-4,0,10],[6,0,0],[6,2,0],[6,0,2]],[[0,1,2],[3,4,5]]);
  const p=s.pair([-4,1.5,1.5],0,1);
  close(p.opposite,[6,1,1]);
  assert.ok(Math.abs(p.projectionMm-Math.sqrt(.5))<1e-10);
});
test("a point on the mid-plane and planes outside the body are rejected",()=>{
  assert.equal(surface().pair([1,2,3],0,1),null);
  assert.equal(surface().pair([-4,2,3],0,10),null);
  assert.equal(surface().pair([-4,2,3],0,-4),null);
});
test("invalid plane or coordinates cannot create a pair",()=>{
  assert.equal(surface().pair([-4,2,3],3,1),null);
  assert.equal(surface().pair([-4,2,3],0,NaN),null);
  assert.equal(surface().pair([Infinity,2,3],0,1),null);
});
test("triangle degeneracy falls back to line segments",()=>{
  close(closestPoint([1,2,0],[0,0,0],[1,0,0],[2,0,0]),[1,0,0]);
});
test("triangle vertex and interior projections remain finite",()=>{
  close(closestPoint([-1,-1,4],[0,0,0],[2,0,0],[0,2,0]),[0,0,0]);
  close(closestPoint([.5,.5,4],[0,0,0],[2,0,0],[0,2,0]),[.5,.5,0]);
});

