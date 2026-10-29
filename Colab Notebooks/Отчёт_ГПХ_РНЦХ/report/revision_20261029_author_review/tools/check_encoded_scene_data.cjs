// Verify that text anonymisation never changed binary-encoded Plotly arrays.
// No images, scientific inputs, Word files or chapter drafts are written.
const fs=require('fs'),path=require('path');
const rev=path.resolve(__dirname,'..');
const code=fs.readFileSync(path.join(__dirname,'render_visuals.cjs'),'utf8').split('async function main(){')[0];
const check=`
let checked=0,bytes=0;
function audit(v){if(!v||typeof v!=='object')return;if(v.bdata){if(anon(v.bdata)!==v.bdata)throw Error('Old anonymiser would alter binary data');checked++;bytes+=Buffer.from(v.bdata,'base64').length;}for(const[k,x]of Object.entries(v))if(k!=='bdata')audit(x);}
const sources=new Set();
for(const e of manifest.figures)for(const a of [e,...(e.assets||[])])if(a.source?.html&&fs.existsSync(path.join(root,a.source.html)))sources.add(a.source.html);
for(const source of sources)for(const s of scenes(fs.readFileSync(path.join(root,source),'utf8')))audit(s.data);
console.log(JSON.stringify({binary_arrays_checked:checked,bytes_checked:bytes,binary_arrays_unchanged_by_prior_text_anonymisation:true}));
`;
new Function('require','__dirname',code+check)(require,__dirname);
