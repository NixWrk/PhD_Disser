// Exactly one text substitution in the original code-native SVG; offline.
const fs=require('fs'),path=require('path'),assert=require('assert/strict'),crypto=require('crypto');
const sharp=require('C:/Users/ПАВЛУША/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const rev=path.resolve(__dirname,'..'),root=path.resolve(rev,'../../../..');
const base=path.resolve(rev,'../revision_20261029_author_review');
const rel=p=>path.relative(root,p).replaceAll('\\','/');
const sha=b=>crypto.createHash('sha256').update(b).digest('hex');
async function main(){
 const src=path.join(base,'figures/V01_materials_sequence.svg'),png=src.replace(/svg$/,'png');
 const original=fs.readFileSync(src,'utf8'),oldText='Анатомическая опора',newText='Индивидуальная анатомия по КТ';
 assert.equal(original.split(oldText).length,2);
 const edited=original.replace(oldText,newText);
 assert.equal(edited.replace(newText,oldText),original);
 const before=await sharp(Buffer.from(original)).resize({width:2200}).png().toBuffer();
 const after=await sharp(Buffer.from(edited)).resize({width:2200}).png().toBuffer();
 const a=await sharp(before).removeAlpha().raw().toBuffer({resolveWithObject:true});
 const b=await sharp(after).removeAlpha().raw().toBuffer({resolveWithObject:true});
 const sourcePixels=await sharp(png).removeAlpha().raw().toBuffer();
 assert(a.data.equals(sourcePixels),'Original SVG must reproduce the base PNG');
 assert.deepEqual(a.info,b.info);
 const w=a.info.width,h=a.info.height,c=a.info.channels;
 // Includes only the text baseline y=704 and its glyphs, in original SVG coordinates.
 const allowed={x0:0,y0:Math.floor(660*h/820),x1:w,y1:Math.ceil(716*h/820)};
 let changed=0,minX=w,minY=h,maxX=-1,maxY=-1;
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  let differs=false;for(let k=0;k<c;k++)if(a.data[(y*w+x)*c+k]!==b.data[(y*w+x)*c+k])differs=true;
  if(!differs)continue;
  assert(y>=allowed.y0&&y<allowed.y1,'Change outside target text row');
  changed++;minX=Math.min(x,minX);maxX=Math.max(x,maxX);minY=Math.min(y,minY);maxY=Math.max(y,maxY);
 }
 assert(changed>0);assert(minX>0&&maxX<w-1,'Text must fit inside the canvas');
 const dest=path.join(rev,'figures/V01_materials_sequence_ct_anatomy.svg');
 fs.writeFileSync(dest,edited);fs.writeFileSync(dest.replace(/svg$/,'png'),after);
 const rec={old_asset:rel(png),new_path:rel(dest.replace(/svg$/,'png')),new_svg_path:rel(dest),
  sha256:sha(after),svg_sha256:sha(Buffer.from(edited)),dimensions_px:[w,h],
  source_svg:{path:rel(src),sha256:sha(Buffer.from(original))},source_png_sha256:sha(fs.readFileSync(png)),
  source_code:{path:rel(path.join(base,'tools/render_visuals.cjs')),line:28},
  text_change:{before:oldText,after:newText},checks:{original_SVG_reproduces_base_PNG_pixels:true,
   exactly_one_text_substitution:true,SVG_attributes_and_other_elements_unchanged:true,
   pixels_outside_target_text_row_equal:true,dimensions_equal:true,no_raster_patching:true},
  changed_pixel_count:changed,changed_pixel_bbox_xyxy:[minX,minY,maxX+1,maxY+1],allowed_text_row_xyxy:[allowed.x0,allowed.y0,allowed.x1,allowed.y1]};
 assert.equal(sha(fs.readFileSync(src)),rec.source_svg.sha256);
 fs.writeFileSync(path.join(__dirname,'materials_anatomy_check.json'),JSON.stringify(rec,null,2));
 console.log(JSON.stringify({path:rec.new_path,checks:rec.checks,changed_bbox:rec.changed_pixel_bbox_xyxy}));
}
main().catch(e=>{console.error(e);process.exitCode=1;});
