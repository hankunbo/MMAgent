/* v2 changes navigation only; the complete exact v1 evidence payload is retained. */
const evidenceNodes=new Map(V.nodes.map(n=>[n.id,n]));
child=(key,suffix)=>evidenceNodes.get(key+suffix)?.payload;
nodes.clear();children.clear();parents.clear();
for(const n of R.navigation){nodes.set(n.id,n);if(n.parent){if(!children.has(n.parent))children.set(n.parent,[]);children.get(n.parent).push(n.id);parents.set(n.id,n.parent)}}
let opened=new Set(['root','nav:STRUCTURE','nav:CUPDISC','nav:CUP_RIM_MEASURE']);
const originalCheck=check, originalOverview=overview;
function pathTo(id){const path=[];while(id){path.unshift(nodes.get(id));id=parents.get(id)}return path}
function leafIds(id){return nodes.get(id)?.type==='requirement'?[id]:(children.get(id)||[]).flatMap(leafIds)}
function imagesFor(id){const imageIds=new Set();for(const leaf of leafIds(id)){for(const role of R.media_roles[leaf]||[])imageIds.add(role);const observation=child(leaf,':fact');if(observation?.image?.reference){for(const [role,image] of Object.entries(R.media.images)){if(image.reference?.asset_code===observation.image.reference.code)imageIds.add(role)}}}return [...imageIds]}
function gallery(imageIds){const box=el('section',undefined,'evidence-images'),grid=el('div',undefined,'pictures');
 for(const role of imageIds){const item=R.media.images[role];if(!item)continue;const fig=el('figure');fig.dataset.role=role;fig.append(el('figcaption',item.title));const b=el('button',undefined,'image-open');b.type='button';b.setAttribute('aria-label','放大：'+item.title);const image=el('img');image.src=item.data_uri;image.alt=item.title;image.loading='lazy';b.append(image);b.onclick=()=>{const d=el('dialog',undefined,'image-dialog'),bar=el('div',undefined,'image-actions'),close=el('button','关闭');bar.append(el('strong',item.title),close);const full=el('img');full.src=item.data_uri;full.alt=item.title;d.append(bar,full,el('p',item.note,'image-note'));close.onclick=()=>d.close();d.onclose=()=>d.remove();document.body.append(d);d.showModal()};fig.append(b,el('p',item.note,'image-note'));raw(fig,'图片来源与摘要',{checksum:item.checksum,reference:item.reference,shape:item.shape});grid.append(fig)}
 if(grid.children.length){box.append(grid,el('p','点击图片可放大。图像均来自本轮已保存输入或产物；展示不新增疾病证据。','image-note'))}
 else box.append(el('p',imageIds.length?'所需图片未装配或未取得，未用其他病例图片代替。':'本项没有对应的图像检查结果；不把眼底图冒充家族史、眼压或OCT证据。','subtle'));
 return box;
}
check=function(n){originalCheck(n);const breadcrumbs=detail.querySelector('.breadcrumbs');if(breadcrumbs)breadcrumbs.textContent=pathTo(n.id).map(x=>x.title).join(' → ');detail.insertBefore(gallery(imagesFor(n.id)),document.getElementById('section-5'))};
overview=function(){originalOverview();detail.insertBefore(gallery(['input','display']),document.getElementById('section-2'));line(detail,'展示层级','临床分支仅用于导航；实际检查数量以报告为准，组织节点不重复计分。')};
show=function(id){if(!nodes.has(id))return;active=id;for(const n of pathTo(id).slice(0,-1))opened.add(n.id);detail.replaceChildren();const n=nodes.get(id);
 if(id==='root')overview();else if(n.type==='requirement')check(n);else{detail.append(el('div',pathTo(id).map(x=>x.title).join(' → '),'breadcrumbs'),el('h2',n.title),el('p','按检查主题组织已有证据。请选择下级检查查看完整六节内容；分支不另算一份疾病证据。','subtle'));const cards=el('div',undefined,'branch-grid');for(const key of children.get(id)||[]){const v=nodes.get(key),card=el('section',undefined,'branch-card');card.append(button(v.title,key),el('p',v.type==='requirement'?v.state:'展开下级检查'));cards.append(card)}detail.append(cards,gallery(imagesFor(id)))}
 draw();detail.scrollTop=0;
};
draw=function(){tree.replaceChildren();const q=search.value.trim().toLocaleLowerCase(),state=filter.value;
 const matches=n=>n.type==='requirement'&&(!q||(n.title+' '+JSON.stringify(n.payload)).toLocaleLowerCase().includes(q))&&(!state||n.state===state);
 const include=id=>matches(nodes.get(id))||(children.get(id)||[]).some(include);
 function build(id){if(!include(id)&&id!=='root')return null;const n=nodes.get(id),li=el('li'),b=button(n.title,id);b.className='node'+(id===active?' active':'')+(id==='root'?' tree-root':'');b.setAttribute('aria-current',String(id===active));
 if(n.type==='requirement')b.append(el('span',n.state,'badge'));
 const sub=(children.get(id)||[]).map(build).filter(Boolean);if(sub.length){const d=el('details'),s=el('summary'),ul=el('ul');d.open=opened.has(id)||Boolean(q||state);d.ontoggle=()=>{if(d.isConnected){if(d.open)opened.add(id);else opened.delete(id)}};b.onclick=e=>{e.preventDefault();e.stopPropagation();show(id)};s.append(b);ul.append(...sub);d.append(s,ul);li.append(d)}else{li.className='leaf-row';li.append(b)}return li}
 tree.append(el('p','医学检查路径树 · 点击名称看详情，点击三角展开','navigation-note'));const ul=el('ul');ul.append(build('root'));tree.append(ul);
 if(!leafIds('root').some(id=>matches(nodes.get(id))))tree.append(el('p','没有匹配检查。','muted'));
};
filter.replaceChildren();const any=el('option','全部状态');any.value='';filter.append(any);for(const s of new Set(R.navigation.filter(n=>n.type==='requirement').map(n=>n.state))){const o=el('option',s);o.value=s;filter.append(o)}
legend.replaceChildren();legend.append(el('span','来源、观察、评分、质量和依赖统一放在右侧六节，不再作为重复树枝。','navigation-note'));
search.oninput=draw;filter.onchange=draw;
document.getElementById('expand').onclick=()=>{opened=new Set(nodes.keys());draw()};document.getElementById('collapse').onclick=()=>{opened=new Set(['root']);draw()};
show('root');
