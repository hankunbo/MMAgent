/* evidence-tree-reader/1.0.0: presentation only, no fetch, inference or scoring. */
'use strict';
const V=JSON.parse(document.getElementById('evidence-view').textContent);
const R=V.reader, nodes=new Map(V.nodes.map(n=>[n.id,n])), children=new Map(), parents=new Map();
for(const e of V.edges.filter(e=>e.relation==='contains')){
 if(!children.has(e.source))children.set(e.source,[]);
 children.get(e.source).push(e.target);parents.set(e.target,e.source);
}
const tree=document.getElementById('tree'),detail=document.getElementById('detail');
const search=document.getElementById('search'),filter=document.getElementById('filter');
const sections=['来源事实','我们的提炼理由','本节点的取证需求','本例工具与观察结果','目前如何判读','计算依赖与判读背景'];
const overviewSections=['病例与目标','结论与关键分数','证据进度与待补项','证据移除与关键依赖','方法、溯源与附录'];
let active='root', kindFilter='', allOpen=false;
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n}
function text(v){return v==null?'本导出未提供':typeof v==='object'?JSON.stringify(v,null,2):String(v)}
function number(v){return typeof v==='number'?Number(v.toFixed(5)).toString():'未评定'}
function points(v){return typeof v==='number'?(v*100).toFixed(4)+' / 100':'未评定'}
function line(p,k,v){const d=el('div',undefined,'reading-line');d.append(el('strong',k),el('span',text(v)));p.append(d)}
function raw(p,label,value){const d=el('details',undefined,'fold');d.append(el('summary',label),el('pre',text(value)));p.append(d)}
function heading(i,titles=sections){const h=el('h3',(i+1)+' · '+titles[i]);h.id='section-'+(i+1);detail.append(h)}
function index(titles){const p=el('div',undefined,'section-index');titles.forEach((t,i)=>{const a=el('a',(i+1)+' '+t);a.href='#section-'+(i+1);p.append(a)});detail.append(p)}
function table(p,labels,rows){const wrap=el('div',undefined,'table-wrap'),t=el('table'),head=el('thead'),tr=el('tr'),body=el('tbody');for(const h of labels){const th=el('th',h);th.scope='col';tr.append(th)}head.append(tr);for(const row of rows){const tr=el('tr');for(const v of row){const td=el('td');td.append(v instanceof Node?v:el('span',text(v)));tr.append(td)}body.append(tr)}t.append(head,body);wrap.append(t);p.append(wrap)}
function button(label,id){const b=el('button',label);b.onclick=()=>show(id);return b}
function badge(p,kind){const [label,color]=R.types[kind];const b=el('span',label,'badge');b.style.borderLeft='4px solid '+color;p.append(b)}
function child(key,suffix){return nodes.get(key+suffix)?.payload}
function tiles(p,summary){const grid=el('div',undefined,'metric-grid');for(const [name,value] of [['综合研究分',summary.confidence?.value],['支持度',summary.combined?.support],['反驳度',summary.combined?.refutation],['冲突度',summary.combined?.conflict]]){const d=el('div',undefined,'metric-tile'+(name==='综合研究分'?' primary':''));d.append(el('span',name),el('strong',points(value)));grid.append(d)}p.append(grid)}
function source(p,s){const box=el('section',undefined,'reading-card');const locator=s.locator||s.reference?.code||s.code||'来源引用';box.append(el('h4',locator));if(s.text)box.append(el('blockquote',s.text));
 // Never turn arbitrary source text into executable URLs or HTML.
 const url=s.url||s.locator;if(typeof url==='string'&&/^https?:\/\//i.test(url)){try{const u=new URL(url);if(['http:','https:'].includes(u.protocol)){const a=el('a','查阅原文 ↗');a.href=u.href;a.target='_blank';a.rel='noopener noreferrer';box.append(a)}}catch{}}
 raw(box,'精确引用、版本与来源状态',s);p.append(box);
}
function observation(p,obs){if(!obs){line(p,'本例结果','未取得对应观察；缺项不视为阴性。');return}
 line(p,'方法',obs.method||obs.result_kind);line(p,'取得状态',obs.status||obs.acquisition);
 if(obs.summary)line(p,'观察摘要',obs.summary);
 let value=obs.value??obs.datum??obs.result; if(typeof value==='string'){try{value=JSON.parse(value)}catch{}}
 if(value&&typeof value==='object'){
  const labels={value:'测量值',age:'年龄',signal:'观察信号',finding:'发现',coverage:'覆盖',clarity:'清晰度',visible_regions:'已观察区域',unseen_regions:'未观察区域',unit:'单位'};
  let shown=false;for(const [k,label] of Object.entries(labels)){if(value[k]!==undefined){line(p,label,value[k]);shown=true}}
  if(!shown)raw(p,'实际观察值',value);
 }else if(value!==undefined)line(p,'观测值',value);
 if(obs.unit)line(p,'单位',obs.unit);raw(p,'观察原记录',obs);
}
function score(p,row){if(!row){line(p,'评分状态','无已保存的疾病方向评阅；不在展示时补分。');return}
 const strip=el('div',undefined,'score-strip');for(const [name,v] of [['可靠性 O',row.observation?.value],['关系 R',row.relation?.value],['适用性 A',row.applicability?.value],['连续判读 e',row.judgment?.value],['支持贡献',row.support],['反驳贡献',row.refutation]]){const s=el('span',name+' ');s.append(el('b',number(v)));strip.append(s)}p.append(strip);
 raw(p,'三维评分理由、权重与原贡献',row);
}
function check(n){const key=n.id,id=n.payload.id,task=R.tasks[id]||{},req=R.requirements[id]||{},def=n.payload.definition||{};
 const obs=child(key,':fact'),row=child(key,':review'),quality=child(key,':quality'),review=R.interpretations[id]||{};
 detail.append(el('div','病例目标 → '+n.title,'breadcrumbs'),el('h2',n.title));badge(detail,'requirement');detail.append(el('span',n.state,'badge'));index(sections);
 heading(0);if(n.payload.sources.length)n.payload.sources.forEach(s=>source(detail,s));else line(detail,'来源','本导出未携带来源正文；不从其他病例补写。');
 heading(1);line(detail,'与目标的关系',def.interpretation||'本导出未提供独立的提炼理由。');
 heading(2);line(detail,'检查需求',def.examination||req.examination||n.title);line(detail,'输入模态',(n.payload.modalities||[]).join('、')||'本导出未提供');line(detail,'本轮已用方法',task.method||obs?.method||'尚无已绑定执行结果');
 heading(3);observation(detail,obs);
 heading(4);const card=el('section',undefined,'reading-card');
 line(card,'检查参照',def.interpretation||'本导出未携带独立判读定义；需回查原来源，转换器不补写阈值。');
 line(card,'本例结果',review.comparison|| (obs?'已取得结果，具体值见第4节。':req.status||'缺少对应患者观察。'));
 line(card,'判读',row?.judgment?.reason||review.judgment?.reason||'缺少已保存的本例判读；无疾病方向贡献不等于检查阴性。');
 if(review.unresolved?.length)line(card,'待补信息',review.unresolved.join('；'));
 score(card,row);detail.append(card);
 heading(5);line(detail,'适用条件',def.applicability_conditions||'本导出未提供');
 if(quality){table(detail,['过程指标','原值'],Object.entries(quality).filter(([k])=>['availability_score','inspected_fraction','effective_quality','gap_score','disease_direction','action','reason','quality_reason'].includes(k)));raw(detail,'完整质量与过程评价',quality)}
 if(row){line(detail,'同源组',row.correlation_group);line(detail,'依赖组',row.dependence_block);raw(detail,'输入依赖路径',row.dependency_paths||[])}
 raw(detail,'检查定义与任务精确绑定',{requirement:req,task,interpretation:review});
}
function overview(){const summary=V.scores.root?.summary||{};detail.append(el('h2','病例总览'));index(overviewSections);
 heading(0,overviewSections);line(detail,'输入',V.sample);line(detail,'范围','保存结果的只读展示；不执行模型、不新增医学判读。');
 heading(1,overviewSections);tiles(detail,summary);detail.append(el('p','研究分不是患病概率；50不是确诊阈值。六类节点表示职责，不表示六份独立证据。','notice'));
 heading(2,overviewSections);const c=V.counts;table(detail,['需求','已评阅','数值判读','非零贡献','中性','方向未评定'],[[c.requirements,c.reviewed,c.numeric,c.nonzero_contributors,c.neutral,c.unassessed]]);
 table(detail,['覆盖域','取得比例','判读比例','分母'],Object.entries(V.scores.root?.coverage||{}).map(([k,v])=>[k,number(v.acquisition),number(v.interpretation),v.denominator]));
 table(detail,['检查','状态','连续判读 e','支持','反驳'],V.nodes.filter(n=>n.type==='requirement').map(n=>{const r=child(n.id,':review');return[button(n.title,n.id),nodes.get(n.id+':review')?.state||n.state,number(r?.judgment?.value),number(r?.support),number(r?.refutation)]}));
 heading(3,overviewSections);detail.append(el('p','以下是原评分保存的移除分析，不改变正常推理，也不代表临床因果。'));
 table(detail,['移除资源或项目','移除后研究分'],V.removals.map(r=>[r.resource||r.id||r.removed_resource||'见原记录',points((r.summary||r.after?.summary||{}).confidence?.value)]));raw(detail,'精确移除分析与覆盖变化',V.removals);
 heading(4,overviewSections);raw(detail,'完成状态',R.completion);raw(detail,'评分方法与保存版本',{scoring_version:V.scoring_version,method_notes:V.scores.method_notes,source_version:V.source_version,source_checksum:V.source_checksum,reader_version:R.version,scope:V.scope});
}
function show(id){if(!nodes.has(id))return;active=id;detail.replaceChildren();let n=nodes.get(id);const p=parents.get(id);
 if(p&&nodes.get(p)?.type==='requirement'){check(nodes.get(p));const section={observation:4,interpretation:5,context:6,execution:6}[n.type];document.getElementById('section-'+section)?.scrollIntoView({block:'start'})}
 else if(n.type==='requirement')check(n);
 else if(id==='root')overview();
 else {detail.append(el('h2',n.title));badge(detail,n.type);if(n.payload?.inventory){table(detail,['候选需求','处理','去向'],n.payload.inventory.rows.map(r=>[r.title,r.disposition,r.target||r.followup]))}raw(detail,'已保存的输入、质量或执行记录',n.payload)}
 draw();if(!(p&&nodes.get(p)?.type==='requirement'))detail.scrollTop=0;
}
function draw(){tree.replaceChildren();const q=search.value.trim().toLocaleLowerCase(),state=filter.value;
 const matches=n=>(!q||(n.title+' '+n.state+' '+JSON.stringify(n.payload)).toLocaleLowerCase().includes(q))&&(!state||n.state===state)&&(!kindFilter||n.type===kindFilter);
 function include(id){return matches(nodes.get(id))||(children.get(id)||[]).some(include)}
 function build(id){if(!include(id))return null;const n=nodes.get(id),li=el('li'),b=button(n.title,id);b.className='node'+(id===active?' active':'');b.style.textAlign='left';b.style.width='100%';b.setAttribute('aria-current',id===active?'true':'false');badge(b,n.type);b.append(el('span',n.state,'badge'));
 const sub=(children.get(id)||[]).map(build).filter(Boolean);if(sub.length){const d=el('details'),s=el('summary'),ul=el('ul');d.open=allOpen||id==='root'||id===active||parents.get(active)===id||Boolean(q||state||kindFilter);s.append(b);ul.append(...sub);d.append(s,ul);li.append(d)}else li.append(b);return li}
 const ul=el('ul'),root=build('root');if(root)ul.append(root);else tree.append(el('p','没有匹配节点。'));tree.append(ul);
}
for(const s of new Set(V.nodes.map(n=>n.state))){const o=el('option',s);o.value=s;filter.append(o)}
const legend=document.getElementById('structure-navigation');legend.hidden=false;
for(const [kind,[label,color]] of Object.entries(R.types)){const b=el('button',label);b.style.borderLeft='4px solid '+color;b.setAttribute('aria-pressed','false');b.onclick=()=>{kindFilter=kindFilter===kind?'':kind;for(const e of legend.children)e.setAttribute('aria-pressed','false');b.setAttribute('aria-pressed',String(kindFilter===kind));draw()};legend.append(b)}
document.getElementById('summary').textContent='综合研究分 '+points(V.confidence)+' · '+V.counts.reviewed+' 项评阅 · '+V.counts.nonzero_contributors+' 项非零贡献';
search.oninput=draw;filter.onchange=draw;document.getElementById('overview').onclick=()=>show('root');
document.getElementById('criteria').onclick=()=>{show('root');document.getElementById('section-3').scrollIntoView({block:'start'})};
document.getElementById('expand').onclick=()=>{allOpen=true;draw()};document.getElementById('collapse').onclick=()=>{allOpen=false;show('root')};
show('root');
