/* Presentation only. Never mutate snapshots, recalculate medical scores or infer facts. */
const READER_VERSION='case-evidence-reader/2.0.0';
const modernReader=!!(D.current_scoring&&D.evidence_records);
const overviewSections=['病例与目标','结论与关键分数','证据进度与待补项','证据移除与关键依赖','方法、溯源与附录'];
const nodeSections=['来源事实','我们的提炼理由','本节点的取证需求','本例工具与观察结果','目前如何判读','计算依赖与判读背景'];
const reviewNames=new Map((D.current_scoring?.reviews||[]).map(r=>[r.id,r.title]));
const reviewById=new Map((D.current_scoring?.reviews||[]).map(r=>[r.id,r]));
const obsById=new Map((D.evidence_records?.observations||[]).map(o=>[o.id,o]));
const kindNames={measurement:'算法测量',proxy_measurement:'代理测量',visual_review:'视觉评阅',model_output:'模型输出',provided_information:'已提供字段',legacy_observation:'保存的观察',unavailable:'缺少输入'};
const directionNames={support_side:'偏支持',refutation_side:'偏反驳',balanced:'正反平衡',unavailable:'未形成方向'};
const roleNames={diagnostic:'医学测量',risk:'风险背景',model:'模型意见',visual:'视觉观察',association:'文献关联',measurement:'技术测量'};
const artifactNames={SLO_IMAGE:'SLO眼底图像',FUNDUS_IMAGE:'彩色眼底图像',OCT_IMAGE:'OCT影像'};
const measurementNames={vcdr:'垂直杯盘比',skeleton_density:'骨架密度',fractal_dimension:'分形维数',tortuosity:'迂曲度',width_proxy_disc_ratio:'宽度代理 / 视盘直径',width_over_same_axis_disc_diameter:'盘沿宽度 / 同轴盘径'};
function percent(v){return v==null?'未提供':(v*100).toFixed(2)+'%'}
function points(v){return v==null?'未评分':(v*100).toFixed(2)}
function signed(v){return v==null?'不可计算':(v>0?'+':'')+(v*100).toFixed(2)}
function fold(label,parent=detail){const box=el('details',undefined,'fold');box.append(el('summary',label));parent.append(box);return box}
function line(parent,label,text){const row=el('div',undefined,'reading-line');row.append(el('strong',label),el('span',text??'未提供'));parent.append(row)}
function tableInto(parent,headers,rows){
 const wrap=el('div',undefined,'table-wrap'),table=el('table'),head=el('thead'),tr=el('tr'),body=el('tbody');
 for(const h of headers){const th=el('th',h);th.setAttribute('scope','col');tr.append(th)}head.append(tr);
 for(const values of rows){const r=el('tr');for(const value of values){const cell=el('td');if(value&&typeof value==='object')cell.append(value);else cell.textContent=String(value??'未提供');r.append(cell)}body.append(r)}
 table.append(head,body);wrap.append(table);parent.append(wrap);return wrap;
}
function metricTiles(parent,items){const grid=el('div',undefined,'metric-grid');for(const [label,value,note,primary] of items){const card=el('div',undefined,'metric-tile'+(primary?' primary':''));card.append(el('span',label),el('strong',value),el('small',note||''));grid.append(card)}parent.append(grid)}
function readerHeading(index,titles,scope){const h=el('h3',(index+1)+' · '+titles[index]);h.id=scope+'-'+(index+1);detail.append(h)}
function sectionIndex(titles,scope){const nav=el('div',undefined,'section-index');nav.setAttribute('aria-label','页内章节');for(const [i,title] of titles.entries()){const a=el('a',(i+1)+' '+title);a.href='#'+scope+'-'+(i+1);a.onclick=ev=>{if(ev?.preventDefault)ev.preventDefault();document.getElementById(scope+'-'+(i+1))?.scrollIntoView?.({block:'start'})};nav.append(a)}detail.append(nav)}
function captureFold(label,callback,parent=detail){
 // Legacy image/trace renderers append to detail; move their output, not their data.
 const before=Array.from(detail.children);callback();const added=Array.from(detail.children).slice(before.length);
 detail.replaceChildren(...before);if(!added.length)return null;
 const box=fold(label,parent);box.append(...added);return box;
}
function nodeButton(key){const b=el('button',byKey.get(key)?.title||key);b.onclick=()=>show(key);return b}
function rootNode(){return D.nodes.find(n=>n.parent===null)||D.nodes[0]}
function nodeChildren(key){return S?S.main_edges.filter(e=>e.from===key).map(e=>byKey.get(e.to)).filter(Boolean):D.nodes.filter(n=>n.parent===key)}
function ownPairs(key){const ids=new Set(D.evidence_records.node_index[key]||[]);return D.evidence_records.interpretations.filter(j=>ids.has(j.id)).map(j=>({j,o:obsById.get(j.observation_id),r:reviewById.get(j.assessment_id)}))}
function datumValue(v){return Array.isArray(v)?v.join(', '):typeof v==='number'?String(Number(v.toPrecision(8))):String(v)}
function datumName(name){const parts=name.split(':'),field=parts.at(-1),prefix=parts.length===2?({artery:'动脉',vein:'静脉'})[parts[0]]:null;return prefix?prefix+' '+(measurementNames[field]||field):measurementNames[name]||name}
function datumText(o){return o.values.length?o.values.map(v=>datumName(v.name)+' = '+(v.value==null?v.missing_reason:datumValue(v.value))+(v.unit&&v.unit!=='1'?' '+v.unit:'')).join('；'):o.summary}
function resourceLabel(key){
 const bindings=D.current_scoring.resources[key]||[];
 for(const b of bindings){
  const artifact=C?.artifacts?.find((_,i)=>b.pointer==='/case_evidence/artifacts/'+i);
  if(artifact){const node=D.nodes.find(n=>(n.patient_evidence?.artifact_codes||[]).includes(artifact.asset_code));if(node)return node.title}
  if(b.pointer==='/case_evidence/input')return '病例原图';
  if(b.pointer==='/vascular_followup/artifact')return '动静脉联合测量结果';
 }
 const existing=D.research_scoring?.removals?.find(r=>r.id==='resource:'+key);
 return existing?.title?.replace(/^移除/,'')||key;
}
function removalTitle(r){return r.removed_assessments.length?r.removed_assessments.map(id=>reviewNames.get(id)||id).join('、'):r.removed_resources.map(resourceLabel).join('、')}
function coverageTable(parent,coverage){tableInto(parent,['评估范围','测量取得 M','可以判读 J'],Object.entries(coverage).map(([scope,c])=>[adequacyScopes[scope]||scope,percent(c.acquisition),percent(c.interpretation)]))}
function channelTable(parent,summary){tableInto(parent,['证据通道','支持 / 100','反驳 / 100','冲突 / 100'],[['diagnostic','医学线索'],['risk','风险背景'],['model','模型意见']].filter(([k])=>summary[k]).map(([key,label])=>[label,points(summary[key].support),points(summary[key].refutation),points(summary[key].conflict)]))}
function branchCards(key){
 const children=nodeChildren(key);if(!children.length)return;
 const grid=el('div',undefined,'branch-grid');
 for(const n of children){const box=el('div',undefined,'branch-card'),summary=D.current_scoring.nodes[n.key];box.append(nodeButton(n.key),el('p',status(n)));if(summary)box.append(el('p','支持 '+points(summary.combined.support)+' · 反驳 '+points(summary.combined.refutation)+' / 100'));grid.append(box)}detail.append(grid);
}
function sourceList(key,parent=detail){
 const e=E.nodes[key]||{},seen=new Set();
 for(const fact of new Set(e.source_facts||[]))parent.append(el('p',fact));
 const items=[...(e.sources||[]),...(vascularItem(key)?.source_ids||[]).map(id=>D.vascular_followup.sources[id])];
 if(!items.length){parent.append(el('p','本项来源为已保存的输入或工具产物，绑定详情见第6栏。','subtle'));return}
 const list=fold('来源网页与原文',parent);
 for(const s of items){if(!s)continue;const text=s.fact||s.summary||s.text||'',id=(s.url||'')+'\n'+text;if(seen.has(id))continue;seen.add(id);const item=el('div',undefined,'source-citation');link(item,s.title||'来源网页',s.url);if(text){const excerpt=fold('展开原文',item);excerpt.append(el('blockquote',text))}list.append(item)}
}
function scoreStrip(parent,r){
 if(!r)return;
 const strip=el('div',undefined,'score-strip');
 for(const [label,v] of [['可靠性',r.observation.value],['关系强度',r.relation.value],['适用性',r.applicability.value]]){const item=el('span',label+' ');item.append(el('b',v==null?'未评分':points(v)+'/100'));strip.append(item)}
 const e=el('span','连续判读 e ');e.append(el('b',r.judgment.value==null?'未赋方向':(r.judgment.value>0?'+':'')+r.judgment.value.toFixed(4)));strip.append(e);parent.append(strip);
 if(r.judgment.value!=null)line(parent,'单项贡献','支持 '+points(r.support)+' / 100；反驳 '+points(r.refutation)+' / 100');
 const reasons=new Map();for(const [label,rating] of [['可靠性',r.observation],['关系强度',r.relation],['适用性',r.applicability],['连续判读',r.judgment]]){const why=rating.reason||rating.missing_reason;if(why)reasons.set(why,[...(reasons.get(why)||[]),label])}
 const panel=fold('评分理由与参数范围',parent);for(const [why,labels] of reasons)line(panel,labels.join(' / '),why);
 if(r.judgment.lower!=null)line(panel,'e 工作范围',r.judgment.lower.toFixed(4)+' — '+r.judgment.upper.toFixed(4));
}
function boundReference(j){
 // A fixed comparator is displayed independently of the later continuous judgment.
 const id=j.assessment_id.startsWith('criterion:')?j.assessment_id.slice('criterion:'.length):null;
 const rule=id&&D.research_criteria?.policy?.criteria?.find(r=>r.id===id);
 return rule?.condition||j.reference_interpretation;
}
function interpretationCards(key){
 const pairs=ownPairs(key),summary=D.current_scoring.nodes[key];
 for(const {j,o,r} of pairs){
  const card=el('section',undefined,'reading-card');card.setAttribute('data-assessment-id',j.assessment_id);
  const title=el('h4',r?.title||o.title);title.append(el('span',recordStates[j.status]||j.status,'badge'));card.append(title);
  line(card,'检查参照',boundReference(j));line(card,'本例结果',datumText(o));line(card,'判读',j.comparison);
  scoreStrip(card,r);
  if(j.status!=='assessed'){line(card,'待补信息',j.unresolved);if(j.next_action!==j.unresolved)line(card,'下一步',j.next_action)}
  detail.append(card);
 }
 if(!pairs.length){detail.append(el('p',nodeChildren(key).length?'本节点汇总下级判读；具体检查参照、数值和三维评分保留在对应子节点。':'本节点为输入、产物或组织关系，具体判读见其服务的检查。','subtle'));branchCards(key);if(P.nodes[key]?.supporting)line(detail,'用途',P.nodes[key].supporting.assessment)}
 if(summary){
  const box=fold('本节点汇总与逐项定位（去重后）');
  metricTiles(box,[['综合研究分',points(summary.confidence.value),'/ 100'],['支持',points(summary.combined.support),'/ 100'],['反驳',points(summary.combined.refutation),'/ 100'],['冲突',points(summary.combined.conflict),'/ 100']]);
  if(!pairs.length)tableInto(box,['判读项目','位置','连续判读 e'],D.current_scoring.reviews.filter(r=>summary.assessment_ids.includes(r.id)).map(r=>[r.title,nodeButton(r.node),r.judgment.value==null?'未赋方向':r.judgment.value.toFixed(4)]));
 }
}
function nodeObservations(key){
 const pairs=ownPairs(key);
 if(pairs.length)tableInto(detail,['观察 / 测量','取得状态','方法','结果'],pairs.map(({o})=>[o.title,o.acquisition==='obtained'?'已有结果':'当前输入无法取证',kindNames[o.result_kind]||o.result_kind,datumText(o)]));
 else detail.append(el('p',S?.node_roles[key]?.result_summary||E.nodes[key]?.result_summary||'本节点汇总下级结果。'));
 // Masks, overlays and execution detail remain reachable once, out of the reading path.
 captureFold('图像、测量明细与工具运行记录',()=>showResults(key,byKey.get(key),E.nodes[key],byKey.get(key).patient_evidence||{}));
}
function dependencyDetails(key){
 const summary=D.current_scoring.nodes[key],ids=new Set(summary?.assessment_ids||[]);
 const rows=D.current_scoring.reviews.filter(r=>ids.has(r.id));
 if(rows.length){const panel=fold('当前有效的输入与计算依赖');tableInto(panel,['判读项目','必需输入（＋为同时需要；或为可替代路径）'],rows.map(r=>[r.title,r.dependency_paths.map(path=>path.map(resourceLabel).join(' ＋ ')).join(' 或 ')||'无可用路径']))}
 if(S){const edges=S.dependency_edges.filter(e=>e.to===key||e.from===key);for(const edge of edges){const box=fold((byKey.get(edge.from)?.title||edge.from)+' → '+(byKey.get(edge.to)?.title||edge.to));box.append(el('p',edge.purpose));targetButton(box,edge.from,'查看依赖：');targetButton(box,edge.to,'服务检查：')}}
 const removals=D.current_scoring.removals.filter(r=>r.affected_assessments.some(id=>ids.has(id)));
 if(removals.length){const box=fold('移除本节点证据或输入的影响');removalTable(box,removals)}
 captureFold('完整判读链、来源资格与精确引用',()=>judgmentTraceDetails(key));
 const b=el('button','查看全图评分方法与移除分析');b.onclick=()=>{overview();document.getElementById('case-4')?.scrollIntoView?.({block:'start'})};detail.append(b);
}
function removalTable(parent,items){tableInto(parent,['移除项','影响判读数','移除后综合分','变化（后 − 前）','取证覆盖 M'],items.map(r=>[removalTitle(r),r.affected_assessments.length,points(r.after.summary.confidence.value),signed(r.confidence_delta),percent(r.after.coverage.slo_image_research?.acquisition)]))}
function methods(parent){
 const box=fold('统一评分口径与阅读说明',parent);
 tableInto(box,['指标','含义'],[['可靠性 / 关系强度 / 适用性','每项0—100；分别说明观察是否可信、与目标疾病的关系以及本例能否适用。'],['连续判读 e','−1反驳 → 0中性 → +1支持；缺失值保留未赋方向。'],['单项支持与反驳','权重 w=可靠性×关系强度×适用性（各维度用0—1）；s=w·max(e,0)，r=w·max(−e,0)。'],['综合研究分 D','D=50+50×(S−R)，S/R为去重聚合后的0—1值。50为代数中点，不是诊断阈值；未校准为患病概率。'],['取证覆盖 M / 判读覆盖 J','按已声明检查槽和固定权重分母计算；取得测量不代表已有可用疾病判读。'],['证据移除','只在分析副本中移除证据或必需资源，重算剩余证据；不删除病例记录、不修改患者测量。'],['参数范围与关键集','参数范围不是统计置信区间；保真集保持本评分接近，不等于足以诊断。']]);
 const notes=fold('完整计算策略',box);for(const note of new Set(D.current_scoring.method_notes))notes.append(el('p',note));
}
function optionalAppendix(parent){
 const box=fold('可选附加实验 · 已保存的工具替换',parent),t=D.current_scoring.tool_replacement;
 box.append(el('p','历史测量对照保留供查阅，不作为本例完成条件，也不改变当前结论。','appendix-note'));
 line(box,'替换定义',t.substitution);
 if(t.original&&t.alternative)tableInto(box,['测量','原方案','替代方案','差值'],Object.keys(t.deltas||{}).map(key=>[({vertical_cup_disc_ratio:'垂直杯盘比',minimum_radial_ratio:'最薄径向盘沿比',disc_pixels:'视盘像素数',cup_pixels:'视杯像素数'})[key]||key,number(t.original[key]),number(t.alternative[key]),number(t.deltas[key])]));
 line(box,'当前状态',({measured_requires_new_interpretation:'测量对照已完成；未形成替代方案的疾病评分',inputs_unavailable:'缺少可核验的输入',qc_rejected:'未通过几何检查'})[t.status]||t.status);
 const more=fold('原始试验说明',box);line(more,'后续可选动作',t.next_action);line(more,'失效旧判读数',String(t.invalidated_assessments.length));line(more,'仅剩其他证据的分数',confidenceText(t.remaining_evidence_summary.confidence)+'（不是替代方案的最终分数）');if(t.error)line(more,'原因',t.error);
}
function missingPanel(parent){
 const pairs=D.evidence_records.interpretations.filter(j=>j.status!=='assessed');
 const groups=new Map();for(const j of pairs){const label=recordStates[j.status]||j.status;groups.set(label,[...(groups.get(label)||[]),j])}
 for(const [label,rows] of groups){const box=fold(label+' · '+rows.length+'项',parent);tableInto(box,['检查','具体缺项 / 用途','下一步'],rows.map(j=>[nodeButton(j.node),j.unresolved,j.next_action]))}
}
function impactOverview(){
 const s=D.current_scoring,c=s.conclusion,k=s.key_analysis;
 const contrasts=[['移除年龄',c.without_age],['移除全部风险背景',c.without_background],['仅图像证据',c.image_evidence],['仅风险背景',c.background_only]];
 tableInto(detail,['证据范围','综合分 / 100','与全部证据的差'],contrasts.map(([label,v])=>[label,points(v.confidence.value),v.confidence.value==null||c.all_evidence.confidence.value==null?'不可计算':signed(v.confidence.value-c.all_evidence.confidence.value)]));
 detail.append(el('p','年龄是否改变方向：'+(c.age_changes_direction?'是':'否')+'；年龄贡献差 '+signed(c.age_delta)+' 分（全部证据 − 移除年龄）。','notice'));
 const meaningful=s.removals.filter(r=>r.removed_assessments.length&&r.confidence_delta!=null&&Math.abs(r.confidence_delta)>1e-10).sort((a,b)=>Math.abs(b.confidence_delta)-Math.abs(a.confidence_delta));
 const impact=fold('单项移除影响排行（前5项）');removalTable(impact,meaningful.slice(0,5));
 const all=fold('全部 '+s.removals.length+' 项证据 / 资源移除对照');removalTable(all,s.removals);
 const key=el('div',undefined,'reading-card');key.append(el('h4','关键证据集'));
 for(const set of k.smallest_fidelity_sets){line(key,'最小保真集',set.assessment_ids.map(id=>reviewNames.get(id)||id).join('、'));line(key,'保留后的分数',points(set.value)+' / 100')}
 if(!k.smallest_fidelity_sets.length)line(key,'分析状态',k.status==='bounded_not_run'?'超出本轮枚举上限，未穷举':'未找到满足当前保真条件的集合');
 for(const set of k.smallest_direction_flip_removals)line(key,'改变方向的最小移除集',set.map(id=>reviewNames.get(id)||id).join('、'));
 const rules=fold('关键集的计算条件',key);rules.append(el('p',k.candidates+'项方向判读；枚举'+k.evaluated_subsets+'个子集。分差≤'+(k.score_tolerance*100).toFixed(2)+'分；保留≥'+(k.minimum_interpretation_retention*100).toFixed(0)+'%判读覆盖及已有论据角色。'));detail.append(key);
}
function modernOverview(){
 active=null;viewGroup='main';render();detail.replaceChildren();detail.scrollTop=0;
 const root=rootNode(),s=D.current_scoring,c=s.conclusion;
 detail.append(el('p','病例证据图 · 总览','eyebrow'),el('h2',root.title));sectionIndex(overviewSections,'case');
 readerHeading(0,overviewSections,'case');
 const facts=el('dl',undefined,'fact-grid');
 const type=C?.input?.artifact_type_code;
 for(const [label,value] of [['评估目标',root.title],['病例 / 样本',C?.sample||C?.case_code||s.case_code],['输入类型',artifactNames[type?.split(':').at(-1)]||type||'见输入记录'],['图像尺寸（高 × 宽）',C?.input?.dimensions?.join(' × ')||'未提供'],['眼别',C?.input?.laterality==='UNKNOWN'?'未知':C?.input?.laterality||'未提供'],['本页规模',D.nodes.length+'个节点 · '+s.reviews.length+'项判读记录']]){const pair=el('div');pair.append(el('dt',label),el('dd',value));facts.append(pair)}
 detail.append(facts);
 readerHeading(1,overviewSections,'case');
 const sum=s.root.summary;metricTiles(detail,[['综合研究分 D',points(sum.confidence.value),'/ 100',true],['支持 S',points(sum.combined.support),'/ 100'],['反驳 R',points(sum.combined.refutation),'/ 100'],['冲突',points(sum.combined.conflict),'/ 100']]);
 detail.append(el('p',c.conclusion));
 detail.append(el('p','研究评分，尚未校准为患病概率。计算口径统一见第5节。','subtle'));
 if(c.age_changes_direction||c.background_changes_direction)detail.append(el('p','关键影响：移除风险背景会改变当前评分方向；具体对照见第4节。','notice'));
 const channels=fold('分通道结果与参数范围');channelTable(channels,sum);line(channels,'参数敏感范围',points(sum.confidence.lower)+' — '+points(sum.confidence.upper)+' / 100');
 readerHeading(2,overviewSections,'case');coverageTable(detail,s.root.coverage);
 const inv=D.evidence_records.inventory;detail.append(el('p','已有观察 '+inv.observation_obtained+' / '+inv.total+' 项；已作方向判读 '+inv.direction_assessed+' 项。记录数与加权覆盖率分别统计。','subtle'));branchCards(root.key);
 const missing=fold('待补检查与匹配参照');missingPanel(missing);
 readerHeading(3,overviewSections,'case');impactOverview();
 readerHeading(4,overviewSections,'case');methods(detail);
 const images=fold('本例图像与工具产物');const pics=el('div',undefined,'pictures');for(const key of Object.keys(C?.images||{}))picture(pics,key);images.append(pics);
 const positions=fold('全部节点与类别');for(const [group,label] of Object.entries(groupLabels)){const box=el('p');const b=el('button',label);b.onclick=()=>groupOverview(group);box.append(b,el('span',' · '+(S?.counts[group]??'见树')+'项'));positions.append(box)}
 optionalAppendix(detail);
 const scenarios=fold('可选规划 · 补齐检查后的覆盖上限');tableInto(scenarios,['检查组','评估范围','取证覆盖上限','判读覆盖上限'],s.missing_scenarios.flatMap(x=>Object.keys(x.acquisition_if_completed).map(scope=>[s.groups.find(g=>g.id===x.group_id)?.title||x.group_id,adequacyScopes[scope]||scope,percent(x.acquisition_if_completed[scope]),percent(x.interpretation_if_reviewed[scope])])));scenarios.append(el('p','只计算覆盖上限，不填入假想患者测量或疾病方向；其他病例评价暂不列为本例完成条件。','subtle'));
 const provenance=fold('版本与存储引用');for(const [label,value] of [['展示模板',READER_VERSION],['评分版本',s.schema_version],['计算策略',s.policy_version],['病例标识',s.case_code],['图快照摘要',D.checksum],['评分摘要',s.checksum]])line(provenance,label,value);
}
function modernShow(key){
 if(!byKey.has(key))return;
 active=key;if(S)viewGroup=S.node_roles[key].group;render();detail.replaceChildren();detail.scrollTop=0;
 const n=byKey.get(key),e=E.nodes[key],parents=new Map((S?.main_edges||D.relationships.map(r=>({from:r.parent,to:r.child}))).map(r=>[r.to,r.from]));
 const path=[],seen=new Set();let cursor=key;while(cursor&&!seen.has(cursor)){seen.add(cursor);path.unshift(byKey.get(cursor)?.title||cursor);cursor=parents.get(cursor)}
 detail.append(el('p',path.join(' › '),'breadcrumbs'),el('h2',n.title),el('span',status(n),'badge'));sectionIndex(nodeSections,'node');
 readerHeading(0,nodeSections,'node');sourceList(key);
 readerHeading(1,nodeSections,'node');detail.append(el('p',P.nodes[key]?.rationale||e.rationale));
 readerHeading(2,nodeSections,'node');inputRequirements(key);captureFold('本例模型选择与备选说明',()=>modelChoice(key));
 readerHeading(3,nodeSections,'node');nodeObservations(key);
 readerHeading(4,nodeSections,'node');interpretationCards(key);
 readerHeading(5,nodeSections,'node');dependencyDetails(key);
}
if(modernReader){
 overview=modernOverview;show=modernShow;
 document.getElementById('overview').textContent='病例总览';
 document.getElementById('criteria').textContent='根节点依据';
 document.getElementById('detail').setAttribute('data-layout-version',READER_VERSION);
}
