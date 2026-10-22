/* Karhutla Intelligence — browser-only educational prototype.
   No API keys are used or stored in this public client-side code. */
(() => {
'use strict';
const $=id=>document.getElementById(id);
const center=[-3.45,105.35], bounds=[[-4.75,104.05],[-2.35,106.55]];
const zoneNames=['Air Sugihan','Tulung Selapan','Pampangan','Pangkalan Lampam','Cengal','Pedamaran'];
const zoneCenters=[[-3.35,105.45],[-3.05,105.75],[-3.05,104.95],[-3.55,105.1],[-3.0,106.05],[-3.55,104.75]];
let seed=42, N=64, env={}, risk=[], hotspots=[], scores={}, models={}, activeModel='Random Forest';
let maps={}, overlays={}, markers=[], fireState=[], fireHistory=[], fireStep=0, fireTimer=null, smokeTimer=null, smokeStep=0, smokeHistory=[], activeLayer='hotspots';
const directionVectors={'Tenggara':[.72,.72],'Timur':[0,1],'Barat':[0,-1],'Utara':[-1,0],'Selatan':[1,0]};
const featureNames=['rainfall','lst','ndvi','moisture','peat','wind','enso','iod'];
function rngFactory(s){let a=(s>>>0)||1;return()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return((t^(t>>>14))>>>0)/4294967296;};}
function randn(r){return Math.sqrt(-2*Math.log(Math.max(r(),1e-9)))*Math.cos(2*Math.PI*r());}
function clamp(x,a,b){return Math.max(a,Math.min(b,x));}
function sigmoid(x){return 1/(1+Math.exp(-clamp(x,-20,20)));}
function toast(msg){const t=$('toast');t.textContent=msg;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2500);}
function makeEnv(){
 const r=rngFactory(seed),fields={};featureNames.forEach(k=>fields[k]=new Float32Array(N*N));let enso=randn(r)*.4+.3,iod=randn(r)*.45+.1;
 for(let y=0;y<N;y++)for(let x=0;x<N;x++){let i=y*N+x,xn=x/(N-1),yn=y/(N-1);
  const peat=(Math.sin(xn*5)+Math.cos(yn*4)+randn(r)*.5>.15)?1:0;
  let ndvi=clamp(.68-.22*xn+.11*Math.sin(yn*7)+randn(r)*.07,.05,.95);
  let rain=clamp(220-90*enso-40*iod+40*Math.sin(yn*5+xn*2)+randn(r)*26,8,420);
  let lst=clamp(28+7*enso+2.5*(1-ndvi)+randn(r)*1.6,22,44);
  let moisture=clamp(.73-rain/650+.15*ndvi-.1*peat+randn(r)*.07,.03,.9);
  fields.rainfall[i]=rain;fields.lst[i]=lst;fields.ndvi[i]=ndvi;fields.moisture[i]=moisture;fields.peat[i]=peat;fields.wind[i]=clamp(-Math.log(Math.max(r(),.01))*2,0.1,12);fields.enso[i]=enso+randn(r)*.12;fields.iod[i]=iod+randn(r)*.12;
 }
 return fields;
}
function rowAt(i){return featureNames.map(k=>env[k][i]);}
function labelProb(v){let [rain,lst,ndvi,moist,peat,wind,enso,iod]=v;let climate=$('scenario')?.value||'normal';let add=climate==='dry'?.4:climate==='enso'?.75:0;return sigmoid(-2+.075*(lst-32)-.009*(rain-160)+1.25*peat+1.4*(.42-moist)+.75*(1-ndvi)+.12*wind+.55*enso+.25*iod+add);}
function trainingData(){
 const r=rngFactory(seed+77),rows=[],ys=[],n=1400;
 for(let i=0;i<n;i++){let v=[10+r()*390,24+r()*19,.05+r()*.9,.03+r()*.87,r()<.35?1:0,.1+r()*11.9,randn(r)*.8,randn(r)*.8];rows.push(v);ys.push(r()<labelProb(v)?1:0);}
 const ids=rows.map((_,i)=>i),sh=rngFactory(seed+12);for(let i=ids.length-1;i>0;i--){let j=Math.floor(sh()*(i+1));[ids[i],ids[j]]=[ids[j],ids[i]];}return{rows,ys,tr:ids.slice(0,1050),te:ids.slice(1050)};
}
function gini(ys,ids){let p=ids.reduce((s,i)=>s+ys[i],0)/(ids.length||1);return 2*p*(1-p);}
function buildTree(ds,ids,depth,r,all){
 const pos=ids.reduce((s,i)=>s+ds.ys[i],0),p=(pos+1)/(ids.length+2);
 if(depth>=4||ids.length<20||p<.02||p>.98)return{p};
 let cols=all.slice();for(let i=cols.length-1;i>0;i--){let j=Math.floor(r()*(i+1));[cols[i],cols[j]]=[cols[j],cols[i]];}cols=cols.slice(0,4);
 let best=null,bgain=0,base=gini(ds.ys,ids);
 for(const f of cols){const arr=ids.map(i=>[ds.rows[i][f],i]).sort((a,b)=>a[0]-b[0]);for(let k=10;k<arr.length-10;k+=Math.max(1,Math.floor(arr.length/15))){if(arr[k][0]===arr[k-1][0])continue;let th=(arr[k][0]+arr[k-1][0])/2,l=[],rr=[];for(const i of ids)(ds.rows[i][f]<=th?l:rr).push(i);let gain=base-l.length/ids.length*gini(ds.ys,l)-rr.length/ids.length*gini(ds.ys,rr);if(gain>bgain){bgain=gain;best={f,th,l,rr};}}}
 if(!best)return{p};return{f:best.f,th:best.th,l:buildTree(ds,best.l,depth+1,r,all),r:buildTree(ds,best.rr,depth+1,r,all)};
}
function treePredict(t,v){while(t.f!==undefined)t=v[t.f]<=t.th?t.l:t.r;return t.p;}
function fitRF(ds){let r=rngFactory(seed+2),trees=[],all=featureNames.map((_,i)=>i);for(let k=0;k<9;k++){let ids=[];for(let i=0;i<ds.tr.length;i++)ids.push(ds.tr[Math.floor(r()*ds.tr.length)]);trees.push(buildTree(ds,ids,0,r,all));}return{predict:v=>trees.reduce((s,t)=>s+treePredict(t,v),0)/trees.length};}
function fitBoost(ds){let bias=0;ds.tr.forEach(i=>bias+=ds.ys[i]);bias=Math.log((bias+1)/(ds.tr.length-bias+1));let pred=new Map(ds.tr.map(i=>[i,bias])),stumps=[],lr=.2;
 for(let it=0;it<18;it++){let best={gain:-1};for(let f=0;f<featureNames.length;f++){let vals=ds.tr.map(i=>ds.rows[i][f]).sort((a,b)=>a-b);for(let q=2;q<10;q++){let th=vals[Math.floor(q/10*vals.length)],gl=0,gr=0,nl=0,nr=0;for(const i of ds.tr){let res=ds.ys[i]-sigmoid(pred.get(i));if(ds.rows[i][f]<=th){gl+=res;nl++;}else{gr+=res;nr++;}}if(!nl||!nr)continue;let gain=gl*gl/nl+gr*gr/nr;if(gain>best.gain)best={gain,f,th,left:gl/nl,right:gr/nr};}}
 if(best.gain<0)break;stumps.push(best);for(const i of ds.tr)pred.set(i,pred.get(i)+lr*(ds.rows[i][best.f]<=best.th?best.left:best.right));}
 return{predict:v=>sigmoid(bias+stumps.reduce((s,t)=>s+lr*(v[t.f]<=t.th?t.left:t.right),0))};}
function fitSVM(ds){let means=featureNames.map((_,f)=>ds.tr.reduce((s,i)=>s+ds.rows[i][f],0)/ds.tr.length),std=featureNames.map((_,f)=>Math.sqrt(ds.tr.reduce((s,i)=>s+(ds.rows[i][f]-means[f])**2,0)/ds.tr.length)||1),w=featureNames.map(()=>0),b=0,r=rngFactory(seed+99);
 for(let ep=0;ep<9;ep++){let ids=ds.tr.slice();for(let i=ids.length-1;i>0;i--){let j=Math.floor(r()*(i+1));[ids[i],ids[j]]=[ids[j],ids[i]];}for(const id of ids){let x=ds.rows[id].map((v,f)=>(v-means[f])/std[f]),y=ds.ys[id]?1:-1,margin=y*(w.reduce((s,z,f)=>s+z*x[f],0)+b),eta=.02/(1+ep*.2);w=w.map((z,f)=>z*(1-eta*.002)+(margin<1?eta*y*x[f]:0));if(margin<1)b+=eta*y;}}
 return{predict:v=>sigmoid(w.reduce((s,z,f)=>s+z*(v[f]-means[f])/std[f],0)+b),classify:v=>w.reduce((s,z,f)=>s+z*(v[f]-means[f])/std[f],0)+b>0?1:0};}
function evaluate(model,ds,svm=false){let tp=0,tn=0,fp=0,fn=0,ps=[],ys=[];for(const i of ds.te){let p=model.predict(ds.rows[i]),y=ds.ys[i],pr=svm?model.classify(ds.rows[i]):p>=.5?1:0;ps.push(p);ys.push(y);if(pr&&y)tp++;else if(pr&&!y)fp++;else if(!pr&&y)fn++;else tn++;}let pos=ps.filter((_,i)=>ys[i]),neg=ps.filter((_,i)=>!ys[i]),win=0,tie=0;for(let p of pos)for(let n of neg){if(p>n)win++;else if(p===n)tie++;}return{'ROC-AUC':pos.length&&neg.length?(win+tie*.5)/(pos.length*neg.length):.5,Accuracy:(tp+tn)/ds.te.length,F1:2*tp/(2*tp+fp+fn||1)};}
function trainModels(){let ds=trainingData();models={'Random Forest':fitRF(ds),'Gradient Boosting':fitBoost(ds),'Linear SVM':fitSVM(ds)};scores={};for(const name in models)scores[name]=evaluate(models[name],ds,name==='Linear SVM');activeModel=$('modelSelect')?.value||'Random Forest';risk=Array.from({length:N*N},(_,i)=>models[activeModel].predict(rowAt(i)));renderModel();}
function makeHotspots(){let r=rngFactory(seed+888),count=48;hotspots=[];for(let i=0;i<count;i++){let zone=i%zoneCenters.length,c=zoneCenters[zone],lat=c[0]+(r()-.5)*.36,lon=c[1]+(r()-.5)*.4;hotspots.push({id:`SIM-${String(i+1).padStart(3,'0')}`,lat,lon,confidence:Math.round(70+r()*30),zone:zoneNames[zone],hours:Math.floor(r()*24),frp:(.3+r()*18).toFixed(1),synthetic:true});}}
function makeMap(id){const m=L.map(id,{zoomControl:true,preferCanvas:true}).setView(center,8);const osm=L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:18,attribution:'&copy; OpenStreetMap contributors'});const topo=L.tileLayer('https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',{maxZoom:17,attribution:'Map data: &copy; OpenStreetMap contributors, SRTM | Style: OpenTopoMap'});osm.addTo(m);return{map:m,base:{osm,topo}};}
function addLayers(id){addRaster(id,'risk');addRaster(id,'fire');addRaster(id,'smoke');switchLayer('hotspots');}
function initMaps(){for(const id of ['map','hotspotMap','fireMap','smokeMap','riskMap']){maps[id]=makeMap(id);if(id==='map'||id==='hotspotMap')addHotspotMarkers(id);if(id==='map')addLayers(id);if(id==='riskMap')addRaster(id,'risk');if(id==='fireMap')addRaster(id,'fire');if(id==='smokeMap')addRaster(id,'smoke');}
 setTimeout(()=>Object.values(maps).forEach(o=>o.map.invalidateSize()),250);}
function iconFor(cls){return L.divIcon({className:'',html:`<div class="${cls}"></div>`,iconSize:[14,14],iconAnchor:[7,7]});}
function addHotspotMarkers(id){const group=L.layerGroup().addTo(maps[id].map);hotspots.forEach(h=>L.marker([h.lat,h.lon],{icon:iconFor('hotspot-marker')}).bindPopup(`<div class="popup-title">${h.id} · Hotspot sintetis</div><div class="popup-row">Confidence simulasi: <b>${h.confidence}%</b></div><div class="popup-row">Zona: ${h.zone}</div><div class="popup-row">FRP relatif: ${h.frp} MW (sintetis)</div><div class="popup-row">Waktu simulasi: ${h.hours} jam lalu</div>`).addTo(group));if(id==='map')overlays.hotspots=group;else overlays.hotspotMap=group;}
function canvasRaster(type,step=0){const size=240,c=document.createElement('canvas');c.width=c.height=size;const ctx=c.getContext('2d'),im=ctx.createImageData(size,size),fire=fireHistory[Math.min(step,fireHistory.length-1)]||new Uint8Array(N*N);
 let smoke=new Float32Array(size*size);
 if(type==='smoke'){const vec=directionVectors[$('smokeDirection')?.value||'Tenggara'],spd=+$('smokeSpeedSelect')?.value||2,em=+$('emission')?.value||.7;for(let sy=0;sy<size;sy++)for(let sx=0;sx<size;sx++){let val=0;for(const h of hotspots){const fy=(h.lat-bounds[0][0])/(bounds[1][0]-bounds[0][0])*size,fx=(h.lon-bounds[0][1])/(bounds[1][1]-bounds[0][1])*size;let sourceX=fx-vec[1]*step*spd*1.25,sourceY=fy+vec[0]*step*spd*1.25;let dx=sx-sourceX,dy=sy-sourceY;let along=dx*vec[1]+dy*(-vec[0]),cross=Math.abs(dx*(-vec[0])+dy*vec[1]);let sigma=5+step*.42;let plume=Math.exp(-Math.pow(cross/(sigma+along*.035),2))*Math.exp(-Math.max(0,-along)/8)*Math.exp(-Math.max(0,along)/(24+step*1.4));val=Math.max(val,plume*(h.confidence/100)*em);}smoke[sy*size+sx]=val;}}
 for(let y=0;y<size;y++)for(let x=0;x<size;x++){let i=y*size+x,idx=(y*size+x)*4,a=0,rr=0,gg=0,bb=0;
 if(type==='risk'){let gx=Math.floor(x/size*N),gy=Math.floor(y/size*N),v=risk[gy*N+gx]||0;let cols=[[247,235,173],[243,164,76],[213,83,54],[154,43,47]],z=v*3,k=Math.min(2,Math.floor(z)),f=z-k;rr=cols[k][0]*(1-f)+cols[k+1][0]*f;gg=cols[k][1]*(1-f)+cols[k+1][1]*f;bb=cols[k][2]*(1-f)+cols[k+1][2]*f;a=170;}
 else if(type==='fire'){let gx=Math.floor(x/size*N),gy=Math.floor(y/size*N),v=fire[gy*N+gx]||0;if(v){rr=v===1?255:190;gg=v===1?179:47;bb=v===1?48:36;a=v===1?225:185;}}
 else {let v=smoke[i];rr=95;gg=105;bb=105;a=Math.round(clamp(v,0,1)*150);}
 im.data[idx]=rr;im.data[idx+1]=gg;im.data[idx+2]=bb;im.data[idx+3]=a;
 }
 ctx.putImageData(im,0,0);return c.toDataURL('image/png');}
function addRaster(id,type){const overlay=L.imageOverlay(canvasRaster(type,0),bounds,{opacity:type==='smoke'?.9:type==='risk'?.7:.85,interactive:false});overlay.addTo(maps[id].map);overlays[type+'_'+id]=overlay;if(id==='map')overlays[type]=overlay;}
function refreshRaster(type,step=0){const url=canvasRaster(type,step);for(const id of ['map',type==='fire'?'fireMap':type==='smoke'?'smokeMap':'riskMap']){const o=overlays[type+'_'+id];if(o)o.setUrl(url);}}
function switchLayer(layer){activeLayer=layer;document.querySelectorAll('#layerSwitch button').forEach(b=>b.classList.toggle('selected',b.dataset.layer===layer));const names=['hotspots','risk','fire','smoke'];for(const name of names){const o=overlays[name];if(o){if(name===layer)maps.map.map.addLayer(o);else maps.map.map.removeLayer(o);}}if(layer==='hotspots'&&overlays.hotspots)overlays.hotspots.addTo(maps.map.map);if(layer!=='hotspots'&&overlays.hotspots)maps.map.map.removeLayer(overlays.hotspots);if(layer==='risk')refreshRaster('risk');if(layer==='fire')refreshRaster('fire',fireStep);if(layer==='smoke')refreshRaster('smoke',smokeStep);$('legend').innerHTML=`<b>Legenda</b>${layer==='hotspots'?'<span><i class="legend-dot hotspot"></i>Hotspot sintetis</span>':layer==='risk'?'<span><i class="legend-gradient"></i>Kerawanan relatif</span>':layer==='fire'?'<span><i class="legend-dot fire-active"></i>Api aktif</span><span><i class="legend-dot fire-burned"></i>Area terbakar</span>':'<span><i class="legend-smoke"></i>Plume asap relatif</span>'}`;}
function renderModel(){let s=scores[activeModel]||{'ROC-AUC':.7};$('aucValue').textContent=s['ROC-AUC'].toFixed(3);$('miniBars').innerHTML=Object.keys(scores).map(n=>`<i title="${n}" style="height:${Math.max(7,scores[n]['ROC-AUC']*35)}px;${n===activeModel?'background:#1d6048':'background:#a9c9b4'}"></i>`).join('');}
function renderKPIs(){ $('kpiHotspots').textContent=hotspots.length;$('kpiRisk').textContent=(100*risk.filter(x=>x>.65).length/risk.length).toFixed(1)+'%';$('kpiBurned').textContent=(fireHistory[fireStep]||[]).filter(v=>v>0).length+' sel';$('kpiWind').textContent=$('fireDirection').value;}
function renderZones(){const counts=zoneNames.map(z=>hotspots.filter(h=>h.zone===z).length),max=Math.max(...counts,1);$('zoneBars').innerHTML=zoneNames.map((z,i)=>`<div class="zone-row"><span>${z}</span><div class="bar-track"><div class="bar-fill" style="width:${counts[i]/max*100}%"></div></div><b>${counts[i]}</b></div>`).join('');$('timeline').innerHTML=[['T−24 jam','Inisialisasi'],['T−12 jam','Pemicu'],['T sekarang','Simulasi']].map((x,i)=>`<div class="timeline-step"><b>${x[0]}</b><small>${x[1]}</small></div>`).join('');}
function renderTable(){const q=($('hotspotSearch')?.value||'').toLowerCase();const rows=hotspots.filter(h=>`${h.id} ${h.zone} ${h.confidence}`.toLowerCase().includes(q));$('hotspotTable').innerHTML=rows.map(h=>`<tr><td><b>${h.id}</b></td><td>${h.lat.toFixed(5)}</td><td>${h.lon.toFixed(5)}</td><td>${h.confidence}%</td><td>${h.zone}</td><td>${h.hours} jam lalu</td></tr>`).join('');}
function makeFire(){stopFire();const r=rngFactory(seed+333),state=new Uint8Array(N*N),starts=[];const candidates=hotspots.slice().sort((a,b)=>b.confidence-a.confidence).slice(0,Math.max(1,+$('ignitionCount').value));for(let h of candidates){let x=clamp(Math.floor((h.lon-bounds[0][1])/(bounds[1][1]-bounds[0][1])*N),0,N-1),y=clamp(Math.floor((bounds[1][0]-h.lat)/(bounds[1][0]-bounds[0][0])*N),0,N-1),idx=y*N+x;state[idx]=1;starts.push(idx);}let hist=[state.slice()],cur=state,vec=directionVectors[$('fireDirection')?.value||'Tenggara'],steps=+$('fireSteps')?.value||35,wind=+$('windSpeed')?.value||18,moist=+$('fuelMoisture')?.value||35;
 for(let t=0;t<steps;t++){let nxt=cur.slice();for(let i=0;i<cur.length;i++)if(cur[i]===1){nxt[i]=2;let y=Math.floor(i/N),x=i%N;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;let yy=y+dy,xx=x+dx;if(yy<0||yy>=N||xx<0||xx>=N)continue;let j=yy*N+xx;if(cur[j])continue;let p=.04+.46*risk[j]+.28*(1-moist/100)+wind/1300;let along=dx*vec[1]+dy*(-vec[0]);p*=along>.5?1.5:along<-.5?.65:1;if(r()<clamp(p,.02,.9))nxt[j]=1;}}cur=nxt;hist.push(cur.slice());if(!cur.includes(1)){while(hist.length<=steps)hist.push(cur.slice());break;}}
 fireHistory=hist;fireStep=0;renderFire();}
function renderFire(){let arr=fireHistory[fireStep]||new Uint8Array(N*N);$('fireStep').textContent='LANGKAH '+fireStep;let burned=arr.filter(v=>v===2).length,active=arr.filter(v=>v===1).length;$('fireMetrics').innerHTML=[['SEL TERBAKAR',burned],['API AKTIF',active],['TOTAL TERDAMPAK',burned+active],['LANGKAH',fireStep]].map(a=>`<div class="metric-card"><small>${a[0]}</small><strong>${a[1]}</strong></div>`).join('');$('kpiBurned').textContent=(burned+active)+' sel';refreshRaster('fire',fireStep);}
function stopFire(){if(fireTimer)clearInterval(fireTimer);fireTimer=null;if($('playFire'))$('playFire').textContent='▶ Putar animasi';}
function playFire(){if(fireTimer){stopFire();return;}if(fireStep>=fireHistory.length-1)fireStep=0;$('playFire').textContent='Ⅱ Jeda';fireTimer=setInterval(()=>{fireStep++;if(fireStep>=fireHistory.length){stopFire();fireStep=fireHistory.length-1;}renderFire();},250);}
function makeSmokeHistory(){const steps=48,vec=directionVectors[$('smokeDirection')?.value||'Tenggara'],speed=+$('smokeSpeedSelect')?.value||2,em=+$('emission')?.value||.7;let cur=new Float32Array(N*N);for(const h of hotspots){let x=Math.floor((h.lon-bounds[0][1])/(bounds[1][1]-bounds[0][1])*N),y=Math.floor((bounds[1][0]-h.lat)/(bounds[1][0]-bounds[0][0])*N);for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){let xx=x+dx,yy=y+dy;if(xx>=0&&xx<N&&yy>=0&&yy<N)cur[yy*N+xx]=Math.max(cur[yy*N+xx],em*(h.confidence/100)*Math.exp(-(dx*dx+dy*dy)/2));}}
 let hist=[cur.slice()];for(let t=0;t<steps;t++){let nxt=new Float32Array(N*N);for(let y=0;y<N;y++)for(let x=0;x<N;x++){let i=y*N+x,srcX=Math.round(x-vec[1]*speed),srcY=Math.round(y+vec[0]*speed),adv=srcX>=0&&srcX<N&&srcY>=0&&srcY<N?cur[srcY*N+srcX]*.83:0,local=cur[i]*.5,dif=0,c=0;for(const[dy,dx]of [[-1,0],[1,0],[0,-1],[0,1]]){let yy=y+dy,xx=x+dx;if(yy>=0&&yy<N&&xx>=0&&xx<N){dif+=cur[yy*N+xx];c++;}}nxt[i]=clamp(Math.max(adv,local,c?dif/c*.13:0),0,1);}cur=nxt;hist.push(cur.slice());}return hist;}
function renderSmoke(){ $('smokeStep').textContent='LANGKAH '+smokeStep;refreshRaster('smoke',smokeStep);}
function stopSmoke(){if(smokeTimer)clearInterval(smokeTimer);smokeTimer=null;if($('playSmoke'))$('playSmoke').textContent='▶ Putar plume';}
function playSmoke(){if(smokeTimer){stopSmoke();return;}if(smokeStep>=smokeHistory.length-1)smokeStep=0;$('playSmoke').textContent='Ⅱ Jeda';smokeTimer=setInterval(()=>{smokeStep++;if(smokeStep>=smokeHistory.length){stopSmoke();smokeStep=smokeHistory.length-1;}renderSmoke();},240);}
function changeBasemap(val){const base=val==='topo'?'topo':'osm';for(const obj of Object.values(maps)){obj.base.osm.remove();obj.base.topo.remove();obj.base[base].addTo(obj.map);}}
function updateScenario(){makeEnv();trainModels();makeFire();smokeHistory=makeSmokeHistory();smokeStep=0;renderSmoke();renderKPIs();renderZones();renderLayerSafe();toast('Skenario diperbarui menggunakan data sintetis.');}
function renderLayerSafe(){if(overlays.risk)refreshRaster('risk');if(overlays.fire)refreshRaster('fire',fireStep);if(overlays.smoke)refreshRaster('smoke',smokeStep);}
function downloadCSV(){const header='id,latitude,longitude,confidence,zone,hours_ago,frp_relative,synthetic';const lines=hotspots.map(h=>[h.id,h.lat,h.lon,h.confidence,h.zone,h.hours,h.frp,true].join(','));const blob=new Blob([[header,...lines].join('\n')],{type:'text/csv;charset=utf-8;'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='hotspot-sintetis-oki.csv';a.click();URL.revokeObjectURL(url);toast('CSV hotspot sintetis diunduh.');}
function setUpNavigation(){document.querySelectorAll('.nav-item').forEach(b=>b.addEventListener('click',()=>{const view=b.dataset.view;document.querySelectorAll('.nav-item').forEach(x=>x.classList.toggle('active',x===b));document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id==='view-'+view));if(innerWidth<950)$('sidebar')?.classList.remove('open');setTimeout(()=>Object.values(maps).forEach(o=>o.map.invalidateSize()),100);}));}
function init(){
 if(typeof L==='undefined'){document.querySelectorAll('.map-panel').forEach(panel=>{const el=panel.querySelector('#map, #hotspotMap, #fireMap, #smokeMap, #riskMap');if(el){el.innerHTML='<div class=\"map-error\"><b>Peta belum dapat dimuat.</b><br>Periksa koneksi internet atau pemblokiran CDN, lalu muat ulang halaman.</div>';}});return;}
 $('clock').textContent=new Intl.DateTimeFormat('id-ID',{dateStyle:'medium',timeStyle:'short'}).format(new Date());
 $('menuToggle').addEventListener('click',()=>document.querySelector('.sidebar').classList.toggle('open'));
 $('dismissAlert').addEventListener('click',()=>$('dismissAlert').parentElement.remove());
 $('helpBtn').addEventListener('click',()=>toast('Demo sintetis. Data aktual FIRMS belum tersambung ke website publik.'));
 $('layerSwitch').addEventListener('click',e=>{let b=e.target.closest('button[data-layer]');if(b)switchLayer(b.dataset.layer);});
 $('basemap').addEventListener('change',e=>changeBasemap(e.target.value));
 $('resetMap').addEventListener('click',()=>maps.map.map.fitBounds(bounds));
 $('modelSelect').addEventListener('change',()=>{activeModel=$('modelSelect').value;trainModels();risk=Array.from({length:N*N},(_,i)=>models[activeModel].predict(rowAt(i)));renderKPIs();renderLayerSafe();});
 $('riskModel').addEventListener('change',()=>{activeModel=$('riskModel').value;$('modelSelect').value=activeModel;trainModels();risk=Array.from({length:N*N},(_,i)=>models[activeModel].predict(rowAt(i)));refreshRaster('risk');});
 $('runScenario').addEventListener('click',updateScenario);
 $('scenario').addEventListener('change',updateScenario);
 $('windSpeed').addEventListener('input',()=>{$('windSpeedValue').textContent=$('windSpeed').value;});
 $('fuelMoisture').addEventListener('input',()=>{$('fuelValue').textContent=$('fuelMoisture').value;});
 $('ignitionCount').addEventListener('input',()=>{$('ignitionValue').textContent=$('ignitionCount').value;});
 $('playFire').addEventListener('click',playFire);$('resetFire').addEventListener('click',()=>{stopFire();makeFire();});
 $('fireDirection').addEventListener('change',()=>{makeFire();$('kpiWind').textContent=$('fireDirection').value;});
 $('fireSteps').addEventListener('change',makeFire);
 $('playSmoke').addEventListener('click',playSmoke);
 const resetSmoke=()=>{stopSmoke();smokeHistory=makeSmokeHistory();smokeStep=0;renderSmoke();};$('resetSmoke').addEventListener('click',resetSmoke);
 ['smokeDirection','smokeSpeedSelect','emission'].forEach(id=>$(id).addEventListener('change',resetSmoke));
 $('hotspotSearch').addEventListener('input',renderTable);$('exportHotspots').addEventListener('click',downloadCSV);
 $('seed')?.addEventListener('change',()=>{seed=+$('seed').value||42;updateScenario();});
 setUpNavigation();
 makeEnv();trainModels();risk=Array.from({length:N*N},(_,i)=>models[activeModel].predict(rowAt(i)));makeHotspots();makeFire();smokeHistory=makeSmokeHistory();initMaps();renderKPIs();renderZones();renderTable();renderModel();
}
function renderLayer(){switchLayer(activeLayer);}
window.addEventListener('load',init);
})();