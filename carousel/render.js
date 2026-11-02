const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1080,height:1350}});
await p.goto('file://'+__dirname+'/slides.html',{waitUntil:'networkidle'});await p.evaluate(()=>document.fonts.ready);
const s=await p.$$('section');for(let i=0;i<s.length;i++)await s[i].screenshot({path:`slide_${i+1}.png`});await b.close();})();
