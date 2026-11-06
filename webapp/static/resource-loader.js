/** Bounded loading with retries for temporary tunnel failures. */
export function createResourceLoader({fetcher=fetch,wait=ms=>new Promise(r=>setTimeout(r,ms)),attempts=6,concurrency=3,onRetry=()=>{}}={}){
 let active=0;const queued=[];
 async function slot(fn){if(active>=concurrency)await new Promise(resolve=>queued.push(resolve));else active++;try{return await fn();}finally{if(queued.length)queued.shift()();else active--;}}
 async function request(url,kind){return slot(async()=>{
  for(let n=0;n<attempts;n++){
   try{
    const r=await fetcher(url,{credentials:'same-origin',signal:AbortSignal.timeout(30000)});
    if(r.status===401||r.status===403){const e=Error('Access expired. Refresh the page to reconnect.');e.permanent=true;throw e;}
    if(!r.ok){const e=Error('Resource temporarily unavailable.');e.permanent=r.status>=400&&r.status<500&&![408,429].includes(r.status);throw e;}
    return kind==='json'?await r.json():await r.text();
   }catch(e){if(e.permanent||n===attempts-1)throw e;onRetry(n+1);await wait(Math.min(500*2**n,5000));}
  }
 });}
 return {text:url=>request(url,'text'),json:url=>request(url,'json')};
}
