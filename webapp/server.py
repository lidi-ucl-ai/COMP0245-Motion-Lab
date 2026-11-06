"""Local-only interactive learned-policy simulator. Standard-library HTTP/SSE server."""
import argparse, json, math, mimetypes, os, queue, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).parent/'cache'/'matplotlib'))
sys.path.insert(0, str(ROOT))
from policies import MODELS, load_policy, reset_for_policy
from env import RobotEnv, ASSETS
from monitoring import StateMonitor
STATIC=Path(__file__).parent/'static'
OUTPUT=Path(__file__).parent/'runs'; OUTPUT.mkdir(exist_ok=True)
DEFAULT=dict(target=[0.,0.,1.2],base=[.06659161,.06211989,0],
    arm=[.02977991,1.53119626,.01796166,.9663676,-.02630708,1.27137698,.03898663],duration=30,model='torque')
def finite_vector(value,n,name):
    a=np.asarray(value,dtype=float)
    if a.shape!=(n,) or not np.isfinite(a).all():raise ValueError(f'{name} must contain {n} finite numbers')
    return a

class Engine:
    def __init__(self):
        self.commands=queue.Queue();self.lock=threading.Lock();self.latest={};self.ready=threading.Event()
        threading.Thread(target=self.work,daemon=True).start()
        if not self.ready.wait(60):raise RuntimeError('Simulator failed to initialise')
        if getattr(self,'fatal',None):raise RuntimeError(self.fatal)
    def command(self,action,data):
        response=queue.Queue();self.commands.put((action,data,response))
        result=response.get(timeout=30)
        if 'error' in result and 'seq' not in result:raise ValueError(result['error'])
        return result
    def snapshot(self):
        with self.lock:return self.latest
    def poses(self):
        states=self.e.client.getLinkStates(self.e.rid,self.links,computeForwardKinematics=True)
        return {str(i):[list(s[4]),list(s[5])] for i,s in zip(self.links,states)}
    def publish(self):
        st=self.e.state
        self.seq+=1
        row=dict(seq=self.seq,run_id=self.run_id,status=self.status,time=self.e.k*self.e.dt,
            error=float(np.linalg.norm(self.e.target-st['ee'])),target=self.e.target.tolist(),ee=st['ee'].tolist(),
            base=st['pos'].tolist(),arm=st['q'][4:11].tolist(),poses=self.poses(),
            phase=int(self.policy.navigation.phase%4) if getattr(self.policy,'navigation',None) is not None else int(getattr(self.policy,'phase',getattr(getattr(self.policy,'base_memory',None),'phase',-1))),success=self.info.get('success',False),reason=self.info.get('reason','ready'),
            domain_selection=getattr(self.policy,'log',[]),
            model=self.config['model'],wall_elapsed=round(time.perf_counter()-self.start_wall,3),
            checkpoint_sha256=self.digest,
            telemetry=self.monitor.capture(self.e.k*self.e.dt,st['ee'],
                self.e.client.getBaseVelocity(self.e.rid)[0],self.e.prev_action,self.e.held*self.e.dt))
        with self.lock:self.latest=row
        if self.status=='running' or self.info.get('done'):self.trace.append(row)
    def configure(self,data):
        if self.status in ('running','paused'):raise ValueError('Stop the active run before changing its initial state.')
        cfg={**DEFAULT,**data}
        target=finite_vector(cfg['target'],3,'Target');base=finite_vector(cfg['base'],3,'Base x/y/yaw');arm=finite_vector(cfg['arm'],7,'Joint angles')
        duration=float(cfg['duration'])
        if not math.isfinite(duration) or not .5<=duration<=600:raise ValueError('Duration must be between 0.5 and 600 seconds.')
        if not 0<=target[2]<=2:raise ValueError('Target height must be between 0 and 2 m.')
        if np.linalg.norm(base[:2])>4:raise ValueError('Start the base within 4 m of the origin.')
        lo=self.e.lower[4:11];hi=self.e.upper[4:11];limited=lo<hi
        if np.any(limited&((arm<lo)|(arm>hi))):raise ValueError('Joint angles exceed URDF limits. Check J2, J4 and J6.')
        if cfg['model'] not in MODELS:raise ValueError('Unknown model')
        nominal=self.e.default_q[4:11].copy();nominal[5]=1.25
        self.run_id=uuid.uuid4().hex[:12];self.config=cfg
        self.scenario=dict(id=self.run_id,seed=20261025,target=target.tolist(),
            base=[*(base[:2]-np.asarray(self.e.default_pos)[:2]),base[2]],arm_offset=(arm-nominal).tolist(),duration=duration)
        self.obs=reset_for_policy(cfg['model'], self.e, self.scenario)
        self.policy = load_policy(cfg['model'], self.e, self.obs)
        assert np.allclose(self.e.init_q,arm,atol=1e-9,rtol=0)
        self.digest=MODELS[cfg['model']].get('sha256');self.trace=[];self.info={};self.status='ready';self.start_wall=time.perf_counter();self.monitor=StateMonitor()
        self.publish()
    def save(self):
        record=dict(config=self.config,scenario=self.scenario,result=self.snapshot(),samples=self.trace,
            notes='Fresh measured physics states; selected report policy or explicitly selected expert; nominal 25 Hz samples')
        (OUTPUT/(self.run_id+'.json')).write_text(json.dumps(record),encoding='utf-8')
    def work(self):
        try:
            self.e=RobotEnv();self.seq=0;self.status='initialising';self.policy=None
            self.visuals=[]
            for v in self.e.client.getVisualShapeData(self.e.rid):
                if v[1]<0:continue
                filename=v[4].decode() if isinstance(v[4],bytes) else v[4]
                asset=Path(filename).resolve().relative_to(ASSETS.resolve()).as_posix() if filename else None
                self.visuals.append(dict(link=v[1],kind=v[2],dimensions=v[3],asset=asset,position=v[5],quaternion=v[6],color=v[7]))
            self.links=sorted(set(v['link'] for v in self.visuals))
            self.configure(DEFAULT)
        except Exception as exc:
            self.fatal=str(exc);self.ready.set();return
        self.ready.set();deadline=time.perf_counter()
        while True:
            try:
                action,data,response=self.commands.get(timeout=0 if self.status=='running' else .04)
                try:
                    if action=='preview':self.configure(data)
                    elif action=='start':
                        self.configure(data);self.status='running';self.start_wall=time.perf_counter();deadline=self.start_wall;self.publish()
                    elif action=='pause':
                        if self.status!='running':raise ValueError('No active run to pause')
                        self.status='paused';self.publish()
                    elif action=='resume':
                        if self.status!='paused':raise ValueError('No paused run to resume')
                        self.status='running';deadline=time.perf_counter();self.publish()
                    elif action=='stop':
                        if self.status not in ('running','paused'):raise ValueError('No active run to stop')
                        self.status='stopped';self.info.update(done=True,success=False,reason='user_stop');self.publish();self.save()
                    else:raise ValueError('Unknown action')
                    response.put(self.snapshot())
                except Exception as exc:response.put({'error':str(exc)})
            except queue.Empty:pass
            if self.status!='running':continue
            try:
                for _ in range(40):
                    self.obs,self.info=self.e.step(self.policy.act(self.obs))
                    if self.info['done']:break
                if self.info['done']:self.status='finished'
                self.publish()
                if self.status=='finished':self.save()
                deadline+=.04;time.sleep(max(0,deadline-time.perf_counter()))
                if deadline<time.perf_counter()-.2:deadline=time.perf_counter()
            except Exception as exc:
                self.status='error';self.info=dict(done=True,success=False,reason=str(exc));self.publish();self.save()

class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'
    def log_message(self,fmt,*args):
        if '/api/state' not in str(args):super().log_message(fmt,*args)
    def reply(self,status,data,ctype='application/json'):
        if not isinstance(data,bytes):data=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(status);self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
    def file(self,root,relative):
        path=(root/relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():return self.reply(404,{'error':'Not found'})
        self.reply(200,path.read_bytes(),mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
    def do_GET(self):
        path=unquote(urlsplit(self.path).path)
        if path=='/api/meta':
            return self.reply(200,dict(default=DEFAULT,models=MODELS,visuals=engine.visuals,
                lower=engine.e.lower[4:11].tolist(),upper=engine.e.upper[4:11].tolist(),state=engine.snapshot()))
        if path=='/api/state':return self.reply(200,engine.snapshot())
        if path=='/api/events':
            self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.send_header('Connection','close');self.end_headers();seq=-1
            try:
                while True:
                    state=engine.snapshot()
                    if state.get('seq')!=seq:
                        seq=state['seq'];self.wfile.write(('data: '+json.dumps(state)+'\n\n').encode());self.wfile.flush()
                    time.sleep(.035)
            except (BrokenPipeError,ConnectionResetError):return
        if path.startswith('/assets/'):return self.file(ASSETS,path[8:])
        if path.startswith('/runs/'):return self.file(OUTPUT,path[6:])
        return self.file(STATIC,'index.html' if path=='/' else path.lstrip('/'))
    def do_POST(self):
        try:
            # Only same-origin browser commands can mutate the private simulator.
            origin=self.headers.get('Origin');host=self.headers.get('Host')
            if origin and urlsplit(origin).netloc!=host:return self.reply(403,{'error':'Origin mismatch'})
            path=urlsplit(self.path).path;length=int(self.headers.get('Content-Length','0'))
            if path=='/api/video':
                import subprocess
                if not 1<=length<=80_000_000:raise ValueError('The recording must be under 80 MB.')
                run=self.headers.get('X-Run-Id','')
                if len(run)!=12 or any(c not in '0123456789abcdef' for c in run) or not (OUTPUT/(run+'.json')).exists():raise ValueError('Finish or stop the run before exporting.')
                source=OUTPUT/(run+'.mjpeg');source.write_bytes(self.rfile.read(length));out=OUTPUT/(run+'.mp4')
                result=subprocess.run(['ffmpeg','-y','-v','error','-f','image2pipe','-framerate','25','-vcodec','mjpeg','-i',str(source),'-an','-vf','scale=trunc(iw/2)*2:trunc(ih/2)*2','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],capture_output=True,timeout=90)
                if result.returncode:raise ValueError('Video conversion failed')
                check=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=nb_read_frames,width,height:format=duration','-of','json',str(out)],capture_output=True,text=True,timeout=30)
                validation=json.loads(check.stdout)
                expected=int(self.headers.get('X-Frame-Count','0'))
                if check.returncode or int(validation['streams'][0]['nb_read_frames'])!=expected:raise ValueError('Video frame-count validation failed')
                (OUTPUT/(run+'.video.json')).write_text(json.dumps(dict(validation=validation,text_overlays=False,fps=25,source='Browser WebGL renders of saved measured states; 1x simulation time; no new physics execution'),indent=2))
                source.unlink()
                return self.reply(200,{'url':'/runs/'+out.name,'validation':validation})
            if length>16384:raise ValueError('Request too large')
            data=json.loads(self.rfile.read(length) or b'{}')
            return self.reply(200,engine.command(path.removeprefix('/api/'),data))
        except (ValueError,KeyError,TypeError,queue.Empty) as exc:self.reply(400,{'error':str(exc)})
        except Exception as exc:self.reply(500,{'error':str(exc)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=7870);a=p.parse_args()
    engine=Engine();print(f'COMP0245 live: http://127.0.0.1:{a.port}',flush=True)
    class DemoHTTPServer(ThreadingHTTPServer):
        request_queue_size=128
        daemon_threads=True
    DemoHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()
