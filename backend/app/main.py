from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from backend.app.rag.service import CandidateRAGService

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_service = CandidateRAGService()
    try:
        yield
    finally:
        await app.state.rag_service.close()


app = FastAPI(
    title="OpenJobs Candidate Screening Agent",
    version="0.3.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(payload: ChatRequest, request: Request) -> dict:
    try:
        result = await request.app.state.rag_service.search(payload.message)
        return result.model_dump()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>OpenJobs 候选人筛选</title>
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/dompurify/dist/purify.min.js"></script>
  <style>
    :root { color-scheme: light; --ink:#162033; --muted:#6b7280; --line:#e5e7eb; --blue:#2563eb; }
    * { box-sizing:border-box; }
    body { margin:0; font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      background:#f5f7fb; color:var(--ink); }
    main { max-width:900px; min-height:100vh; margin:auto; background:white; display:flex;
      flex-direction:column; box-shadow:0 0 40px rgba(15,23,42,.06); }
    header { padding:24px 28px 18px; border-bottom:1px solid var(--line); }
    h1 { font-size:20px; margin:0 0 6px; } header p { color:var(--muted); margin:0; font-size:14px; }
    #messages { flex:1; padding:26px; overflow:auto; }
    .message { max-width:85%; margin:0 0 22px; padding:14px 18px; border-radius:14px; line-height:1.65; }
    .user { margin-left:auto; background:var(--blue); color:white; }
    .assistant { background:#f8fafc; border:1px solid var(--line); }
    .assistant h3 { margin-top:18px; } .assistant h3:first-child { margin-top:0; }
    .meta { color:var(--muted); font-size:12px; margin-top:12px; }
    form { display:flex; gap:10px; padding:18px 24px 24px; border-top:1px solid var(--line); }
    textarea { flex:1; resize:none; min-height:54px; max-height:150px; border:1px solid #cbd5e1;
      border-radius:12px; padding:14px; font:inherit; outline:none; }
    textarea:focus { border-color:var(--blue); box-shadow:0 0 0 3px rgba(37,99,235,.12); }
    button { border:0; border-radius:12px; background:var(--blue); color:white; padding:0 22px;
      font-weight:650; cursor:pointer; } button:disabled { opacity:.55; cursor:wait; }
  </style>
</head>
<body><main>
  <header><h1>OpenJobs 候选人筛选 Agent</h1><p>描述岗位、硬性要求和优先条件，我会给出 Top 5 与推荐理由。</p></header>
  <section id="messages"><div class="message assistant">你好，请告诉我你想找什么样的候选人。</div></section>
  <form id="form"><textarea id="input" placeholder="例如：找有 5 年以上经验的 Python 后端工程师，熟悉云平台优先"></textarea><button>发送</button></form>
</main>
<script>
const form=document.querySelector('#form'), input=document.querySelector('#input'),
  messages=document.querySelector('#messages'), button=form.querySelector('button');
function add(content,role,markdown=false){
  const el=document.createElement('div'); el.className='message '+role;
  el.innerHTML=markdown?DOMPurify.sanitize(marked.parse(content)):content.replaceAll('<','&lt;').replaceAll('>','&gt;');
  messages.appendChild(el); el.scrollIntoView({behavior:'smooth',block:'end'}); return el;
}
form.addEventListener('submit',async(e)=>{
  e.preventDefault(); const text=input.value.trim(); if(!text)return;
  add(text,'user'); input.value=''; button.disabled=true;
  const loading=add('正在检索和分析…','assistant');
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:text})});
    const data=await res.json(); if(!res.ok)throw new Error(data.detail||'请求失败');
    loading.innerHTML=DOMPurify.sanitize(marked.parse(data.answer));
    const p=data.parsed_query, meta=document.createElement('div'); meta.className='meta';
    meta.textContent=`语义查询：${p.semantic_query} · 硬条件 ${p.metadata_filter_must.length} · 优先条件 ${p.metadata_filter_should.length}`;
    loading.appendChild(meta);
  }catch(err){ loading.textContent='发生错误：'+err.message; }
  finally{ button.disabled=false; input.focus(); }
});
input.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();form.requestSubmit();}});
</script></body></html>"""
