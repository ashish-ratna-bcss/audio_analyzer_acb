# ACB Audio Analyzer — API Integration Guide

Forensic Telugu/English speech-to-text + speaker diarization. Submit an audio/video
file, get back a full transcription with speaker labels, a diarization timeline,
and a court-ready conversation table.

No human-review gate: a job runs to `completed` on its own and the result is
immediately available. Speaker labels come straight from diarization (pyannote).

---

## 1. Auth

Every endpoint requires the API key header:

```
x-api-key: <API_KEY>
```

The key is the `API_KEY` env var on the server. If unset (local dev), auth is open.

Base URL: `http://<server>/` (nginx → API). HTTPS on `:443` in deployment.

---

## 2. Lifecycle at a glance

```
POST /cases                         -> { case_id }
POST /cases/{case_id}/files         -> { file_id, job_id }   (starts processing)
        |
        |  (poll)  GET /jobs/{job_id}                -> status, stage, progress
        |  (push)  webhook -> your callback_url      -> same payload per stage
        v
GET /jobs/{job_id}/result           -> transcription + diarization + table
POST /jobs/{job_id}/rerun           -> { job_id (new) }       (reprocess same file)
GET  /cases/{case_id}               -> files + latest job (re-hydrate app state)
```

---

## 3. Endpoints

### 3.1 Create a case
`POST /cases`
```json
201 -> { "case_id": "76ae1731-..." }
```
A case groups one or more files (e.g. all recordings for one investigation).

### 3.2 Upload a file (starts processing)
`POST /cases/{case_id}/files`  — `multipart/form-data`

| field | type | required | notes |
|-------|------|----------|-------|
| `audio` | file | yes | audio or video; allowed extensions enforced server-side |
| `separate` | bool | no | run source separation (slower; for overlapped speech) |
| `callback_url` | string | no | webhook target for status events (see §4) |

```json
202 -> { "file_id": "3e2bf95d-...", "job_id": "4c8d4ea3-..." }
```
Errors: `400` unsupported format, `404` unknown case, `413` file too large.

### 3.3 Job status (poll)
`GET /jobs/{job_id}`
```json
200 -> {
  "job_id": "4c8d4ea3-...",
  "case_id": "76ae1731-...",
  "file_id": "3e2bf95d-...",
  "status": "running",          // queued|running|completed|failed|quarantined|certified
  "stage": "L5",                // current pipeline stage (see §5)
  "progress": 75,               // 0..100
  "degraded_flags": [],
  "error": null,
  "is_terminal": false,         // true once status is completed/failed/quarantined/certified
  "result_url": null            // set to /jobs/{id}/result when ready
}
```

### 3.4 Get the result
`GET /jobs/{job_id}/result`  (or `GET /cases/{case_id}/files/{file_id}/result`)

`409` until the job is terminal; `200` when `completed`/`certified`:
```json
{
  "job_id": "...", "case_id": "...", "file_id": "...",
  "status": "completed",
  "source_hash_sha256": "...",                  // integrity: hash of the original
  "transcript": {
    "file_id": "...", "case_id": "...", "status": "completed",
    "segments": [
      { "segment_id": "...", "start": 351.8, "end": 356.3,
        "speaker": "Speaker_1", "overlap": false,
        "text": "ఏది ఇప్పుడు హాస్పిటల్ ...", "language": "te",
        "confidence": 0.84, "source_pass": "indic_conformer",
        "flagged_for_review": false, "review_status": null, "reviewer_id": null }
    ]
  },
  "diarization": {
    "file_id": "...", "speakers": ["Speaker_1", "Speaker_2"],
    "timeline": [ { "start": 322.3, "end": 324.0, "speaker": "Speaker_1" } ],
    "model_version": "pyannote/speaker-diarization-3.1"
  },
  "conversation_table": {
    "file_id": "...",
    "rows": [ { "sl": 1, "time": "05.51", "person": "Speaker_1",
                "conversation": "ఏది ఇప్పుడు ...", "language": "te" } ]
  }
}
```

### 3.5 Re-run (reprocess the same file)
`POST /jobs/{job_id}/rerun`
```json
202 -> { "job_id": "c418c283-... (new)", "case_id": "...", "file_id": "...",
         "rerun_of": "4c8d4ea3-..." }
```
Re-stages the immutable original under a **new** job id. The old job and its
segments are untouched. Same `options` (incl. `callback_url`) carry over. Use
after tuning, or to retry a `failed` job.

### 3.6 List a case (re-hydrate app state)
`GET /cases/{case_id}`
```json
{ "case_id": "...", "files": [
    { "file_id": "...", "original_filename": "BVR_23_02_2021.mp4",
      "latest_job": { ...same shape as GET /jobs/{id}... } } ] }
```

---

## 4. Two ways to track progress

### A) Webhooks (recommended) — supply `callback_url` on upload
The server POSTs this JSON to your URL on **every stage transition and terminal
state** (best-effort; failures never break the job):
```json
{ "job_id": "...", "case_id": "...", "file_id": "...",
  "status": "running", "stage": "L5", "progress": 75, "result_url": null }
```
Final event:
```json
{ "status": "completed", "stage": "L8", "progress": 100,
  "result_url": "/jobs/{job_id}/result" }
```
On completion, call `GET {result_url}`. The `callback_url` must be reachable from
the server (a public URL, or an internal host the API container can route to).

### B) Polling — `GET /jobs/{job_id}` every few seconds
Stop when `is_terminal` is true. Typical cadence: 3–5s.

---

## 5. Stages & progress

| stage | meaning | progress |
|-------|---------|----------|
| L0 | ingest original | 5 |
| L1 | normalize audio | 15 |
| L2 | enhancement (denoise) | 30 |
| L2b | source separation (only if `separate=true`) | 35 |
| L3 | voice-activity detection | 45 |
| L4 | diarization (speakers) | 55 |
| L5 | ASR (transcription) | 75 |
| L6 | confidence report | 85 |
| L8 | output generation | 95 |
| — | `completed` | 100 |

Terminal statuses: `completed` (success), `failed` (error, see `error` field),
`quarantined` (bad/unreadable input), `certified` (optional manual sign-off).

---

## 6. What your app should store

Per uploaded file, persist in your own DB:

| field | from | why |
|-------|------|-----|
| `case_id` | `POST /cases` | groups files; your case record |
| `file_id` | upload response | identifies the recording |
| `job_id` | upload response (and each rerun) | poll status / fetch result |
| `original_filename` | your upload | display |
| last known `status`/`stage` | webhook or poll | UI without re-fetch |

Minimal schema:
```
case(id, ...your case fields...)
recording(id, case_id, file_id, original_filename, current_job_id, status, stage)
```
On rerun, update `current_job_id` to the new job id (keep history if you want an
audit trail). To recheck later, call `GET /jobs/{current_job_id}` or
`GET /cases/{case_id}`.

---

## 7. React / Next.js example

```ts
const API = process.env.NEXT_PUBLIC_API_URL!;       // e.g. https://acb.example.com
const KEY = process.env.API_KEY!;                    // server-side only; proxy in prod
const h = { "x-api-key": KEY };

// 1) Create a case (once per investigation)
export async function createCase() {
  const r = await fetch(`${API}/cases`, { method: "POST", headers: h });
  return (await r.json()).case_id as string;
}

// 2) Upload a file; optionally pass a webhook URL
export async function uploadFile(caseId: string, file: File, callbackUrl?: string) {
  const fd = new FormData();
  fd.append("audio", file);
  if (callbackUrl) fd.append("callback_url", callbackUrl);
  const r = await fetch(`${API}/cases/${caseId}/files`, { method: "POST", headers: h, body: fd });
  return await r.json() as { file_id: string; job_id: string };
}

// 3a) Poll status until terminal
export async function pollJob(jobId: string, onTick: (s: any) => void) {
  while (true) {
    const r = await fetch(`${API}/jobs/${jobId}`, { headers: h });
    const s = await r.json();
    onTick(s);                       // s.status, s.stage, s.progress
    if (s.is_terminal) return s;
    await new Promise(res => setTimeout(res, 4000));
  }
}

// 4) Fetch the result when completed
export async function getResult(jobId: string) {
  const r = await fetch(`${API}/jobs/${jobId}/result`, { headers: h });
  if (r.status === 409) throw new Error("not finished");
  return await r.json();             // { transcript, diarization, conversation_table }
}

// 5) Re-run
export async function rerun(jobId: string) {
  const r = await fetch(`${API}/jobs/${jobId}/rerun`, { method: "POST", headers: h });
  return (await r.json()).job_id as string;
}
```

```tsx
// Render the conversation table
function Transcript({ result }: { result: any }) {
  return (
    <table>
      <thead><tr><th>#</th><th>Time</th><th>Speaker</th><th>Text</th></tr></thead>
      <tbody>
        {result.conversation_table.rows.map((r: any) => (
          <tr key={r.sl}><td>{r.sl}</td><td>{r.time}</td><td>{r.person}</td><td>{r.conversation}</td></tr>
        ))}
      </tbody>
    </table>
  );
}
```

### 3b) Webhook receiver (Next.js Route Handler)
```ts
// app/api/acb-webhook/route.ts
export async function POST(req: Request) {
  const e = await req.json();        // { job_id, status, stage, progress, result_url }
  await db.recording.update({
    where: { jobId: e.job_id },
    data: { status: e.status, stage: e.stage, progress: e.progress },
  });
  // optional: when completed, fetch + store the transcript
  // if (e.status === "completed") await ingestResult(e.job_id);
  return new Response("ok");
}
```
Pass `callback_url: "https://yourapp.com/api/acb-webhook"` to `uploadFile`.

> Keep the API key server-side. In a browser app, proxy these calls through your
> Next.js backend (Route Handlers) rather than exposing `x-api-key` to the client.

---

## 8. Notes & limits

- **Accuracy** (current, BVR test file): region CER ~52%, 2 speakers correctly
  separated, language auto-detected. Output is a strong machine draft; for legal
  certification a human should still verify against the audio. Lower CER needs
  model fine-tuning on a labeled corpus, not API changes.
- **`flagged_for_review`** on a segment = low confidence / low enh-vs-original
  agreement. Surface these for optional human checking; very-low-signal segments
  are auto-blanked (text empty) so the record never carries fabricated words.
- **Languages**: auto-detected, generalized via a file-level language prior — no
  language is hardcoded. Works for any IndicConformer-supported language.
- **Idempotency**: each upload creates a new file+job. Re-uploading the same file
  makes a separate file_id. Use `rerun` to reprocess an existing file.
```
