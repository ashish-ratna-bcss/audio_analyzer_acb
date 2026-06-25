API Endpoints
Base URL: http://98.86.63.69

Header (all requests): X-API-Key: f379241418da1092837aaa6b7138e850e4b99b1b2f88ba90d527e6a0b4b4a600

Method	Endpoint	Purpose
POST	/cases	Create new case → returns case_id
POST	/cases/{case_id}/files	Upload audio/video file → returns job_id
GET	/jobs/{job_id}	Poll job status + progress
GET	/jobs/{job_id}/result	Full result (transcript, speakers, conversation table)
GET	/cases/{case_id}/files/{file_id}/result	Same result by case+file
GET	/cases/{case_id}	List all files + latest job per file in case
POST	/jobs/{job_id}/rerun	Re-process same file with new job
POST	/cases/{case_id}/files/{file_id}/certify	Mark transcript certified
Test from Laptop (curl)

# Variables
BASE=http://98.86.63.69
KEY=f379241418da1092837aaa6b7138e850e4b99b1b2f88ba90d527e6a0b4b4a600
AUDIO=/path/to/your/file.mp4   # change this

# 1. Create case
CASE=$(curl -s -X POST "$BASE/cases" -H "X-API-Key: $KEY" | python3 -c "import json,sys; print(json.load(sys.stdin)['case_id'])")
echo "case_id: $CASE"

# 2. Upload file
RESP=$(curl -s -X POST "$BASE/cases/$CASE/files" \
  -H "X-API-Key: $KEY" \
  -F "audio=@$AUDIO")
JOB=$(echo $RESP | python3 -c "import json,sys; print(json.load(sys.stdin)['job_id'])")
echo "job_id: $JOB"

# 3. Poll status
curl -s "$BASE/jobs/$JOB" -H "X-API-Key: $KEY" | python3 -m json.tool

# 4. Get result (once status = completed)
curl -s "$BASE/jobs/$JOB/result" -H "X-API-Key: $KEY" | python3 -m json.tool

# 5. Pretty-print conversation table only
curl -s "$BASE/jobs/$JOB/result" -H "X-API-Key: $KEY" | \
  python3 -c "
import json,sys
d=json.load(sys.stdin)
for r in d['conversation_table']['rows']:
    print(f\"{r['sl']:>3}. [{r['time']}] {r['person']:>15}: {r['conversation']}\")"

# 6. Save full result to file
curl -s "$BASE/jobs/$JOB/result" -H "X-API-Key: $KEY" > result.json
Poll Until Done (one-liner)

JOB=<your_job_id>
while true; do
  S=$(curl -s "http://98.86.63.69/jobs/$JOB" -H "X-API-Key: f379241418da1092837aaa6b7138e850e4b99b1b2f88ba90d527e6a0b4b4a600" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['status'], d['stage'], str(d['progress'])+'%')")
  echo $S
  echo $S | grep -q "completed\|failed" && break
  sleep 15
done
Store in Your App

BASE_URL  = http://98.86.63.69
API_KEY   = f379241418da1092837aaa6b7138e850e4b99b1b2f88ba90d527e6a0b4b4a600
Webhook: pass callback_url=https://yourapp.com/webhook on file upload — server POSTs job status on every stage transition, so you don't need to poll.