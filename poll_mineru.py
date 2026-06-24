import subprocess, json, os, time, shutil

token = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI3NDkwMTg5MiIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc4MTUzMjEzMywiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiMWVmOWIyY2ItOGVhYi00NTUxLTlkYzMtYzE1ZTBhNjU5ZDQ5IiwiZW1haWwiOiIzMDQzNzA4NzQxQHFxLmNvbSIsImV4cCI6MTc4OTMwODEzM30.sA-RMV7yUzHmLeRtY_KcEEsc9K93pl8mnQGtHQA19Lmtu_jvyiSh1771Coj2SI2SLDwSRTeLKq_QNT_-BgAJiA"

base = "/data/share/hxd/haojiang/Papers/paper-reading/topics/VLM-Bottleneck-Analysis-and-Method-Design"

tasks = {
    "778a0f5f-dcd4-45c7-9dbf-be6ec3a503ef": "encoding/[Arxiv 2025] Q-Zoom",
    "97258929-6ab9-4fe8-b18b-936a1fc720c9": "encoding/[Arxiv 2025] CARES",
    "e4556372-3fa1-4bd6-b43d-39c08ee54e96": "encoding/[Arxiv 2026] iGVLM",
    "9253362d-f341-4eb2-9e1e-d065bd243fab": "encoding/[Arxiv 2026] Perceptual-Bandwidth-Bottleneck",
    "94374840-5642-4403-bf7e-6ab9c9962c3f": "grounding/[Arxiv 2026] Vision-aligned-Latent-Reasoning",
    "f7e6031d-157f-4fa8-b1f3-a4701d40ab2a": "hierarchical-design/[Arxiv 2026] Hierarchical-Visual-Cues",
    "3d11697c-1fee-4239-9d96-83549e702539": "invoke/[Arxiv 2026] Iterative-Evidence-Refinement",
    "8ec8d5fc-8ed0-4fe3-b75e-c2ca66b8666e": "invoke/[Arxiv 2026] Thinking-with-Visual-Grounding",
    "fe8d5929-be8a-435f-8f97-52f724c86172": "reward/[Arxiv 2026] RegionReasoner",
    "e52dfd74-844b-4d7c-97a0-487635c5ef6b": "reward/[Arxiv 2025] VisualPRM",
    "ee70b8fd-65a5-44b5-a8a1-221bf88da839": "reward/[Arxiv 2026] Perception-centric-PRM",
    "6037a9ca-9d82-45cf-bd13-dbef8d63a57d": "long-reasoning/[Arxiv 2026] Imagine-Before-Predict",
    "b92cb805-5a19-4660-82d0-c02b4db3ed58": "bottleneck-analysis/[Arxiv 2026] More-Images-More-Problems",
    "19b87f01-4c06-4533-ac5d-d5f10b061ae4": "multi-image/[Arxiv 2026] Dual-Mechanisms-Spatial-Binding",
    "a72b63d1-946c-4917-9bf0-dd16663879c0": "medical/[Arxiv 2026] MedSynapse-V-v2",
}

print("Polling 15 MinerU tasks...")
for i in range(60):
    all_done = True
    for tid, dir_name in tasks.items():
        out_dir = os.path.join(base, dir_name)
        if os.path.exists(os.path.join(out_dir, "full.md")):
            continue
        resp = subprocess.check_output([
            "curl", "-s", f"https://mineru.net/api/v4/extract/task/{tid}",
            "-H", f"Authorization: Bearer {token}"
        ]).decode()
        state = json.loads(resp)["data"].get("state", "?")
        if state == "done":
            url = json.loads(resp)["data"]["full_zip_url"]
            zip_path = os.path.join(out_dir, "result.zip")
            mineru_out = os.path.join(out_dir, "mineru_out")
            subprocess.run(["curl", "-sL", "-o", zip_path, url], check=True)
            subprocess.run(["unzip", "-o", zip_path, "-d", mineru_out], capture_output=True)
            for item in os.listdir(mineru_out):
                src = os.path.join(mineru_out, item)
                dst = os.path.join(out_dir, item)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            shutil.rmtree(mineru_out)
            os.remove(zip_path)
            print(f"  Done: {dir_name} ({os.path.getsize(os.path.join(out_dir, 'full.md'))}B)")
        if state != "done":
            all_done = False
    print(f"[{i+1}] {sum(1 for d in tasks.values() if os.path.exists(os.path.join(base, d, 'full.md')))}/15 done")
    if all_done:
        print("All complete!")
        break
    time.sleep(10)
