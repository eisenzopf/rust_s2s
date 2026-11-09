#!/usr/bin/env python3
"""Run the rust_s2s GPU tests on a short-lived EC2 GPU instance.

What it does:
  1. Checks that HEAD is pushed (the instance clones that exact commit).
  2. Creates (once) a private S3 bucket for logs, and a presigned PUT URL for
     this run's log. The instance needs no IAM role and no credentials.
  3. Launches one GPU instance (default g6.xlarge, NVIDIA L4) from the AWS
     Deep Learning Base OSS NVIDIA Driver AMI (Ubuntu 24.04) with
     scripts/aws/gpu-userdata.sh. Shutdown behavior is "terminate", and the
     script powers off when done or at a hard deadline, so the instance always
     deletes itself.
  4. Waits for the log, prints it, and terminates the instance explicitly
     (also on errors and Ctrl-C).

Credentials: S2S_AWS_ACCESS_KEY_ID / S2S_AWS_SECRET_ACCESS_KEY (optional
S2S_AWS_SESSION_TOKEN) if set, else the normal AWS credential chain. The
S2S_-prefixed names exist because some sandboxes pre-set AWS_ACCESS_KEY_ID to a
placeholder. Minimal IAM permissions are in scripts/aws/iam-policy.json.

Usage:
  python3 scripts/aws/run_gpu_tests.py [--region us-east-1] [--instance-type g6.xlarge]
  python3 scripts/aws/run_gpu_tests.py --dry-run   # print the plan, call nothing
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import re
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
USERDATA_TEMPLATE = ROOT / "scripts" / "aws" / "gpu-userdata.sh"
REPO_URL = "https://github.com/eisenzopf/rust_s2s.git"
CUDA_RUST_REV = "367bc0f8e671070e00eebc7e5498f3e8eebefc72"
AMI_SSM_PARAM = (
    "/aws/service/deeplearning/ami/x86_64/"
    "base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
)
# cuTile needs sm_80+; refuse instance families that can't run it.
UNSUPPORTED_PREFIXES = ("g4dn.", "g4ad.", "p3.", "p2.", "g3.")
TAG = {"Key": "Project", "Value": "rust_s2s-gpu-tests"}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def pushed_commit() -> str:
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain"):
        print("warning: uncommitted changes are NOT tested (the instance clones the pushed commit)")
    remote = git("branch", "-r", "--contains", commit)
    if not remote:
        sys.exit(f"HEAD {commit[:10]} is not on any remote branch; push it first.")
    return commit


def render_userdata(commit: str, put_url: str, deadline_min: int) -> str:
    text = USERDATA_TEMPLATE.read_text()
    for key, val in {
        "REPO_URL": REPO_URL,
        "COMMIT": commit,
        "RESULT_PUT_URL": put_url,
        "DEADLINE_MIN": str(deadline_min),
        "CUDA_RUST_REV": CUDA_RUST_REV,
    }.items():
        text = text.replace(f"@@{key}@@", val)
    leftover = re.findall(r"@@[A-Z_]+@@", text)
    assert not leftover, f"unsubstituted placeholders in user-data: {leftover}"
    if len(text.encode()) > 16 * 1024:
        sys.exit("user-data exceeds EC2's 16 KB limit")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--instance-type", default="g6.xlarge")
    ap.add_argument("--disk-gb", type=int, default=120)
    ap.add_argument("--deadline-min", type=int, default=120, help="hard power-off deadline on the instance")
    ap.add_argument("--dry-run", action="store_true", help="print what would happen; make no AWS calls")
    args = ap.parse_args()

    if args.instance_type.startswith(UNSUPPORTED_PREFIXES):
        sys.exit(f"{args.instance_type}: GPU is older than sm_80, which cuTile Rust does not support")

    commit = pushed_commit()
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_key = f"runs/{stamp}-{commit[:10]}.log"

    if args.dry_run:
        print(f"[dry-run] region={args.region} type={args.instance_type} commit={commit}")
        print(f"[dry-run] AMI from SSM {AMI_SSM_PARAM}")
        print(f"[dry-run] log -> s3://rust-s2s-gpu-tests-<account>-{args.region}/{log_key}")
        ud = render_userdata(commit, "https://example.invalid/presigned-put", args.deadline_min)
        print(f"[dry-run] user-data: {len(ud.encode())} bytes, deadline {args.deadline_min} min")
        return

    import boto3  # imported late so --dry-run works without boto3
    from botocore.exceptions import ClientError

    creds = {}
    if os.environ.get("S2S_AWS_ACCESS_KEY_ID"):
        creds = {
            "aws_access_key_id": os.environ["S2S_AWS_ACCESS_KEY_ID"],
            "aws_secret_access_key": os.environ["S2S_AWS_SECRET_ACCESS_KEY"],
            "aws_session_token": os.environ.get("S2S_AWS_SESSION_TOKEN") or None,
        }
    session = boto3.Session(region_name=os.environ.get("S2S_AWS_REGION", args.region), **creds)
    args.region = session.region_name
    account = session.client("sts").get_caller_identity()["Account"]
    print(f"AWS account {account}, region {args.region}")

    s3 = session.client("s3")
    bucket = f"rust-s2s-gpu-tests-{account}-{args.region}"
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        kwargs = {"Bucket": bucket}
        if args.region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": args.region}
        s3.create_bucket(**kwargs)
        s3.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={k: True for k in (
                "BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")},
        )
        print(f"created private bucket s3://{bucket}")
    put_url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": bucket, "Key": log_key, "ContentType": "text/plain"},
        ExpiresIn=(args.deadline_min + 30) * 60,
    )

    ami = session.client("ssm").get_parameter(Name=AMI_SSM_PARAM)["Parameter"]["Value"]
    ec2 = session.client("ec2")
    root_dev = ec2.describe_images(ImageIds=[ami])["Images"][0]["RootDeviceName"]
    print(f"AMI {ami} (Deep Learning Base OSS NVIDIA Driver, Ubuntu 24.04)")

    instance_id = None
    try:
        resp = ec2.run_instances(
            ImageId=ami,
            InstanceType=args.instance_type,
            MinCount=1,
            MaxCount=1,
            UserData=render_userdata(commit, put_url, args.deadline_min),
            InstanceInitiatedShutdownBehavior="terminate",
            BlockDeviceMappings=[{
                "DeviceName": root_dev,
                "Ebs": {"VolumeSize": args.disk_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
            }],
            MetadataOptions={"HttpTokens": "required"},
            TagSpecifications=[
                {"ResourceType": "instance", "Tags": [TAG, {"Key": "Name", "Value": f"rust_s2s-gpu-{commit[:8]}"}]},
                {"ResourceType": "volume", "Tags": [TAG]},
            ],
        )
        instance_id = resp["Instances"][0]["InstanceId"]
        print(f"launched {instance_id} ({args.instance_type}); it terminates itself when done "
              f"or after {args.deadline_min} min")

        deadline = time.time() + (args.deadline_min + 10) * 60
        last_state = None
        gone_since = None
        while time.time() < deadline:
            try:
                body = s3.get_object(Bucket=bucket, Key=log_key)["Body"].read().decode(errors="replace")
                print("\n" + body)
                print(f"log saved at s3://{bucket}/{log_key}")
                ok = "S2S_GPU_TESTS_DONE status=all-passed" in body
                sys.exit(0 if ok else 1)
            except ClientError:
                pass
            state = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]["State"]["Name"]
            if state != last_state:
                print(f"[{time.strftime('%H:%M:%S')}] instance {state}")
                last_state = state
            if state in ("terminated", "shutting-down", "stopped"):
                gone_since = gone_since or time.time()
                if time.time() - gone_since > 120:  # a final upload would have landed by now
                    try:
                        out = ec2.get_console_output(InstanceId=instance_id, Latest=True).get("Output", "")
                        print("\n--- instance console (tail) ---\n" + out[-8000:])
                    except ClientError as e:
                        print(f"(console output unavailable: {e})")
                    sys.exit("instance exited without uploading a test log")
            time.sleep(30)
        sys.exit("timed out waiting for the test log")
    finally:
        if instance_id:
            ec2.terminate_instances(InstanceIds=[instance_id])
            print(f"terminate requested for {instance_id}")


if __name__ == "__main__":
    main()
