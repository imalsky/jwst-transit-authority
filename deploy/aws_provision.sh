#!/usr/bin/env bash
# Run on the Mac (or anywhere with awscli configured: `aws configure`).
# Provisions ONE Ubuntu 24.04 x86_64 EC2 instance + security group for the
# tool. Alternative with zero CLI setup: Lightsail via the console -- see
# notes.md, Deployment runbooks (AWS runbook), step 2b.
#
# Required env vars:
#   KEY_NAME        existing EC2 key-pair name in the region
#   ALLOW_SSH_CIDR  your IP, e.g. 203.0.113.5/32  (curl -s ifconfig.me)
# Optional:
#   REGION          default us-east-2
#   INSTANCE_TYPE   default m7i-flex.2xlarge (8 vCPU / 32 GiB);
#                   c7i-flex.2xlarge (16 GiB) is cheaper and workable
#   DISK_GB         default 150 (gp3; data ~8 GB + image ~10 GB + caches)
set -euo pipefail

REGION="${REGION:-us-east-2}"
INSTANCE_TYPE="${INSTANCE_TYPE:-m7i-flex.2xlarge}"
DISK_GB="${DISK_GB:-150}"
KEY_NAME="${KEY_NAME:?set KEY_NAME to an existing EC2 key-pair name}"
ALLOW_SSH_CIDR="${ALLOW_SSH_CIDR:?set ALLOW_SSH_CIDR to your.ip.addr.ess/32}"

AMI=$(aws ssm get-parameter --region "$REGION" \
    --name /aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id \
    --query 'Parameter.Value' --output text)
echo "Ubuntu 24.04 AMI in $REGION: $AMI"

SG=$(aws ec2 create-security-group --region "$REGION" \
    --group-name jwst-tool-sg --description "vulcan-jwst-tool server" \
    --query 'GroupId' --output text)
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
    --protocol tcp --port 22 --cidr "$ALLOW_SSH_CIDR"
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
    --protocol tcp --port 80 --cidr 0.0.0.0/0
aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG" \
    --protocol tcp --port 443 --cidr 0.0.0.0/0
echo "Security group: $SG (22 from $ALLOW_SSH_CIDR; 80/443 open, basic-auth gates the app)"

IID=$(aws ec2 run-instances --region "$REGION" \
    --image-id "$AMI" --instance-type "$INSTANCE_TYPE" --key-name "$KEY_NAME" \
    --security-group-ids "$SG" \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$DISK_GB,\"VolumeType\":\"gp3\"}}]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=jwst-tool}]' \
    --query 'Instances[0].InstanceId' --output text)
echo "Instance: $IID (waiting for running state ...)"
aws ec2 wait instance-running --region "$REGION" --instance-ids "$IID"

IP=$(aws ec2 describe-instances --region "$REGION" --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo
echo "Running at $IP"
echo
echo "Recommended: allocate a static IP so DNS/TLS survive restarts:"
echo "  aws ec2 allocate-address --region $REGION --query AllocationId --output text"
echo "  aws ec2 associate-address --region $REGION --instance-id $IID --allocation-id ALLOC_ID"
echo
echo "Next (see notes.md, Deployment runbooks): upload the data bundles, then ssh in and run server_setup.sh"
echo "  ssh -i YOUR_KEY.pem ubuntu@$IP"
