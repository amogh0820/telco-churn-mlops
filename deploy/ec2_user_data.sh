#!/bin/bash
# EC2 user-data for Amazon Linux 2023.
# Paste this into "Advanced details -> User data" when launching the instance.
set -euxo pipefail

dnf update -y
dnf install -y docker
systemctl enable --now docker
usermod -aG docker ec2-user

# SSM agent is preinstalled on AL2023; make sure it is running so the GitHub
# Action can roll the container without SSH keys or an open port 22.
systemctl enable --now amazon-ssm-agent

# A 1 GB instance has no swap, and pip/docker pulls can OOM without it.
dd if=/dev/zero of=/swapfile bs=1M count=1024
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
