#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="jcooper23/muse-voice-mcp"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo 'local')"

echo "==> [1/3] Cross-compiling ${IMAGE_NAME}:latest for linux/amd64..."
docker build --platform linux/amd64 \
  -t "${IMAGE_NAME}:latest" \
  -t "${IMAGE_NAME}:${GIT_SHA}" \
  .

echo "==> [2/3] Pushing ${IMAGE_NAME}:latest and ${IMAGE_NAME}:${GIT_SHA} to Docker Hub..."
docker push "${IMAGE_NAME}:latest"
docker push "${IMAGE_NAME}:${GIT_SHA}"

cat <<EOF

==> [3/3] Deployment Instructions (K3s + Longhorn + Cloudflare Tunnel)

1. Create or update the Portainer / Kubernetes Secret (muse-voice-secrets):
   kubectl create secret generic muse-voice-secrets \\
     --namespace default \\
     --from-literal=TELEPHONY_PROVIDER="telnyx" \\
     --from-literal=OUTBOUND_CALLER_NUMBER="+15550001234" \\
     --from-literal=TELNYX_API_KEY="KEY019..." \\
     --from-literal=TELNYX_CONNECTION_ID="253..." \\
     --from-literal=TWILIO_ACCOUNT_SID="" \\
     --from-literal=TWILIO_AUTH_TOKEN="" \\
     --from-literal=MCP_API_KEY="\$(openssl rand -hex 32)" \\
     --from-literal=WEBHOOK_SIGNING_SECRET="\$(openssl rand -hex 32)" \\
     --dry-run=client -o yaml | kubectl apply -f -

2. Apply Longhorn PVC, Deployment, and Services:
   kubectl apply -f k8s/01-pvc.yaml
   kubectl apply -f k8s/02-deployment.yaml
   kubectl apply -f k8s/03-service.yaml
   kubectl rollout restart deployment/muse-voice-mcp -n default

3. Connect Muse MCP Client (Direct Tailscale Mesh / In-Cluster — No Cloudflare Required):
   - Tailscale Mesh URL:  http://100.115.252.55:30890/mcp/sse
   - ClusterIP DNS URL:   http://muse-voice-mcp-svc.default.svc.cluster.local:8090/mcp/sse
   - ClusterIP Direct:    http://10.43.235.143:8090/mcp/sse
EOF
