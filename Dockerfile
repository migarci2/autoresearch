ARG BASE_IMAGE=autoresearch-mnist-base:cuda128
FROM ${BASE_IMAGE}

ARG VCS_REF=dev

ENV AUTORESEARCH_IMAGE_REVISION=${VCS_REF}

WORKDIR /opt/autoresearch/app
COPY . /opt/autoresearch/app

RUN set -eux; \
    mkdir -p /workspace /workspace/autoresearch /workspace/homes /workspace/shared /workspace/worktrees /workspace/swarm_logs /var/run/sshd; \
    chmod 755 /var/run/sshd; \
    chmod +x /opt/autoresearch/app/scripts/runpod_entrypoint.sh

EXPOSE 22 8080

ENTRYPOINT ["/usr/bin/tini", "-s", "--", "/bin/bash", "/opt/autoresearch/app/scripts/runpod_entrypoint.sh"]
CMD ["bash"]
