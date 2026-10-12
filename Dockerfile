FROM python:3.12-slim AS svmlight-build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    mkdir -p /tmp/svm_light; \
    curl -fsSL --retry 3 https://osmot.cs.cornell.edu/svm_light/current/svm_light.tar.gz -o /tmp/svm_light.tar.gz; \
    tar -xzf /tmp/svm_light.tar.gz -C /tmp/svm_light; \
    build_dir="$(find /tmp/svm_light -type f -name Makefile -printf '%h\n' -quit)"; \
    test -n "$build_dir"; \
    make -C "$build_dir" CFLAGS="-O3 -fcommon" LFLAGS="-O3 -fcommon"; \
    install -m 0755 "$build_dir/svm_learn" /usr/local/bin/svm_learn; \
    install -m 0755 "$build_dir/svm_classify" /usr/local/bin/svm_classify; \
    rm -rf /tmp/svm_light /tmp/svm_light.tar.gz

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIRECTORY=/app/output

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py svm_light.py ./
COPY ["Training dataset.xlsx", "./"]
COPY ["Testing dataset.xlsx", "./"]
COPY --from=svmlight-build /usr/local/bin/svm_learn /usr/local/bin/svm_learn
COPY --from=svmlight-build /usr/local/bin/svm_classify /usr/local/bin/svm_classify

CMD ["python", "main.py"]
