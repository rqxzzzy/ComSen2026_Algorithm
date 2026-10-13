#!/bin/bash
# ============================================================
# drone and bird 数据集分片下载器  (v2)
#
# 背景：Google Storage 直连单线程只有 ~20 KB/s（225 MB 要 3 小时），
#       aria2 因 SSL 握手失败完全不能用。
#       但 curl 能通且支持 Range 请求 —— 于是用多路并行分片。
#
# 两个关键技巧：
#  1) 传 Roboflow 原始链接而非签名后的 GCS 链接：
#     Roboflow 每次都重定向出一个「15 分钟有效」的新签名 URL，
#     所以每个 curl 调用都拿到新鲜签名，不会遇到过期的死链。
#  2) 每片的临时文件名必须带片号（chunk_$i.tmp）。
#     若共用同一个 chunk.tmp，16 个并发进程会抢同一个文件互相覆盖，
#     速度反而掉回单线程水平。
# ============================================================
URL="https://app.roboflow.com/ds/nSMQrzUp0X?key=TTcKC8vGXi"
OUT="D:/ComSen2026/task4/datasets/drone_bird.zip"
TMP="D:/ComSen2026/task4/datasets/_parts_v2"
NPART=16
MAX_RETRY=10

mkdir -p "$TMP"

# ---------- 1. 取总大小 ----------
TOTAL=$(curl -sIL --max-time 60 "$URL" | grep -i "^content-length" | tail -1 | tr -d '\r' | awk '{print $2}')
if [ -z "$TOTAL" ]; then echo "[错误] 取不到文件大小"; exit 1; fi
PART=$(( (TOTAL + NPART - 1) / NPART ))
echo "[信息] 总大小 $TOTAL 字节 ($((TOTAL/1048576)) MB)"
echo "[信息] 分 $NPART 片，每片约 $((PART/1048576)) MB，开始并行下载..."

# ---------- 2. 逐片下载（后台并发）----------
for i in $(seq 0 $((NPART-1))); do
  S=$((i * PART))
  E=$((S + PART - 1))
  [ "$E" -ge "$TOTAL" ] && E=$((TOTAL - 1))
  WANT=$((E - S + 1))

  (
    for attempt in $(seq 1 $MAX_RETRY); do
      HAVE=0
      [ -f "$TMP/p$i" ] && HAVE=$(stat -c %s "$TMP/p$i" 2>/dev/null || echo 0)
      if [ "$HAVE" -ge "$WANT" ]; then break; fi

      FROM=$((S + HAVE))
      curl -sL --max-time 900 -r "${FROM}-${E}" -o "$TMP/chunk_$i.tmp" "$URL" 2>/dev/null
      if [ -f "$TMP/chunk_$i.tmp" ]; then
        cat "$TMP/chunk_$i.tmp" >> "$TMP/p$i"
        rm -f "$TMP/chunk_$i.tmp"
      fi
      sleep 1
    done
    HAVE=$(stat -c %s "$TMP/p$i" 2>/dev/null || echo 0)
    echo "  片 $i 完成: $HAVE/$WANT"
  ) &
done

# ---------- 3. 进度监视（含下载中的临时文件）----------
(
  while true; do
    sleep 20
    DONE=$(du -cb "$TMP"/p* "$TMP"/chunk_*.tmp 2>/dev/null | tail -1 | cut -f1)
    [ -z "$DONE" ] && DONE=0
    PCT=$((DONE * 100 / TOTAL))
    echo "[进度] $((DONE/1048576))/$((TOTAL/1048576)) MB  (${PCT}%)"
    [ "$DONE" -ge "$TOTAL" ] && break
  done
) &
WATCHER=$!

wait
kill $WATCHER 2>/dev/null

# ---------- 4. 合并 ----------
echo "[信息] 合并分片..."
rm -f "$OUT"
for i in $(seq 0 $((NPART-1))); do
  cat "$TMP/p$i" >> "$OUT"
done

FINAL=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
echo "========================================"
echo "下载大小: $FINAL 字节"
echo "期望大小: $TOTAL 字节"
if [ "$FINAL" -eq "$TOTAL" ]; then
  echo "结果: 完整 OK"
  rm -rf "$TMP"
else
  echo "结果: 不匹配，分片保留在 $TMP 以便重试"
fi
echo "========================================"
