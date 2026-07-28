# LeakGuard Jetson video stream protocol

The Jetson Nano sends the annotated OpenCV dashboard to the Qt HMI through a
dedicated persistent TCP connection. Status packets remain on port 5001; JPEG
frames use port 5003.

## Transport

- Jetson Nano: TCP client
- Qt HMI: TCP server
- Default destination: Raspberry Pi port `5003`
- Default frame rate: `5 FPS`
- Default image: `640x480`, JPEG quality `65`
- Byte order for all integer header fields: network byte order (big-endian)

The sender keeps only the newest frame. JPEG encoding, socket writes, and
reconnection run on a background thread so a slow or disconnected HMI does not
block U-Net inference.

## Frame format

Each frame is a 24-byte header followed immediately by `jpeg_length` bytes:

| Offset | Size | Type | Meaning |
|---:|---:|---|---|
| 0 | 4 | ASCII | Magic: `LGIM` |
| 4 | 1 | uint8 | Version: `1` |
| 5 | 1 | uint8 | Flags; bit 0 means annotated dashboard |
| 6 | 2 | uint16 | Header size: `24` |
| 8 | 4 | uint32 | Frame sequence, wrapping after `0xffffffff` |
| 12 | 8 | uint64 | Capture/render Unix timestamp in milliseconds |
| 20 | 4 | uint32 | JPEG payload size |
| 24 | N | bytes | JPEG payload |

The receiver must handle split and combined TCP reads. It should reject unknown
versions, headers larger or smaller than 24 bytes, and payloads larger than the
configured safety limit (default `1,000,000` bytes). If synchronization is lost,
scan for the next `LGIM` magic sequence.

## Jetson execution

The normal Raspberry Pi host override applies to both status and video:

```bash
python3 run_realtime.py \
  --server-host 10.10.16.87 \
  --server-port 5000
```

Optional overrides:

```bash
python3 run_realtime.py \
  --server-host 10.10.16.87 \
  --video-port 5003
```

Disable only video while keeping status transmission:

```bash
python3 run_realtime.py --no-video-stream
```
