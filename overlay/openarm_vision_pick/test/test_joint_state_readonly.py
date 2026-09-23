"""Run with python3; no ROS or hardware needed."""
import struct
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from openarm_vision_pick.joint_state_readonly import query_frame, decode_frame

for motor in range(1, 9):
    frame = query_frame(motor)
    assert struct.unpack_from('=I', frame)[0] == 0x7ff
    assert frame[8:16] == bytes([motor, 0, 0xcc, 0, 0, 0, 0, 0])
    for value, expected in [(0, -12.5), (65535, 12.5), (32768, 25/65535/2)]:
        data = bytes([motor, value >> 8, value & 255, 0, 0, 0, 0, 0])
        frame = struct.pack('=IB3x8s', motor + 0x10, 8, data)
        decoded = decode_frame(frame)
        assert decoded[0] == motor and abs(decoded[1] - expected) < 1e-10
        assert decode_frame(frame[:8] + bytes([motor | 0x10]) + frame[9:]) is None
assert decode_frame(b'bad') is None
assert decode_frame(struct.pack('=IB3x8s', 0x80000011, 8, bytes([1]*8))) is None
try:
    query_frame(9)
except ValueError:
    pass
else:
    raise AssertionError('Non-arm motor accepted')
print('PASS: only state query 0xCC, position decoding, enabled/error/invalid-frame rejection')
