#!/bin/bash
# Bring up all four OpenArm CAN channels after a WSL restart or a replug.
#
#   FOLLOWER : usbipd BUSID 1-7  ->  can0 (右/right) , can1 (左/left)
#   LEADER   : usbipd BUSID 1-8  ->  can2 (右/right) , can3 (左/left)
#
# The can* numbering follows ATTACH ORDER, so 1-7 must be attached first.
# Run the two usbipd attach commands on the WINDOWS side first (no admin
# needed); this script only does the Linux half.
#
#   PS> usbipd attach --wsl --busid 1-7
#   PS> usbipd attach --wsl --busid 1-8
#
# Usage:  ./openarm_can_setup.sh [bitrate] [dbitrate]

BITRATE=${1:-1000000}
DBITRATE=${2:-5000000}
FAIL=0

echo "=== bringing up CAN channels (${BITRATE} / ${DBITRATE} FD) ==="
for i in can0 can1 can2 can3; do
    if [ ! -d "/sys/class/net/$i" ]; then
        printf "  %-6s MISSING - run 'usbipd attach' on Windows first\n" "$i"
        FAIL=1
        continue
    fi
    sudo ip link set "$i" down 2>/dev/null
    if sudo ip link set "$i" up type can \
            bitrate "$BITRATE" dbitrate "$DBITRATE" fd on 2>/dev/null; then
        sudo ip link set "$i" txqueuelen 1000
        printf "  %-6s UP\n" "$i"
    else
        printf "  %-6s FAILED to come up\n" "$i"
        FAIL=1
    fi
done

echo
echo "=== mapping ==="
for i in can0 can1 can2 can3; do
    [ -d "/sys/class/net/$i" ] || continue
    usb=$(basename "$(readlink -f "/sys/class/net/$i/device")")
    case "$usb" in
        1-1*) role="FOLLOWER" ;;
        1-2*) role="LEADER  " ;;
        *)    role="UNKNOWN " ;;
    esac
    state=$(ip -details link show "$i" | awk '/can <FD>/{print $4}')
    printf "  %-6s usb=%-10s %s  %s\n" "$i" "$usb" "$role" "$state"
done

echo
if [ "$FAIL" -eq 0 ]; then
    echo "All four channels ready."
    echo
    echo "Teleop uses  leader-if  then  follower-if:"
    echo "  right arm :  ~/openarm_teleop/build/unilateral_control ... can2 can0"
    echo "  left  arm :  ~/openarm_teleop/build/unilateral_control ... can3 can1"
else
    echo "One or more channels are not ready - see above."
    exit 1
fi
