#!/usr/bin/env bash
# Fixes ydotoold failing with "failed to open uinput device: Permission denied".
#
# Cause: brltty's udev rule (90-brltty-uinput.rules) runs
#   setfacl -m u:brltty:rw /dev/uinput
# on add|change; the resulting ACL leaves the `input` group with no effective
# access (group::---), locking out the ydotoold user service. Removing brltty
# is not an option when qemu-full is installed (it needs qemu-chardev-baum,
# which needs brltty), so this script installs a later udev rule that gives
# the input group its access back, then restarts the daemon.
#
# Usage: sudo ./fix-uinput-access.sh   (idempotent, safe to re-run)
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }
user="${SUDO_USER:?started without sudo}"
uid="$(id -u "$user")"

# 1. Counter-rule: 91-* sorts after 90-brltty-uinput.rules, so this setfacl
#    runs after brltty's on every add|change event and restores group access.
rule=/etc/udev/rules.d/91-uinput-input-group.rules
cat >"$rule" <<'EOF'
# ydotoold/wtf: restore input-group access to uinput after brltty's
# 90-brltty-uinput.rules ACL entry zeroes it (group::---).
KERNEL=="uinput", ACTION=="add|change", RUN+="/usr/bin/setfacl -m g:input:rw /dev/$name"
EOF
echo ">> wrote $rule"

# 2. Reload udev and re-run the rules for uinput (brltty's and ours).
udevadm control --reload
udevadm trigger --action=add /dev/uinput 2>/dev/null || udevadm trigger /dev/uinput
sleep 1
getfacl -p /dev/uinput | grep -v '^#'

# 3. Sanity check: the target user can now open the device for writing.
if runuser -u "$user" -- test -w /dev/uinput; then
	echo ">> $user can write /dev/uinput"
else
	echo "!! $user still cannot write /dev/uinput (member of the input group?)"
	id "$user"
	exit 1
fi

# 4. Restart the user service (it is in start-limit-hit after earlier failures).
userctl() {
	runuser -u "$user" -- env \
		XDG_RUNTIME_DIR="/run/user/$uid" \
		DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$uid/bus" \
		systemctl --user "$@"
}
userctl reset-failed ydotool.service 2>/dev/null || true
userctl enable --now ydotool.service

# 5. Verify the daemon socket exists (glob, not find: /run/user/<uid> contains
#    FUSE mounts such as `doc` that find cannot enter under `set -o pipefail`).
sock=""
for s in "/run/user/$uid"/*ydotool* "/run/user/$uid"/.ydotool*; do
	if [ -S "$s" ]; then
		sock="$s"
		break
	fi
done
if [ -n "$sock" ] && [ -S "$sock" ]; then
	echo ">> ok: $sock"
else
	echo "!! ydotoold socket not found; see journalctl --user -u ydotool.service"
	exit 1
fi

echo ">> done. Test dictation: press the record hotkey, speak, press again."
